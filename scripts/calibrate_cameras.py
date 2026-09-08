"""
Board-free camera cage calibration.

Solves each static camera's camera-to-world transform without a checkerboard,
using the robot itself as the reference. The gripper holds a small coloured
cube; the arm translates to a spread of poses; at each pose we pair the robot's
reported end-effector position with the cube's centroid as each camera sees it.
Kabsch/SVD then recovers the transform.

Why the grasp offset never has to be measured: the gripper orientation is
frozen for the whole sweep, so the cube sits at a constant offset from the end
effector in world coordinates. Solving R*A + t = B folds that offset into t.
The resulting extrinsics therefore map "what the camera sees" directly onto
"the end-effector pose that puts the gripper around it" -- which is exactly the
quantity a top-down grasp needs, so the offset cancels at pick time instead of
becoming a bias. The tradeoff is that this ties the calibration to the
orientation used here; a very different approach angle keeps a small residual.

The cube must stay small in frame. A large mask means the detector latched onto
the gripper or the table, and its centroid would slide relative to the gripper
as the arm moves, breaking the rigid-attachment assumption.

Detection defaults to HSV colour thresholding rather than the open-vocabulary
grounding service. On this rig the text-prompted detector reliably expanded its
box to the whole white gripper assembly (mask area swung 14x across poses),
which would have quietly poisoned the solve with gripper centroids. Thresholding
a saturated cube against a dark table is deterministic and held a 1.6x area
spread, all of it explained by camera distance.

Nothing is overwritten unless --apply is passed, and the previous calibration
is backed up first.

Usage:
    source scripts/env.sh

    # preview the pose sweep without moving anything
    python scripts/calibrate_cameras.py --dry-run

    # check the cube is detected everywhere before committing to the sweep
    python scripts/calibrate_cameras.py --check

    # run the sweep and report accuracy
    python scripts/calibrate_cameras.py

    # install the result (backs up the existing calibration)
    python scripts/calibrate_cameras.py --apply
"""

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime

import cv2
import numpy as np

import robomail.vision as vis
from robomail.vision.cam_utils import get_cam_info

from frankapy import FrankaArm

from robochem.vision import VisionSystem
from robochem.skills.base_skill import to_rigid_transform

CAM_IDS = [2, 3, 4, 5]
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO, "calibration_out")

# Saturated green against a dark red-brown table. Wide on value so the cube is
# still found in the shaded corners of the cage.
HSV_LOWER = (35, 80, 40)
HSV_UPPER = (85, 255, 255)


def kabsch(A: np.ndarray, B: np.ndarray):
    """
    Least-squares rigid transform taking A onto B.

    Args:
        A: 3xN points in the camera frame
        B: 3xN corresponding points in the robot frame

    Returns:
        (R, t) with R @ A + t ~= B
    """
    ca = A.mean(axis=1, keepdims=True)
    cb = B.mean(axis=1, keepdims=True)

    H = (A - ca) @ (B - cb).T
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Guard against the reflection solution SVD can return.
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = cb - R @ ca
    return R, t.ravel()


def detect_cube_hsv(bgr, lower, upper):
    """
    Largest saturated-green blob, as a boolean mask.

    Returns (mask, area). Keeping only the largest connected component drops
    stray green pixels elsewhere in the cage.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    raw = cv2.inRange(hsv, np.array(lower, np.uint8), np.array(upper, np.uint8))
    raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    count, labels, stats, _ = cv2.connectedComponentsWithStats(raw, 8)
    if count <= 1:
        return None, 0

    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == best, int(stats[best, cv2.CC_STAT_AREA])


def detect_all(images, args, vision=None):
    """
    Detect the cube in every camera. Returns {cam_id: (mask, area)}.

    HSV by default; the grounding service is kept as an escape hatch for a
    fiducial that has no clean colour signature.
    """
    if args.detector == "grounding":
        cam_ids = sorted(images)
        masks, _ = vision.scene_analyzer.segment_object(
            [images[c] for c in cam_ids], args.object, cameras=cam_ids
        )
        return {c: (m, int(m.sum())) for c, m in masks.items()}

    out = {}
    for cam_id, image in images.items():
        mask, area = detect_cube_hsv(image, args.hsv_lower, args.hsv_upper)
        if mask is not None:
            out[cam_id] = (mask, area)
    return out


def cube_centroid(vision, mask, depth, intr, args):
    """
    Median centroid of the cube in this camera's own frame.

    Camera frame, not world: the world transform is what we are solving for.

    The mask is eroded first because depth on a silhouette edge interpolates
    between the cube and whatever is metres behind it, and a handful of those
    pixels drag the centroid backwards. Median plus a radius gate then removes
    what erosion misses.
    """
    eroded = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8),
                       iterations=args.erode)
    if eroded.sum() < args.min_points:
        eroded = mask.astype(np.uint8)  # too small to erode; use as-is

    pts = vision.object_localizer._depth_to_points(depth, eroded.astype(bool), intr)
    if len(pts) < args.min_points:
        return None, len(pts), 0.0

    centroid = np.median(pts, axis=0)

    keep = np.linalg.norm(pts - centroid, axis=1) < args.inlier_radius
    if keep.sum() < args.min_points:
        return None, int(keep.sum()), 0.0

    inliers = pts[keep]
    spread = float(np.linalg.norm(inliers.std(axis=0)))
    return np.median(inliers, axis=0), int(keep.sum()), spread


def build_pose_grid(args):
    """
    A spread of positions; needs variation on all three axes to condition the
    solve. Serpentine ordering keeps consecutive moves short.
    """
    xs = np.linspace(args.x_range[0], args.x_range[1], args.nx)
    ys = np.linspace(args.y_range[0], args.y_range[1], args.ny)
    zs = np.linspace(args.z_range[0], args.z_range[1], args.nz)

    poses = []
    for k, z in enumerate(zs):
        y_order = ys if k % 2 == 0 else ys[::-1]
        for j, y in enumerate(y_order):
            x_order = xs if j % 2 == 0 else xs[::-1]
            for x in x_order:
                poses.append(np.array([x, y, z]))
    return poses


def leave_one_out_error(A, B):
    """
    Honest accuracy estimate: refit without each observation and measure how far
    the fit lands from the point it never saw. Fit residuals alone understate
    error because six parameters can absorb a lot of noise.
    """
    n = A.shape[1]
    if n < 6:
        return None

    errors = []
    for i in range(n):
        keep = [j for j in range(n) if j != i]
        R, t = kabsch(A[:, keep], B[:, keep])
        errors.append(np.linalg.norm(R @ A[:, i] + t - B[:, i]))
    return np.array(errors)


def install(cam_ids):
    """
    Copy staged extrinsics over the live calibration, backing up what was there.

    get_cam_info builds paths internally, so locate calib/ the same way it does:
    relative to the cam_utils module.
    """
    import robomail.vision.cam_utils as cu
    calib_dir = os.path.join(os.path.dirname(cu.__file__), "calib")

    backup = os.path.join(
        calib_dir, "Past_Calibrations",
        f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    os.makedirs(backup, exist_ok=True)

    for cam_id in cam_ids:
        name = f"realsense_camera{cam_id}w.npy"
        staged = os.path.join(OUT_DIR, name)
        if not os.path.exists(staged):
            print(f"  cam {cam_id}: nothing staged in {OUT_DIR}, skipping")
            continue

        live = os.path.join(calib_dir, name)
        if os.path.exists(live):
            shutil.copy(live, os.path.join(backup, name))
        shutil.copy(staged, live)
        print(f"  cam {cam_id}: installed")

    print(f"\nInstalled into {calib_dir}")
    print(f"Previous calibration backed up to {backup}")


def run_check(fa, vision, args):
    """Single-pose detection check across all cameras, before the real sweep."""
    centre = np.array([
        float(np.mean(args.x_range)),
        float(np.mean(args.y_range)),
        float(np.mean(args.z_range)),
    ])
    print(f"Moving to the centre of the sweep volume: {np.round(centre, 3)}")

    pose = to_rigid_transform(fa.get_pose())
    pose.translation = centre
    fa.goto_pose(pose, duration=5)
    time.sleep(args.settle)

    data = vision.object_localizer.capture_pointclouds()
    detections = detect_all(data["images"], args, vision)

    print(f"\nRobot reports {np.round(fa.get_pose().translation, 4)}\n")
    ok = True
    for cam_id in args.cameras:
        if cam_id not in detections:
            print(f"  cam {cam_id}: NOT DETECTED")
            ok = False
            continue

        mask, area = detections[cam_id]
        centroid, n, spread = cube_centroid(
            vision, mask, data["depth_images"][cam_id],
            data["intrinsics"][cam_id], args,
        )
        flag = ""
        if area > args.max_area_px:
            flag, ok = "  <-- too large, detector grabbed more than the cube", False
        elif area < args.min_area_px:
            flag, ok = "  <-- too small to trust", False
        elif centroid is None:
            flag, ok = f"  <-- only {n} usable depth points", False

        loc = np.round(centroid, 4) if centroid is not None else None
        print(f"  cam {cam_id}: area={area:6d}px pts={n:5d} spread={spread * 1000:5.1f}mm "
              f"cam_frame={loc}{flag}")

    print("\n" + ("All cameras see the cube. Ready for the sweep."
                  if ok else "Fix the above before sweeping."))
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--detector", choices=["hsv", "grounding"], default="hsv",
                        help="hsv thresholds a coloured cube (default); grounding uses "
                             "the text-prompted service")
    parser.add_argument("--object", default="green cube",
                        help="Text prompt, only used with --detector grounding")
    parser.add_argument("--hsv-lower", nargs=3, type=int, default=list(HSV_LOWER))
    parser.add_argument("--hsv-upper", nargs=3, type=int, default=list(HSV_UPPER))
    parser.add_argument("--cameras", nargs="+", type=int, default=CAM_IDS)
    parser.add_argument("--x-range", nargs=2, type=float, default=[0.42, 0.62])
    parser.add_argument("--y-range", nargs=2, type=float, default=[-0.20, 0.20])
    # Capped at 0.25: above roughly 0.30 the cube leaves cameras 2 and 5's view.
    parser.add_argument("--z-range", nargs=2, type=float, default=[0.10, 0.25])
    parser.add_argument("--nx", type=int, default=3)
    parser.add_argument("--ny", type=int, default=3)
    parser.add_argument("--nz", type=int, default=3)
    parser.add_argument("--min-points", type=int, default=30,
                        help="Minimum mask points for a usable observation")
    parser.add_argument("--min-area-px", type=int, default=250,
                        help="Reject masks smaller than this as unreliable")
    parser.add_argument("--max-area-px", type=int, default=12000,
                        help="Reject masks larger than this; a 4cm cube spans a few "
                             "thousand pixels at cage range, so a huge mask means the "
                             "detector latched onto the gripper or table")
    parser.add_argument("--erode", type=int, default=2,
                        help="Mask erosion iterations, to avoid depth edge bleed")
    parser.add_argument("--inlier-radius", type=float, default=0.05,
                        help="Metres from the median a point may sit and still count")
    parser.add_argument("--settle", type=float, default=0.6,
                        help="Seconds to wait after each move before capturing")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the pose sweep and exit without moving")
    parser.add_argument("--check", action="store_true",
                        help="Detect the cube at one central pose, then exit")
    parser.add_argument("--apply", action="store_true",
                        help="Install the new extrinsics (backs up the old ones)")
    parser.add_argument("--install-only", action="store_true",
                        help="Install what is already staged in calibration_out "
                             "without re-running the sweep")
    parser.add_argument("--skip-grasp", action="store_true",
                        help="Assume the object is already gripped")
    args = parser.parse_args()

    if args.install_only:
        print(f"Installing staged extrinsics for cameras {args.cameras}")
        install(args.cameras)
        return 0

    poses = build_pose_grid(args)

    if args.dry_run:
        print(f"Pose sweep: {len(poses)} positions")
        for i, p in enumerate(poses):
            print(f"  {i:2d}: {np.round(p, 3)}")
        print("\n(dry run: nothing moved)")
        return 0

    grounding_url = os.environ.get("GROUNDING_URL")
    if args.detector == "grounding" and not grounding_url:
        print("GROUNDING_URL not set; run 'source scripts/env.sh'")
        return 1

    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Opening cameras {args.cameras}...")
    cameras = {c: vis.CameraClass(cam_number=c) for c in args.cameras}
    vision = VisionSystem(cameras=cameras, grounding_url=grounding_url)
    vision.object_localizer.camera_ids = list(args.cameras)

    print("Connecting to FrankaArm...")
    fa = FrankaArm()

    try:
        if not args.skip_grasp:
            fa.open_gripper()
            print("\n" + "=" * 60)
            print("Place the cube between the gripper fingers.")
            input("Press Enter once it is positioned...")
            print("=" * 60)
            fa.close_gripper()

        width = fa.get_gripper_width()
        print(f"Gripper width: {width:.4f} m")
        if width < 0.005:
            print("Gripper is closed on nothing. Aborting.")
            return 1

        if args.check:
            return run_check(fa, vision, args)

        # Orientation is frozen for the whole sweep so the cube keeps a
        # constant offset from the end effector.
        reference = fa.get_pose()
        orientation = reference.rotation.copy()

        observations = {c: {"cam": [], "rob": [], "area": []} for c in args.cameras}
        skipped = []

        print(f"\nSweeping {len(poses)} poses")
        for i, target in enumerate(poses):
            print(f"\n--- pose {i + 1}/{len(poses)}: {np.round(target, 3)} ---")

            pose = to_rigid_transform(reference)
            pose.rotation = orientation
            pose.translation = target

            try:
                fa.goto_pose(pose, duration=4)
            except Exception as e:
                print(f"  move failed: {e}")
                skipped.append((i, "move failed"))
                continue

            time.sleep(args.settle)

            actual = fa.get_pose().translation.copy()
            print(f"  robot reports: {np.round(actual, 4)}")

            data = vision.object_localizer.capture_pointclouds()
            detections = detect_all(data["images"], args, vision)

            if not detections:
                print("  cube not detected in any camera")
                skipped.append((i, "no detection"))
                continue

            for cam_id, (mask, area) in sorted(detections.items()):
                if area > args.max_area_px:
                    print(f"  cam {cam_id}: mask {area}px too large, rejecting")
                    continue
                if area < args.min_area_px:
                    print(f"  cam {cam_id}: mask {area}px too small, rejecting")
                    continue

                centroid, n, spread = cube_centroid(
                    vision, mask, data["depth_images"][cam_id],
                    data["intrinsics"][cam_id], args,
                )
                if centroid is None:
                    print(f"  cam {cam_id}: only {n} usable depth points, rejecting")
                    continue

                observations[cam_id]["cam"].append(centroid)
                observations[cam_id]["rob"].append(actual)
                observations[cam_id]["area"].append(area)
                print(f"  cam {cam_id}: area={area:6d}px pts={n:5d} "
                      f"spread={spread * 1000:5.1f}mm cam_frame={np.round(centroid, 4)}")

        # ------------------------------------------------------------ solve
        print("\n" + "=" * 60)
        print("SOLVING")
        print("=" * 60)

        results = {}
        for cam_id in args.cameras:
            A = np.array(observations[cam_id]["cam"]).T
            B = np.array(observations[cam_id]["rob"]).T

            if A.size == 0 or A.shape[1] < 6:
                got = 0 if A.size == 0 else A.shape[1]
                print(f"\ncam {cam_id}: only {got} observations; need >= 6. Skipping.")
                continue

            R, t = kabsch(A, B)
            residuals = np.linalg.norm((R @ A + t[:, None]) - B, axis=0)
            loo = leave_one_out_error(A, B)

            transform = np.eye(4)
            transform[:3, :3] = R
            transform[:3, 3] = t

            old = np.asarray(get_cam_info(cam_id)[1])
            shift = np.linalg.norm(transform[:3, 3] - old[:3, 3])
            # Rotation difference as a single angle, via the trace identity.
            dR = R @ old[:3, :3].T
            angle = np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))

            areas = observations[cam_id]["area"]
            print(f"\ncam {cam_id}: {A.shape[1]} observations, "
                  f"mask area {min(areas)}-{max(areas)}px")
            print(f"  fit residual   mean={residuals.mean() * 1000:6.1f} mm  "
                  f"median={np.median(residuals) * 1000:6.1f} mm  "
                  f"max={residuals.max() * 1000:6.1f} mm")
            if loo is not None:
                print(f"  held-out error mean={loo.mean() * 1000:6.1f} mm  "
                      f"median={np.median(loo) * 1000:6.1f} mm  "
                      f"max={loo.max() * 1000:6.1f} mm")
            print(f"  differs from current calibration by "
                  f"{shift * 1000:.1f} mm / {angle:.2f} deg")

            results[cam_id] = {
                "transform": transform,
                "n": int(A.shape[1]),
                "residual_mean_mm": float(residuals.mean() * 1000),
                "residual_median_mm": float(np.median(residuals) * 1000),
                "residual_max_mm": float(residuals.max() * 1000),
                "heldout_mean_mm": None if loo is None else float(loo.mean() * 1000),
                "heldout_max_mm": None if loo is None else float(loo.max() * 1000),
                "shift_from_current_mm": float(shift * 1000),
                "rotation_change_deg": float(angle),
            }

            np.save(os.path.join(OUT_DIR, f"realsense_camera{cam_id}w.npy"), transform)

        if skipped:
            print(f"\nSkipped poses: {skipped}")

        report = {
            "timestamp": datetime.now().isoformat(),
            "detector": args.detector,
            "poses_attempted": len(poses),
            "cameras": {
                str(c): {k: v for k, v in r.items() if k != "transform"}
                for c, r in results.items()
            },
        }
        with open(os.path.join(OUT_DIR, "report.json"), "w") as f:
            json.dump(report, f, indent=2)

        print(f"\nNew extrinsics written to {OUT_DIR}")

        # ------------------------------------------------------------ apply
        if args.apply and results:
            install(list(results))
        elif results:
            print("Not installed. Re-run with --apply to install.")

        print("\nNext: re-run scripts/diagnose_grounding.py to confirm the")
        print("cameras now agree on the cup's position.")

        return 0

    finally:
        for cam in cameras.values():
            try:
                cam.stop_pipeline()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
