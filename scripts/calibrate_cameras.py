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

The cube is fitted as a cube (--centre model, the default since 2026-10-10).
Each camera sees only the cube's top and the faces toward it, so the median of
its visible points sits toward that camera and up. The 2026-09-04 solve took
that median as the cube centre and baked the bias into every extrinsic: each
camera pushed its cloud away from itself, and cameras on opposite sides put
the same paper ~36 mm apart. The model fit requires each camera's raw points
to lie on a cube of --cube-size around the TCP instead
(robochem/vision/cube_calibration.py, tested by scripts/test_cube_calibration.py).
--centre median keeps the old solve for comparison; it is always printed.

Cameras are captured one at a time. Streaming all four together has wedged
this host's USB controller (progress.md, 2026-09-25).

Every sweep saves its raw observations, so the solve can be redone without
the robot: --from-observations calibration_out/observations_<time>.npy.

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

    # re-solve a saved sweep, no robot
    python scripts/calibrate_cameras.py --from-observations calibration_out/observations_<time>.npy

    # install the result (backs up the existing calibration)
    python scripts/calibrate_cameras.py --apply
"""

import argparse
import atexit
import json
import os
import shutil
import sys
import time
from datetime import datetime

import cv2
import numpy as np

from robomail.vision.cam_utils import get_cam_info

from frankapy import FrankaArm

from robochem.vision import VisionSystem
from robochem.vision.cube_calibration import (
    fit_cube_model, implied_centres, kabsch, subsample, table_height, table_inliers,
)
from robochem.skills.base_skill import to_rigid_transform

CAM_IDS = [2, 3, 4, 5]
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(REPO, "calibration_out")

# Saturated green against a dark red-brown table. Wide on value so the cube is
# still found in the shaded corners of the cage.
HSV_LOWER = (35, 80, 40)
HSV_UPPER = (85, 255, 255)


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
    Median centroid of the cube in this camera's own frame, plus the inlier
    points themselves, which the cube-model fit uses.

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
        return None, len(pts), 0.0, None

    centroid = np.median(pts, axis=0)

    keep = np.linalg.norm(pts - centroid, axis=1) < args.inlier_radius
    if keep.sum() < args.min_points:
        return None, int(keep.sum()), 0.0, None

    inliers = pts[keep]
    spread = float(np.linalg.norm(inliers.std(axis=0)))
    return np.median(inliers, axis=0), int(keep.sum()), spread, inliers


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

    data = capture(vision)
    detections = detect_all(data["images"], args, vision)

    print(f"\nRobot reports {np.round(fa.get_pose().translation, 4)}\n")
    ok = True
    for cam_id in args.cameras:
        if cam_id not in detections:
            print(f"  cam {cam_id}: NOT DETECTED")
            ok = False
            continue

        mask, area = detections[cam_id]
        centroid, n, spread, _ = cube_centroid(
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


def capture(vision, drain=5):
    """
    capture_pointclouds, after dropping the frames a persistent camera
    queued while the arm was moving.
    """
    loc = vision.object_localizer
    for cam in loc.cameras.values():
        for _ in range(drain):
            cam.get_next_frame(get_point_cloud=False, get_verts=False)
    return loc.capture_pointclouds()


def merge_records(paths):
    """
    Several saved sweeps as one record, e.g. a camera re-swept after it
    dropped out. Valid only if the cube never left the grip in between:
    the cube's offset from the TCP is folded into every camera alike.
    """
    merged = None
    for k, path in enumerate(paths):
        rec = np.load(path, allow_pickle=True).item()
        if merged is None:
            merged = {"cube_size": rec["cube_size"], "orientation": rec["orientation"],
                      "obs": {}, "table": {}}
        elif abs(rec["cube_size"] - merged["cube_size"]) > 1e-6:
            raise ValueError(f"{path}: cube size differs from {paths[0]}")
        for c, obs in rec["obs"].items():
            for o in obs:
                o = dict(o)
                o["pose"] = 1000 * k + o["pose"]     # keep poses of each file apart
                merged["obs"].setdefault(c, []).append(o)
        merged["table"].update(rec.get("table") or {})
    return merged


def keep_cameras_alive(loc, pause=0.3):
    """
    Hold every camera object the sweep opens, and pause after each close.

    The sweep starts and stops each pipeline 4 times a pose. On 2026-10-10 it
    segfaulted inside librealsense after about a minute; one likely cause is
    a closed camera being garbage-collected while its frame threads still
    run. Keeping the objects referenced and giving the device a moment after
    stop() costs ~1 s a pose. A mitigation, not a proven fix: --resume is the
    backstop.
    """
    held = []
    open_camera, close_camera = loc._open_camera, loc._close_camera

    def open_and_hold(cam_id):
        cam, owned = open_camera(cam_id)
        held.append(cam)
        return cam, owned

    def close_and_wait(cam):
        close_camera(cam)
        time.sleep(pause)

    loc._open_camera = open_and_hold
    loc._close_camera = close_and_wait
    return held


def table_points(depth, intr, T, stride=4):
    """Every stride-th pixel of one depth image, in world, for the table check."""
    h, w = depth.shape[:2]
    v, u = np.mgrid[0:h:stride, 0:w:stride]
    v, u = v.ravel(), u.ravel()
    z = depth[v, u].astype(float) / 1000.0
    ok = (z > 0.1) & (z < 2.0)
    v, u, z = v[ok], u[ok], z[ok]
    pts = np.column_stack([(u - intr["cx"]) * z / intr["fx"],
                           (v - intr["cy"]) * z / intr["fy"], z])
    return pts @ T[:3, :3].T + T[:3, 3]


def intr_dict(intr):
    cx = getattr(intr, "cx", getattr(intr, "ppx", None))
    cy = getattr(intr, "cy", getattr(intr, "ppy", None))
    return {"fx": float(intr.fx), "fy": float(intr.fy), "cx": float(cx), "cy": float(cy)}


def to_matrix(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def solve(record, args):
    """
    Solve every camera from a sweep record, print the report, stage the result.

    record: {"cube_size", "orientation" (3x3 world-from-hand), "obs":
    {cam: [{"pose", "tcp", "median", "points", "area"}]}, "table": {cam:
    {"depth", "intr"}}} as captured by the sweep or loaded from disk.
    """
    obs = record["obs"]
    cube = float(record["cube_size"])
    cams = [c for c in args.cameras if len(obs.get(c, [])) >= 6]
    for c in args.cameras:
        if c not in cams:
            print(f"\ncam {c}: only {len(obs.get(c, []))} observations; need >= 6. Skipping.")
    if not cams:
        return {}

    print("\n" + "=" * 60)
    print("SOLVING")
    print("=" * 60)

    # The old solve: the median of the visible points is taken as the centre.
    median_T, results = {}, {}
    print("\nMedian + Kabsch (the 2026-09-04 method; biased toward each camera):")
    obs = dict(obs)
    for c in cams:
        # Drop gross misdetections before anything else. Cam 5's re-sweep had
        # two observations 0.6 m and 1.9 m off among 39 within 20 mm; left in,
        # they wrecked this solve and the cube fit started from it.
        keep = list(obs[c])
        for _ in range(3):
            A = np.array([o["median"] for o in keep]).T
            B = np.array([o["tcp"] for o in keep]).T
            R, t = kabsch(A, B)
            residuals = np.linalg.norm((R @ A + t[:, None]) - B, axis=0)
            limit = max(0.030, 4 * np.median(residuals))
            if (residuals <= limit).all() or (residuals <= limit).sum() < 6:
                break
            keep = [o for o, r in zip(keep, residuals) if r <= limit]
        dropped = sorted(set(o["pose"] for o in obs[c]) - set(o["pose"] for o in keep))
        obs[c] = keep
        median_T[c] = to_matrix(R, t)
        loo = leave_one_out_error(A, B)
        print(f"  cam {c}: {A.shape[1]} obs, residual mean {residuals.mean() * 1000:5.1f} mm, "
              f"held-out mean {loo.mean() * 1000:5.1f} max {loo.max() * 1000:5.1f} mm"
              + (f"; dropped poses {dropped} as misdetections" if dropped else ""))

    final = dict(median_T)
    fit = None
    table = record.get("table") or {}
    flat = {}
    if args.centre == "model" and args.level_table is not None and table:
        # Bare tabletop in each camera's own frame, picked with the median
        # solve; the fit then also holds every camera's table at the height.
        for c in cams:
            if c in table:
                pts = table_points(table[c]["depth"], table[c]["intr"], np.eye(4))
                flat[c] = table_inliers(pts, median_T[c], args.level_table)
        print("\nTable points in the fit: "
              + ", ".join(f"cam {c} {len(p)}" for c, p in sorted(flat.items())))
    lift = 0.0
    if args.centre == "model":
        where = (f"table at z {args.level_table * 1000:+.1f} mm, cube height above the "
                 f"TCP fitted" if flat else "centre = TCP")
        print(f"\nCube model ({cube * 1000:.1f} mm cube, {where}), from the median solve...")
        fit = fit_cube_model({c: obs[c] for c in cams}, np.asarray(record["orientation"]),
                             cube, median_T, table=flat or None,
                             table_z=args.level_table if flat else None)
        final = fit["transforms"]
        if fit["cube_above_tcp_mm"] is not None:
            lift = np.array([0.0, 0.0, fit["cube_above_tcp_mm"] / 1000])
            print(f"  cube centre sits {fit['cube_above_tcp_mm']:+.1f} mm above the TCP")
        print(f"  converged: {fit['success']}; cube sits {fit['cube_tilt_deg']:.1f} deg "
              f"off square in the grip")

    # Per observation: where this camera's points put the cube centre.
    Rc = fit["cube_rotation"] if fit else np.asarray(record["orientation"])
    centres = {}
    print(f"\nPer observation, cube centre each camera implies vs the TCP "
          f"({args.centre} solve):")
    for c in cams:
        errs = []
        for o in obs[c]:
            ctr = implied_centres(o["points"], final[c], Rc, cube, o["tcp"] + lift)
            if ctr is None:
                continue
            centres.setdefault(o["pose"], {})[c] = ctr
            errs.append(np.linalg.norm(ctr - o["tcp"] - lift))
        errs = np.array(errs) * 1000
        rms = f", surface rms {fit['rms_mm'][c]:.1f} mm" if fit else ""
        print(f"  cam {c}: mean {errs.mean():5.1f}  median {np.median(errs):5.1f}  "
              f"max {errs.max():5.1f} mm{rms}")
        results[c] = {"n": len(obs[c]), "centre_err_mean_mm": float(errs.mean()),
                      "centre_err_max_mm": float(errs.max())}
        if fit:
            results[c]["surface_rms_mm"] = fit["rms_mm"][c]

    spreads = []
    for per in centres.values():
        pts = list(per.values())
        if len(pts) >= 2:
            spreads.append(max(np.linalg.norm(a - b) for i, a in enumerate(pts)
                               for b in pts[i + 1:]))
    if spreads:
        spreads = np.array(spreads) * 1000
        print(f"\nCameras agreeing on the cube, per pose (largest pairwise gap): "
              f"median {np.median(spreads):.1f}  max {spreads.max():.1f} mm")

    # The table, from the last capture: the check that needs no ground truth
    # between cameras. With --level-table it is also in the fit, so this
    # then shows how well the fit could hold it there.
    heights = {}
    if table:
        print("\nTable z at (0.525, 0.0), mm: installed / median solve / this solve")
        for c in cams:
            if c not in table:
                continue
            d, intr = table[c]["depth"], table[c]["intr"]
            row = []
            for T in (np.asarray(get_cam_info(c)[1]), median_T[c], final[c]):
                fitp = table_height(table_points(d, intr, T))
                row.append(None if fitp is None else fitp[0] * 1000)
            heights[c] = row[2]
            print("  cam {}: ".format(c) + " / ".join(
                "  n/a" if z is None else f"{z:+6.1f}" for z in row))

        if args.level_table is not None and not flat:
            got = [z for z in heights.values() if z is not None]
            if got:
                dz = args.level_table - np.mean(got) / 1000
                for c in final:
                    final[c] = final[c].copy()
                    final[c][2, 3] += dz
                print(f"\n--level-table {args.level_table}: every camera moved "
                      f"{dz * 1000:+.1f} mm in z (i.e. the cube centre sits "
                      f"{dz * 1000:+.1f} mm above the TCP)")

    dx, dy = args.cube_offset
    if dx or dy:
        # The cube sat (dx, dy) from the TCP, so every camera saw the world
        # shifted by minus that; the fit cannot tell. Same shift for all.
        for c in final:
            final[c] = final[c].copy()
            final[c][0, 3] += dx
            final[c][1, 3] += dy
        print(f"\n--cube-offset: every camera moved ({dx * 1000:+.1f}, {dy * 1000:+.1f}) mm in xy")

    print("\nChange from the installed calibration:")
    for c in cams:
        old = np.asarray(get_cam_info(c)[1])
        shift = np.linalg.norm(final[c][:3, 3] - old[:3, 3])
        dR = final[c][:3, :3] @ old[:3, :3].T
        angle = np.degrees(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1)))
        print(f"  cam {c}: {shift * 1000:.1f} mm / {angle:.2f} deg")
        results[c].update({"shift_from_current_mm": float(shift * 1000),
                           "rotation_change_deg": float(angle),
                           "table_z_mm": heights.get(c)})
        np.save(os.path.join(OUT_DIR, f"realsense_camera{c}w.npy"), final[c])

    report = {
        "timestamp": datetime.now().isoformat(),
        "centre": args.centre,
        "cube_size_m": cube,
        "cube_tilt_deg": fit["cube_tilt_deg"] if fit else None,
        "level_table": args.level_table,
        "cube_offset_xy": list(args.cube_offset),
        "pose_gap_median_mm": float(np.median(spreads)) if len(spreads) else None,
        "pose_gap_max_mm": float(np.max(spreads)) if len(spreads) else None,
        "cameras": {str(c): r for c, r in results.items()},
    }
    with open(os.path.join(OUT_DIR, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nNew extrinsics written to {OUT_DIR}")
    return final


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
    parser.add_argument("--cube-size", type=float, default=0.0398,
                        help="Cube edge, metres (the green cube measured 39.8 mm, 2026-10-10)")
    parser.add_argument("--centre", choices=["model", "median"], default="model",
                        help="model fits the cube's shape (default); median is the old, "
                             "camera-biased solve")
    parser.add_argument("--level-table", type=float, default=None, metavar="Z",
                        help="Move every camera together in z so the table reads Z. "
                             "Only if the table really is at Z in the base frame")
    parser.add_argument("--resume", default=None, metavar="NPY",
                        help="Continue a sweep that stopped part-way, from its saved "
                             "observations. The cube must still be in the same grip")
    parser.add_argument("--cube-offset", nargs=2, type=float, default=[0.0, 0.0],
                        metavar=("DX", "DY"),
                        help="Where the cube centre sat from the TCP during the sweep, "
                             "base frame, metres. The sweep cannot see it; every camera "
                             "moves by it. 2026-10-10: point_at.py put the fingertips "
                             "3-5 mm toward the base on every object, so +0.004 0")
    parser.add_argument("--from-observations", nargs="+", default=None, metavar="NPY",
                        help="Re-solve saved sweeps; no robot, no cameras. Several files "
                             "are merged: only valid if the cube stayed in the same grip")
    # Wider and lower than the 2026-09-04 sweep (0.42-0.62, +-0.20, 0.10-0.25)
    # so less of the table is extrapolated. The bench under it must be clear.
    parser.add_argument("--x-range", nargs=2, type=float, default=[0.38, 0.66])
    parser.add_argument("--y-range", nargs=2, type=float, default=[-0.25, 0.25])
    # Capped at 0.25: above roughly 0.30 the cube leaves cameras 2 and 5's view.
    parser.add_argument("--z-range", nargs=2, type=float, default=[0.07, 0.25])
    parser.add_argument("--nx", type=int, default=4)
    parser.add_argument("--ny", type=int, default=4)
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

    os.makedirs(OUT_DIR, exist_ok=True)

    if args.from_observations:
        record = merge_records(args.from_observations)
        print(f"Re-solving {' + '.join(args.from_observations)}: "
              + ", ".join(f"cam {c} {len(v)} obs" for c, v in sorted(record["obs"].items())))
        final = solve(record, args)
        if args.apply and final:
            install(list(final))
        elif final:
            print("Not installed. Re-run with --apply to install.")
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

    # Several cameras: capture_pointclouds opens, reads and closes one at a
    # time. A single camera streams for the whole run instead: alone it
    # cannot load the USB controller, and never stopping it avoids the
    # start/stop cycling after which cam 5 delivered no frames from pose 12
    # on (2026-10-10).
    cameras = {}
    if len(args.cameras) == 1:
        import robomail.vision as vis
        cameras = {args.cameras[0]: vis.CameraClass(cam_number=args.cameras[0])}
        atexit.register(lambda: [c.stop_pipeline() for c in cameras.values()])
    vision = VisionSystem(cameras=cameras, grounding_url=grounding_url)
    vision.object_localizer.camera_ids = list(args.cameras)
    if not cameras:
        keep_cameras_alive(vision.object_localizer)

    print("Connecting to FrankaArm...")
    fa = FrankaArm()

    if not args.skip_grasp and not args.resume:
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
    if args.resume:
        saved = args.resume
        record = np.load(saved, allow_pickle=True).item()
        orientation = np.asarray(record["orientation"])
        poses = [np.asarray(p) for p in record["poses"]]
        print(f"Resuming {saved}: {len(record['done'])} of {len(poses)} poses done")
    else:
        saved = os.path.join(OUT_DIR, f"observations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npy")
        orientation = reference.rotation.copy()
        record = {"cube_size": args.cube_size, "orientation": orientation,
                  "poses": [np.asarray(p) for p in poses], "done": [],
                  "obs": {c: [] for c in args.cameras}, "table": {}}
    print(f"Observations are saved after every pose to {saved}")
    skipped = []
    data = None

    print(f"\nSweeping {len(poses)} poses")
    for i, target in enumerate(poses):
        if i in record["done"]:
            continue
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

        data = capture(vision)
        detections = detect_all(data["images"], args, vision)

        if not detections:
            print("  cube not detected in any camera")
            skipped.append((i, "no detection"))
            record["done"].append(i)
            np.save(saved, record, allow_pickle=True)
            continue

        for cam_id, (mask, area) in sorted(detections.items()):
            if area > args.max_area_px:
                print(f"  cam {cam_id}: mask {area}px too large, rejecting")
                continue
            if area < args.min_area_px:
                print(f"  cam {cam_id}: mask {area}px too small, rejecting")
                continue

            centroid, n, spread, inliers = cube_centroid(
                vision, mask, data["depth_images"][cam_id],
                data["intrinsics"][cam_id], args,
            )
            if centroid is None:
                print(f"  cam {cam_id}: only {n} usable depth points, rejecting")
                continue

            record["obs"][cam_id].append({
                "pose": i, "tcp": actual, "median": centroid, "area": area,
                "points": subsample(inliers, 400, seed=i),
            })
            print(f"  cam {cam_id}: area={area:6d}px pts={n:5d} "
                  f"spread={spread * 1000:5.1f}mm cam_frame={np.round(centroid, 4)}")

        # Saved now, so a crash (librealsense segfaulted mid-sweep on
        # 2026-10-10) costs only a --resume.
        record["done"].append(i)
        np.save(saved, record, allow_pickle=True)

    if skipped:
        print(f"\nSkipped poses: {skipped}")

    # The last capture doubles as the table check.
    if data is None:
        data = capture(vision)
    if data is not None:
        for c in args.cameras:
            if c in data["depth_images"] and c in data["intrinsics"]:
                record["table"][c] = {"depth": data["depth_images"][c],
                                      "intr": intr_dict(data["intrinsics"][c])}

    np.save(saved, record, allow_pickle=True)
    print(f"\nRaw observations saved to {saved} (re-solve with --from-observations)")

    final = solve(record, args)

    if args.apply and final:
        install(list(final))
    elif final:
        print("Not installed. Re-run with --from-observations ... --apply to install.")

    print("\nNext: python scripts/view_pointcloud.py --per-camera, and the agreement check.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
