"""
End-to-end calibration validation: place the cube, look at it, pick it back up.

Three independent checks, in increasing strictness:

1. Table plane agreement. With the arm retracted, every camera reconstructs the
   tabletop and we compare the plane each one reports. Cameras that disagree on
   a large flat surface they can all see are miscalibrated relative to each
   other, and this needs no ground truth at all.

2. Cross-camera object agreement. The cube is placed on the table, the arm
   retracts clear of the cage, and each camera independently localises it. The
   spread between cameras is the quantity that was ~50 mm before recalibration.

3. Regrasp. The full pipeline localises the cube and the arm is commanded to
   pick it up. Gripper width afterwards says whether it actually closed on the
   cube, which tests perception, calibration and the skill together.

Placement is perception-guided rather than a drop from a guessed height: the
cube's underside is measured against the fitted table plane and the arm lowers
until the gap is a few millimetres, so the cube settles instead of bouncing to
an unknown spot.

Usage:
    source scripts/env.sh

    python scripts/validate_calibration.py --table-only
    python scripts/validate_calibration.py
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

import robomail.vision as vis

from frankapy import FrankaArm

from robochem.vision.object_localizer import ObjectLocalizer
from robochem.skills.base_skill import to_rigid_transform

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from calibrate_cameras import detect_cube_hsv, HSV_LOWER, HSV_UPPER  # noqa: E402

CAM_IDS = [2, 3, 4, 5]

# Somewhere the arm is clear of the table and out of the cameras' way.
RETRACT = np.array([0.32, 0.0, 0.45])


def world_points(loc, cam_id, mask, depth, intr):
    """Masked pixels as world-frame points, through the production path."""
    pts_cam = loc._depth_to_points(depth, mask, intr)
    if len(pts_cam) == 0:
        return np.empty((0, 3))
    extr = loc.cameras[cam_id].get_cam_extrinsics()
    return loc._transform_points(pts_cam, extr)


def fit_table(points, workspace, iters=200, thresh=0.006):
    """
    Plane fit over the tabletop, returned as (z_at_workspace_centre, tilt_deg, n).

    Restricted to the workspace footprint first so the cage walls and whatever
    is behind them cannot win the fit. Plain RANSAC on three-point samples.
    """
    (x0, x1), (y0, y1) = workspace
    keep = ((points[:, 0] > x0) & (points[:, 0] < x1) &
            (points[:, 1] > y0) & (points[:, 1] < y1))
    pts = points[keep]
    if len(pts) < 200:
        return None

    rng = np.random.default_rng(0)
    best_inliers, best_plane = None, None

    for _ in range(iters):
        idx = rng.choice(len(pts), 3, replace=False)
        a, b, c = pts[idx]
        n = np.cross(b - a, c - a)
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n = n / norm
        d = -n @ a

        inliers = np.abs(pts @ n + d) < thresh
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers, best_plane = inliers, (n, d)

    if best_plane is None or best_inliers.sum() < 200:
        return None

    # Refit on the inliers via least squares for a stabler normal.
    inl = pts[best_inliers]
    centroid = inl.mean(axis=0)
    _, _, Vt = np.linalg.svd(inl - centroid)
    n = Vt[2]
    n = n if n[2] > 0 else -n
    d = -n @ centroid

    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    z_centre = -(n[0] * cx + n[1] * cy + d) / n[2]
    tilt = np.degrees(np.arccos(np.clip(n[2], -1, 1)))
    return float(z_centre), float(tilt), int(best_inliers.sum())


def check_table(loc, args):
    """Check 1: do the cameras agree on the tabletop?"""
    print("\n" + "=" * 68)
    print("CHECK 1: table plane agreement")
    print("=" * 68)

    data = loc.capture_pointclouds()
    workspace = (tuple(args.x_range), tuple(args.y_range))

    zs = {}
    for cam_id in args.cameras:
        if cam_id not in data["depth_images"]:
            continue
        depth = data["depth_images"][cam_id]
        allpix = np.ones(depth.shape, bool)
        pts = world_points(loc, cam_id, allpix, depth,
                           data["intrinsics"][cam_id])
        fit = fit_table(pts, workspace)
        if fit is None:
            print(f"  cam {cam_id}: could not fit a plane")
            continue
        z, tilt, n = fit
        zs[cam_id] = z
        print(f"  cam {cam_id}: table z = {z * 1000:7.1f} mm   "
              f"tilt from level = {tilt:5.2f} deg   inliers = {n}")

    if len(zs) < 2:
        print("\nNot enough cameras fitted to compare.")
        return None

    spread = max(zs.values()) - min(zs.values())
    print(f"\n  cross-camera table z spread: {spread * 1000:.1f} mm")
    print("  " + ("GOOD" if spread < 0.01 else
                  "MARGINAL" if spread < 0.02 else "POOR"))
    return float(np.median(list(zs.values())))


def locate_cube(loc, args, label):
    """Per-camera world centroid of the cube. Returns {cam_id: centroid}."""
    data = loc.capture_pointclouds()
    out = {}
    for cam_id in args.cameras:
        if cam_id not in data["images"]:
            continue
        mask, area = detect_cube_hsv(data["images"][cam_id],
                                     args.hsv_lower, args.hsv_upper)
        if mask is None or area < 150:
            print(f"  cam {cam_id}: cube not detected ({label})")
            continue

        eroded = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8), 1)
        if eroded.sum() < 30:
            eroded = mask.astype(np.uint8)

        pts = world_points(loc, cam_id, eroded.astype(bool),
                           data["depth_images"][cam_id],
                           data["intrinsics"][cam_id])
        if len(pts) < 30:
            print(f"  cam {cam_id}: only {len(pts)} points ({label})")
            continue

        centroid = np.median(pts, axis=0)
        keep = np.linalg.norm(pts - centroid, axis=1) < 0.05
        out[cam_id] = {
            "centroid": np.median(pts[keep], axis=0),
            "bottom_z": float(np.percentile(pts[keep][:, 2], 5)),
            "area": area,
            "n": int(keep.sum()),
        }
    return out


def report_agreement(per_cam):
    """Spread of the per-camera centroids: the headline number."""
    if len(per_cam) < 2:
        print("  fewer than two cameras localised the cube")
        return None

    for cam_id, v in sorted(per_cam.items()):
        print(f"  cam {cam_id}: world = {np.round(v['centroid'], 4)}  "
              f"area = {v['area']:5d}px  pts = {v['n']:5d}")

    cents = np.array([v["centroid"] for _, v in sorted(per_cam.items())])
    consensus = np.median(cents, axis=0)
    devs = np.linalg.norm(cents - consensus, axis=1)

    print(f"\n  consensus position : {np.round(consensus, 4)}")
    print(f"  per-axis spread    : {np.round(cents.std(axis=0) * 1000, 1)} mm")
    print(f"  worst camera off by: {devs.max() * 1000:.1f} mm")
    print("  " + ("GOOD" if devs.max() < 0.01 else
                  "MARGINAL" if devs.max() < 0.025 else "POOR"))
    return consensus


def place_cube(fa, loc, args, table_z):
    """
    Lower the held cube until its underside nearly touches, then release.

    Height comes from perception rather than a guess about how the cube sits in
    the fingers, which is the unknown this whole exercise avoids measuring.
    """
    print("\n" + "=" * 68)
    print("PLACING THE CUBE")
    print("=" * 68)

    target = np.array([args.place_xy[0], args.place_xy[1], 0.20])
    pose = to_rigid_transform(fa.get_pose())
    pose.translation = target
    fa.goto_pose(pose, duration=5)
    time.sleep(args.settle)

    held = locate_cube(loc, args, "held")
    if not held:
        print("Cube not visible while held; cannot place safely.")
        return None

    bottoms = [v["bottom_z"] for v in held.values()]
    bottom = float(np.median(bottoms))
    gap = bottom - table_z
    print(f"  cube underside at z = {bottom * 1000:.1f} mm, "
          f"table at {table_z * 1000:.1f} mm, gap = {gap * 1000:.1f} mm")

    descend = gap - args.release_gap
    if descend <= 0:
        print("  already at or below the table; not descending")
    else:
        if descend > args.max_descend:
            print(f"  refusing to descend {descend * 1000:.0f} mm "
                  f"(limit {args.max_descend * 1000:.0f} mm)")
            return None
        print(f"  descending {descend * 1000:.1f} mm")
        pose = to_rigid_transform(fa.get_pose())
        pose.translation = fa.get_pose().translation - np.array([0, 0, descend])
        fa.goto_pose(pose, duration=4)
        time.sleep(args.settle)

    print("  opening gripper")
    fa.open_gripper()
    time.sleep(0.5)

    placed_ee = fa.get_pose().translation.copy()
    pose = to_rigid_transform(fa.get_pose())
    pose.translation = placed_ee + np.array([0, 0, 0.12])
    fa.goto_pose(pose, duration=4)

    pose = to_rigid_transform(fa.get_pose())
    pose.translation = RETRACT
    fa.goto_pose(pose, duration=5)
    time.sleep(args.settle)
    print(f"  arm retracted to {np.round(fa.get_pose().translation, 3)}")

    return placed_ee


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cameras", nargs="+", type=int, default=CAM_IDS)
    parser.add_argument("--hsv-lower", nargs=3, type=int, default=list(HSV_LOWER))
    parser.add_argument("--hsv-upper", nargs=3, type=int, default=list(HSV_UPPER))
    parser.add_argument("--x-range", nargs=2, type=float, default=[0.40, 0.65])
    parser.add_argument("--y-range", nargs=2, type=float, default=[-0.20, 0.20])
    parser.add_argument("--place-xy", nargs=2, type=float, default=[0.52, 0.0],
                        help="Where to set the cube down")
    parser.add_argument("--release-gap", type=float, default=0.003,
                        help="Metres between cube underside and table at release")
    parser.add_argument("--max-descend", type=float, default=0.20,
                        help="Refuse to descend further than this in one move")
    parser.add_argument("--settle", type=float, default=0.8)
    parser.add_argument("--table-only", action="store_true",
                        help="Run the table plane check and stop")
    parser.add_argument("--no-regrasp", action="store_true",
                        help="Place and measure, but do not attempt the pick")
    args = parser.parse_args()

    print(f"Opening cameras {args.cameras}...")
    cameras = {c: vis.CameraClass(cam_number=c) for c in args.cameras}

    # ObjectLocalizer directly, not VisionSystem: these checks segment the cube
    # by colour, so there is no reason to require the grounding service to be up.
    loc = ObjectLocalizer(cameras=cameras, camera_ids=list(args.cameras))

    print("Connecting to FrankaArm...")
    fa = FrankaArm()

    try:
        width = fa.get_gripper_width()
        print(f"Gripper width: {width:.4f} m")

        if not args.table_only:
            print("Retracting the arm clear of the table...")
            pose = to_rigid_transform(fa.get_pose())
            pose.translation = RETRACT
            fa.goto_pose(pose, duration=5)
            time.sleep(args.settle)

        table_z = check_table(loc, args)
        if args.table_only or table_z is None:
            return 0 if table_z is not None else 1

        if width < 0.005:
            print("\nGripper is empty; skipping the place step.")
            return 1

        placed_ee = place_cube(fa, loc, args, table_z)
        if placed_ee is None:
            return 1

        print("\n" + "=" * 68)
        print("CHECK 2: cross-camera agreement on the placed cube")
        print("=" * 68)
        per_cam = locate_cube(loc, args, "placed")
        consensus = report_agreement(per_cam)
        if consensus is None:
            return 1

        offset = consensus[:2] - placed_ee[:2]
        print(f"\n  consensus xy vs release xy: {np.round(offset * 1000, 1)} mm "
              f"({np.linalg.norm(offset) * 1000:.1f} mm)")
        print("  (a few mm is the cube settling on release, not a calibration error)")

        if args.no_regrasp:
            return 0

        print("\n" + "=" * 68)
        print("CHECK 3: regrasp through the full pipeline")
        print("=" * 68)
        print(f"  commanding a grasp at {np.round(consensus, 4)}")

        pose = to_rigid_transform(fa.get_pose())
        pose.translation = np.array([consensus[0], consensus[1],
                                     consensus[2] + 0.10])
        fa.goto_pose(pose, duration=5)

        fa.open_gripper()
        pose = to_rigid_transform(fa.get_pose())
        pose.translation = consensus.copy()
        fa.goto_pose(pose, duration=4)
        time.sleep(0.4)

        fa.close_gripper()
        time.sleep(0.5)
        final = fa.get_gripper_width()
        print(f"\n  gripper width after closing: {final:.4f} m")

        if 0.025 < final < 0.05:
            print("  GRASP SUCCEEDED: holding the 4 cm cube")
            pose = to_rigid_transform(fa.get_pose())
            pose.translation = consensus + np.array([0, 0, 0.12])
            fa.goto_pose(pose, duration=4)
            lifted = fa.get_gripper_width()
            print(f"  width after lifting 12 cm: {lifted:.4f} m")
            print("  " + ("still held; end-to-end validation passed"
                          if lifted > 0.025 else "dropped during the lift"))
            return 0

        print("  GRASP FAILED: gripper closed on "
              + ("nothing" if final < 0.025 else "something too wide"))
        return 1

    finally:
        for cam in cameras.values():
            try:
                cam.stop_pipeline()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
