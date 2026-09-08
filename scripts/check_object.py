"""
Read-only pre-flight: where do the cameras think an object is?

Segments a named object with the grounding service, reconstructs it from each
camera independently, and reports how much the cameras disagree. Nothing moves,
so this is safe to run before committing the arm to a grasp.

Cameras are opened one at a time by default, not all four at once. The cage
cameras share a single USB controller on this machine, and four simultaneous
848x480 depth+colour streams is roughly 1.6 Gbps plus ~2.8 A of bus power, which
has twice destabilised the host. A cup on a table is not going anywhere between
captures, so sequential capture costs a few seconds and removes the risk. Pass
--simultaneous to open them all together if you need a synchronised snapshot of
something that moves.

Disagreement between cameras is the number that matters. Each camera's
reconstruction is independent, so if they land on the same point in world
coordinates the extrinsics are consistent; if they scatter, the calibration is
wrong regardless of how confident the detector was.

Usage:
    source scripts/env.sh
    python scripts/check_object.py --object "white paper cup"
    python scripts/check_object.py --object "white paper cup" --cameras 4 5
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

import robomail.vision as vis
from robomail.vision.cam_utils import get_cam_info

from robochem.vision.object_localizer import ObjectLocalizer, get_live_intrinsics
from robochem.vision.grounding_client import GroundingClient

CAM_IDS = [2, 3, 4, 5]
OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "diag_out"
)


def grab(cam_id, warmup, retries=2):
    """
    Open one camera, let exposure settle, take a frame, close it again.

    The warmup matters: the first frames after a pipeline start are exposed for
    whatever the sensor last saw, and a black frame segments into nothing.
    """
    for attempt in range(retries + 1):
        cam = None
        try:
            cam = vis.CameraClass(cam_number=cam_id)
            for _ in range(warmup):
                cam.get_next_frame(get_point_cloud=False, get_verts=False)
            img, depth, _, _ = cam.get_next_frame(get_point_cloud=False,
                                                  get_verts=False)
            intr = get_live_intrinsics(cam)
            coverage = float((depth > 0).mean())
            if coverage < 0.05:
                print(f"  cam {cam_id}: depth almost empty "
                      f"({coverage * 100:.1f}% valid) - depth stream may be dead")
            else:
                print(f"  cam {cam_id}: depth {coverage * 100:.0f}% valid, "
                      f"median {np.median(depth[depth > 0]) / 1000:.2f} m")
            return img.copy(), depth.copy(), intr
        except Exception as e:
            print(f"  cam {cam_id}: attempt {attempt + 1} failed: "
                  f"{type(e).__name__}: {e}")
            if attempt == retries:
                return None, None, None
            time.sleep(1.5)
        finally:
            if cam is not None:
                try:
                    cam.stop_pipeline()
                except Exception:
                    pass
    return None, None, None


def capture_sequential(cam_ids, warmup):
    """One camera at a time. Keeps peak USB load to a single stream."""
    frames = {}
    for cam_id in cam_ids:
        print(f"  opening cam {cam_id}...", flush=True)
        img, depth, intr = grab(cam_id, warmup)
        if img is None:
            print(f"  cam {cam_id}: SKIPPED")
            continue
        frames[cam_id] = (img, depth, intr)
        print(f"  cam {cam_id}: captured {img.shape[1]}x{img.shape[0]}")
    return frames


def capture_simultaneous(cam_ids, warmup):
    """All cameras open at once. Higher USB load; only for moving scenes."""
    cams, frames = {}, {}
    try:
        for cam_id in cam_ids:
            cams[cam_id] = vis.CameraClass(cam_number=cam_id)
        for _ in range(warmup):
            for cam in cams.values():
                cam.get_next_frame(get_point_cloud=False, get_verts=False)
        for cam_id, cam in cams.items():
            img, depth, _, _ = cam.get_next_frame(get_point_cloud=False,
                                                  get_verts=False)
            frames[cam_id] = (img.copy(), depth.copy(), get_live_intrinsics(cam))
    finally:
        for cam in cams.values():
            try:
                cam.stop_pipeline()
            except Exception:
                pass
    return frames


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--object", default="white paper cup")
    parser.add_argument("--cameras", nargs="+", type=int, default=CAM_IDS)
    parser.add_argument("--warmup", type=int, default=12,
                        help="Frames to discard so auto-exposure can settle")
    parser.add_argument("--simultaneous", action="store_true",
                        help="Open all cameras at once (higher USB load)")
    parser.add_argument("--inlier-radius", type=float, default=0.08,
                        help="Metres from the median a point may sit and count")
    parser.add_argument("--min-points", type=int, default=30)
    parser.add_argument("--gripper-width", type=float, default=0.08,
                        help="Maximum gripper opening in metres")
    parser.add_argument("--grasp-margin", type=float, default=0.008,
                        help="Clearance to leave on the gripper opening")
    parser.add_argument("--min-grasp-height", type=float, default=0.015,
                        help="Metres above the object's base to keep the fingers "
                             "clear of the table")
    parser.add_argument("--save-points", default="diag_out/fused_points.npy",
                        help="Where to write the fused cloud for offline analysis")
    args = parser.parse_args()

    grounding_url = os.environ.get("GROUNDING_URL")
    if not grounding_url:
        print("GROUNDING_URL not set; run 'source scripts/env.sh'")
        return 1

    client = GroundingClient(grounding_url)
    print(f"grounding service: {client.health()}")

    mode = "simultaneous" if args.simultaneous else "sequential"
    print(f"\nCapturing from {args.cameras} ({mode})")
    frames = (capture_simultaneous(args.cameras, args.warmup) if args.simultaneous
              else capture_sequential(args.cameras, args.warmup))

    if not frames:
        print("\nNo cameras captured.")
        return 1

    os.makedirs(OUT_DIR, exist_ok=True)
    images = {c: f[0] for c, f in frames.items()}

    # Save the raw frames unconditionally. When a camera fails to detect or has
    # no usable depth, the picture is the only way to tell why.
    for cam_id, img in images.items():
        cv2.imwrite(os.path.join(OUT_DIR, f"raw_cam{cam_id}.png"), img)

    print(f"\nSegmenting {args.object!r} with the grounding service...")
    masks, confs = client.segment(images, args.object)
    if not masks:
        print("  NOT DETECTED in any camera")
        return 1

    # ObjectLocalizer with no cameras: only its pure reconstruction helpers are
    # used here, since the frames were already captured and the pipelines closed.
    loc = ObjectLocalizer(cameras={}, camera_ids=[])

    print(f"\n{'cam':>4} {'conf':>6} {'area':>8} {'pts':>6}  world position")
    print("-" * 62)
    centroids = {}
    all_points = []
    for cam_id in sorted(masks):
        mask = masks[cam_id]
        _, depth, intr = frames[cam_id]

        eroded = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8), 1)
        if eroded.sum() < args.min_points:
            eroded = mask.astype(np.uint8)

        pts_cam = loc._depth_to_points(depth, eroded.astype(bool), intr)
        if len(pts_cam) < args.min_points:
            print(f"{cam_id:>4}  only {len(pts_cam)} depth points, skipping")
            continue

        pts = loc._transform_points(pts_cam, np.asarray(get_cam_info(cam_id)[1]))
        centre = np.median(pts, axis=0)
        keep = np.linalg.norm(pts - centre, axis=1) < args.inlier_radius
        if keep.sum() < args.min_points:
            print(f"{cam_id:>4}  only {int(keep.sum())} inliers, skipping")
            continue

        centroids[cam_id] = np.median(pts[keep], axis=0)
        all_points.append(pts[keep])
        print(f"{cam_id:>4} {confs.get(cam_id) or 0:6.3f} {int(mask.sum()):8d} "
              f"{int(keep.sum()):6d}  {np.round(centroids[cam_id], 4)}")

        overlay = images[cam_id].copy()
        overlay[mask] = (0.45 * overlay[mask]
                         + 0.55 * np.array([0, 0, 255])).astype(np.uint8)
        cv2.imwrite(os.path.join(OUT_DIR, f"check_cam{cam_id}.png"), overlay)

    if len(centroids) < 2:
        print("\nFewer than two cameras localised it; cannot cross-check.")
        return 1

    stack = np.array([centroids[c] for c in sorted(centroids)])
    consensus = np.median(stack, axis=0)
    devs = np.linalg.norm(stack - consensus, axis=1)

    print(f"\nconsensus position : {np.round(consensus, 4)}")
    print(f"per-axis spread    : {np.round(stack.std(axis=0) * 1000, 1)} mm")
    print(f"worst camera off by: {devs.max() * 1000:.1f} mm")

    print(f"\nNote: for a hollow or convex object each camera reconstructs only the")
    print(f"surface it can see, so some per-camera spread is shape, not calibration.")
    print(f"Check whether the offsets are radially organised before blaming extrinsics.")

    fused = np.vstack(all_points)
    if args.save_points:
        np.save(args.save_points, fused)
        print(f"\nfused cloud saved to {args.save_points} "
              f"({len(fused)} points) for offline grasp analysis")

    report_fused(fused, args)
    return 0


def report_fused(points, args):
    """
    Geometry of the merged reconstruction, which is what a grasp acts on.

    Fusing across cameras fills in the surfaces any single view misses, so the
    footprint is far closer to the object's true cross-section than one camera's
    silhouette. Width is measured in horizontal slices because tapered objects
    (a paper cup being the obvious case) are narrow enough to grasp lower down
    even when the rim is wider than the gripper can open.
    """
    lo, hi = points.min(axis=0), points.max(axis=0)
    print("\n" + "=" * 62)
    print("FUSED GEOMETRY (all cameras merged)")
    print("=" * 62)
    print(f"  points        : {len(points)}")
    print(f"  centroid      : {np.round(np.median(points, axis=0), 4)}")
    print(f"  bounding box  : {np.round((hi - lo) * 1000, 1)} mm (x, y, z)")
    print(f"  z from {lo[2] * 1000:.1f} mm to {hi[2] * 1000:.1f} mm "
          f"(height {(hi[2] - lo[2]) * 1000:.1f} mm)")

    print(f"\n  {'slice z (mm)':>14} {'pts':>6} {'narrow':>8} {'wide':>8} "
          f"{'yaw':>6}  {'centre xy':>22}")
    print("  " + "-" * 70)

    limit = args.gripper_width - args.grasp_margin
    edges = np.linspace(lo[2], hi[2], 7)
    best = None
    for a, b in zip(edges[:-1], edges[1:]):
        sl = points[(points[:, 2] >= a) & (points[:, 2] < b)]
        if len(sl) < 20:
            continue
        narrow, wide, yaw = width_by_direction(sl[:, :2])
        centre = np.median(sl[:, :2], axis=0)
        fits = narrow < limit
        print(f"  {a * 1000:6.0f}-{b * 1000:<7.0f} {len(sl):6d} {narrow * 1000:7.1f} "
              f"{wide * 1000:7.1f} {yaw:5.0f}d  {str(np.round(centre, 4)):>22}"
              + ("" if fits else "   too wide"))
        # Among slices that fit, take the one with the most room to spare, but
        # require some height above the object's base so the fingers do not
        # reach for the table. On a tapered cup the rim technically fits and is
        # the worst choice: it is the widest point and the easiest to crush.
        z_mid = float((a + b) / 2)
        if fits and z_mid >= lo[2] + args.min_grasp_height:
            clearance = limit - narrow
            if best is None or clearance > best[4]:
                best = (z_mid, centre, narrow, yaw, clearance)

    print()
    if best is None:
        print(f"  Nothing narrower than {limit * 1000:.0f} mm at any height.")
        print(f"  This object will not fit the gripper. Do not attempt a grasp.")
        return

    z, centre, narrow, yaw, _ = best
    print(f"  Suggested grasp: xy = {np.round(centre, 4)}, z = {z:.4f} m, "
          f"yaw = {yaw:.0f} deg")
    print(f"  Object is {narrow * 1000:.1f} mm across that way; gripper opens to "
          f"{args.gripper_width * 1000:.0f} mm "
          f"({(args.gripper_width - narrow) * 1000:.1f} mm clearance).")
    print(f"  Closing along the narrow axis, so the {yaw:.0f} deg heading matters:")
    print(f"  cross-axis error is absorbed by the finger width, but error along")
    print(f"  the closing direction pushes the object instead of gripping it.")


def width_by_direction(xy, step_deg=5, lo_pct=3, hi_pct=97):
    """
    Narrowest and widest horizontal extent of a cross-section, and the heading
    of the narrow axis in degrees.

    The gripper can yaw, so the relevant width is along its closing direction,
    and we are free to pick the direction where the object is thinnest. Extent
    is a percentile range rather than max-minus-min: peak-to-peak is set by the
    two most extreme points, so a few stray reconstruction points inflate it.
    """
    centred = xy - np.median(xy, axis=0)
    widths = []
    for deg in np.arange(0, 180, step_deg):
        theta = np.radians(deg)
        proj = centred @ np.array([np.cos(theta), np.sin(theta)])
        widths.append((np.percentile(proj, hi_pct) - np.percentile(proj, lo_pct),
                       float(deg)))
    narrow, yaw = min(widths)
    wide = max(w for w, _ in widths)
    return float(narrow), float(wide), yaw


if __name__ == "__main__":
    sys.exit(main())
