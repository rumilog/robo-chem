"""
Capture color + depth stills from the static cage cameras (2-5).

Opens one camera at a time to avoid wedging the shared USB controller.
Writes a timestamped folder under scene_captures/ for offline benchmarks
(fake camera readings for collaborators).

Usage:
    source scripts/env.sh
    python scripts/capture_scene.py
    python scripts/capture_scene.py --label after_pick
    python scripts/capture_scene.py --cameras 2 4
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime

import cv2
import robomail.vision as vis

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_ROOT = os.path.join(ROOT, "scene_captures")
CAM_IDS = [2, 3, 4, 5]
WARMUP = 15


def grab(cam_id: int, warmup: int = WARMUP):
    cam = vis.CameraClass(cam_number=cam_id)
    try:
        for _ in range(warmup):
            cam.get_next_frame(get_point_cloud=False, get_verts=False)
        color, depth, _, _ = cam.get_next_frame(
            get_point_cloud=False, get_verts=False
        )
        return color.copy(), depth.copy()
    finally:
        cam.stop_pipeline()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--label",
        default="scene",
        help="Folder name prefix (default: scene). Example: after_pick",
    )
    parser.add_argument(
        "--cameras",
        type=int,
        nargs="+",
        default=CAM_IDS,
        help="Camera IDs to capture (default: 2 3 4 5)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=WARMUP,
        help="Frames to discard for auto-exposure settle (default: 15)",
    )
    args = parser.parse_args()

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(OUT_ROOT, f"{args.label}_{stamp}")
    os.makedirs(out, exist_ok=True)
    print(f"Saving to {out}")

    for cam_id in args.cameras:
        print(f"Opening cam {cam_id}...")
        try:
            color, depth = grab(cam_id, warmup=args.warmup)
        except Exception as e:
            print(f"  FAILED cam {cam_id}: {type(e).__name__}: {e}")
            continue

        color_path = os.path.join(out, f"cam{cam_id}_color.png")
        depth_path = os.path.join(out, f"cam{cam_id}_depth.png")
        viz_path = os.path.join(out, f"cam{cam_id}_depth_viz.png")

        cv2.imwrite(color_path, color)
        cv2.imwrite(depth_path, depth)
        depth_viz = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(
            "uint8"
        )
        depth_viz = cv2.applyColorMap(depth_viz, cv2.COLORMAP_TURBO)
        cv2.imwrite(viz_path, depth_viz)
        print(f"  wrote cam{cam_id}_color.png {color.shape}")
        time.sleep(0.5)

    with open(os.path.join(out, "README.txt"), "w") as f:
        f.write(
            "Workstation scene stills for offline perception benchmarking.\n"
            f"Captured: {stamp}\n"
            f"Label: {args.label}\n"
            "Cameras: RealSense static cage cams (848x480 BGR + Z16 depth).\n"
            "Files:\n"
            "  cam{N}_color.png     - BGR color frame\n"
            "  cam{N}_depth.png     - raw 16-bit depth (mm)\n"
            "  cam{N}_depth_viz.png - colorized depth for quick viewing\n"
        )

    print(f"DONE {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
