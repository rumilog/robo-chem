"""
Read-only perception diagnostic.

Opens the static cage cameras, captures a frame from each, saves the color
images, and checks the assumptions robochem's vision layer makes about the
robomail camera API. Does not move the robot.

Usage:
    source scripts/env.sh
    python scripts/diagnose_perception.py
"""

import os
import sys

import numpy as np
import cv2

import robomail.vision as vis

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "diag_out")
CAM_IDS = [2, 3, 4, 5]


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)

    cameras = {}
    for cam_id in CAM_IDS:
        print(f"\n=== camera {cam_id} ===")
        try:
            cam = vis.CameraClass(cam_number=cam_id)
        except Exception as e:
            print(f"  FAILED to open: {e}")
            continue
        cameras[cam_id] = cam
        print(f"  serial: {cam.get_cam_serial()}")

        frame = cam.get_next_frame()
        print(f"  get_next_frame() returned {len(frame)} values")

        color, depth, pc, verts = frame
        print(f"  color {color.shape} {color.dtype}   depth {depth.shape} {depth.dtype}")
        print(f"  depth nonzero: {np.count_nonzero(depth)}/{depth.size}"
              f"  range(mm): {depth[depth > 0].min() if np.any(depth) else 0}"
              f"-{depth.max()}")
        print(f"  pointcloud points: {len(pc.points) if pc is not None else None}")

        intr = cam.get_cam_intrinsics()
        print(f"  intrinsics type: {type(intr).__name__}")
        for attr in ["fx", "fy", "cx", "cy", "ppx", "ppy"]:
            print(f"    .{attr}: {getattr(intr, attr, '<MISSING>')}")

        extr = cam.get_cam_extrinsics()
        print(f"  extrinsics type: {type(extr).__name__} shape="
              f"{getattr(extr, 'shape', None)}")

        # Cameras stream bgr8, so this is already BGR and imwrite is correct.
        cv2.imwrite(os.path.join(OUT_DIR, f"cam{cam_id}_color.png"), color)

    print("\n=== robochem API assumptions ===")
    if cameras:
        cam = next(iter(cameras.values()))
        print(f"  has get_intrinsics()  : {hasattr(cam, 'get_intrinsics')}   "
              f"(object_localizer calls this)")
        print(f"  has get_cam_intrinsics(): {hasattr(cam, 'get_cam_intrinsics')}")

    print("\n=== Vision3D fusion ===")
    try:
        v3d = vis.Vision3D()
        print(f"  Vision3D OK; fuse method present: "
              f"{hasattr(v3d, 'unnormalize_fuse_point_clouds_no_base')}")
        print(f"  available fuse methods: "
              f"{[m for m in dir(v3d) if 'fuse' in m.lower()]}")
    except Exception as e:
        print(f"  Vision3D FAILED: {e}")

    for cam in cameras.values():
        cam.stop_pipeline()

    print(f"\nImages written to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
