"""
Per-camera reconstruction diagnostic.

The masks can be perfect while the 3D output is still wrong, because each
camera's depth is unprojected with its own intrinsics and extrinsics. This
prints what each camera independently thinks the object's position and size
are, so a bad calibration shows up as one camera reporting an implausible
extent rather than as a fusion problem.

Usage:
    source scripts/env.sh
    python scripts/diagnose_grounding.py --object "white paper cup"
"""

import argparse
import os
import sys

import numpy as np

import robomail.vision as vis

from robochem.vision import VisionSystem

CAM_IDS = [2, 3, 4, 5]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object", default="white paper cup")
    parser.add_argument("--cameras", nargs="+", type=int, default=CAM_IDS)
    args = parser.parse_args()

    grounding_url = os.environ.get("GROUNDING_URL")
    if not grounding_url:
        print("GROUNDING_URL not set; run 'source scripts/env.sh'")
        return 1

    cameras = {c: vis.CameraClass(cam_number=c) for c in args.cameras}
    vision = VisionSystem(cameras=cameras, grounding_url=grounding_url)
    vision.object_localizer.camera_ids = list(args.cameras)

    try:
        print("\n=== camera intrinsics in use ===")
        for cam_id in args.cameras:
            intr = cameras[cam_id].get_cam_intrinsics()
            ratio = intr.fx / intr.fy if intr.fy else float("nan")
            flag = "  <-- implausible fx/fy" if not (0.8 < ratio < 1.25) else ""
            print(f"  cam {cam_id}: fx={intr.fx:8.2f} fy={intr.fy:8.2f} "
                  f"cx={intr.cx:8.2f} cy={intr.cy:8.2f} fx/fy={ratio:.2f}{flag}")

        data = vision.object_localizer.capture_pointclouds()
        cam_ids = [c for c in args.cameras if c in data["images"]]
        masks, confidences = vision.scene_analyzer.segment_object(
            [data["images"][c] for c in cam_ids], args.object, cameras=cam_ids
        )

        if not masks:
            print(f"\n{args.object!r} not detected in any camera")
            return 1

        by_cam = vision.object_localizer.get_object_points_by_camera(
            masks, depth_images=data["depth_images"], intrinsics=data["intrinsics"]
        )

        print(f"\n=== per-camera reconstruction of {args.object!r} ===")
        print(f"{'cam':>4} {'mask px':>9} {'conf':>6} {'points':>8} "
              f"{'centroid xyz (m)':>28} {'extent xyz (cm)':>26}")
        print("-" * 92)

        for cam_id in sorted(by_cam):
            pts = by_cam[cam_id]
            c = pts.mean(axis=0)
            e = (pts.max(axis=0) - pts.min(axis=0)) * 100
            print(f"{cam_id:>4} {int(masks[cam_id].sum()):>9} "
                  f"{confidences.get(cam_id) or 0:>6.2f} {len(pts):>8} "
                  f"{np.array2string(c, precision=3, floatmode='fixed'):>28} "
                  f"{np.array2string(e, precision=1, floatmode='fixed'):>26}")

        print("\nA correctly calibrated camera should report an extent close to the")
        print("object's true size. Cameras reporting a much larger extent are")
        print("unprojecting with bad intrinsics, not segmenting badly.")
    finally:
        for cam in cameras.values():
            cam.stop_pipeline()

    return 0


if __name__ == "__main__":
    sys.exit(main())
