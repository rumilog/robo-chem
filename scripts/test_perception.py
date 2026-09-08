"""
End-to-end perception test: VLM detection -> SAM segmentation -> 3D pointcloud
-> grasp pose. Read-only, the robot is never commanded.

Usage:
    source scripts/env.sh
    python scripts/test_perception.py --objects "white paper cup" "clear plastic beaker"
    python scripts/test_perception.py --describe          # also dump a scene description
"""

import argparse
import os
import sys

import numpy as np
import cv2

import robomail.vision as vis

from robochem.vision import VisionSystem

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "diag_out")
CAM_IDS = [2, 3, 4, 5]


def save_mask_overlay(image, mask, path):
    overlay = image.copy()
    overlay[mask] = (0.5 * overlay[mask] + 0.5 * np.array([0, 0, 255])).astype(np.uint8)
    ys, xs = np.where(mask)
    if len(xs):
        cv2.rectangle(overlay, (xs.min(), ys.min()), (xs.max(), ys.max()), (0, 255, 0), 2)
    cv2.imwrite(path, overlay)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects", nargs="+", default=["white paper cup"])
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--cameras", nargs="+", type=int, default=CAM_IDS)
    args = parser.parse_args()

    grounding_url = os.environ.get("GROUNDING_URL")
    if not grounding_url:
        print("GROUNDING_URL is not set. Run 'source scripts/env.sh' first.")
        return 1

    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Opening cameras {args.cameras}...")
    cameras = {c: vis.CameraClass(cam_number=c) for c in args.cameras}

    print(f"Connecting to grounding service at {grounding_url}...")
    vision = VisionSystem(cameras=cameras, grounding_url=grounding_url)
    vision.object_localizer.camera_ids = list(args.cameras)

    failures = []
    try:
        if args.describe:
            print("\n=== scene description ===")
            images = vision.capture_scene()
            desc = vision.scene_analyzer.get_scene_description(images)
            import json
            print(json.dumps(desc, indent=2)[:3000])

        for obj in args.objects:
            print(f"\n{'=' * 60}\n=== locating: {obj!r}\n{'=' * 60}")
            located = vision.locate(obj, force_refresh=True)

            if located is None:
                print("  NOT FOUND")
                failures.append(obj)
                continue

            centroid = located["centroid"]
            dims = located["dimensions"]
            print(f"  cameras used: {located['cameras']}")
            if located.get("rejected_cameras"):
                print(f"  rejected    : {located['rejected_cameras']}")
            print(f"  confidences : {located['confidences']}")
            print(f"  points      : {len(located['points'])}")
            print(f"  centroid    : {np.round(centroid, 4)} (robot base frame, m)")
            print(f"  dimensions  : {np.round(dims, 4)} m")

            grasp = vision.compute_grasp_pose(located["points"], obj, grasp_type="auto")
            if grasp is None:
                print("  grasp pose  : FAILED")
                failures.append(f"{obj}:grasp")
            else:
                print(f"  grasp xyz   : {np.round(grasp[:3, 3], 4)}")
                print(f"  grasp R     :\n{np.round(grasp[:3, :3], 3)}")

            # Save mask overlays so the detection can be eyeballed.
            data = vision.object_localizer.capture_pointclouds()
            masks, _ = vision.scene_analyzer.segment_object(
                [data["images"][c] for c in args.cameras if c in data["images"]],
                obj,
                cameras=[c for c in args.cameras if c in data["images"]],
            )
            slug = obj.replace(" ", "_")
            for cam_id, mask in masks.items():
                save_mask_overlay(
                    data["images"][cam_id], mask,
                    os.path.join(OUT_DIR, f"mask_{slug}_cam{cam_id}.png"),
                )
    finally:
        for cam in cameras.values():
            cam.stop_pipeline()

    print("\n" + "=" * 60)
    if failures:
        print(f"FAILURES: {failures}")
        return 1
    print(f"Perception pipeline OK. Overlays in {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
