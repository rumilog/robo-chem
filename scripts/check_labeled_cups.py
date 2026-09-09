"""
Offline check: multi-cup SAM + VLM label matching on saved stills.

Does not move the robot. Needs the grounding service (with return_all support)
and OPENAI_API_KEY.

Usage:
    source scripts/env.sh
    python scripts/check_labeled_cups.py \
      --images-dir scene_captures/spoon\\ graspped_20260909_155259 \
      --label "citric acid"
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import cv2

from robochem.vision.grounding_client import GroundingClient
from robochem.vision.label_resolver import LabelResolver, labels_match, normalize_label


def load_color_cams(images_dir: str):
    paths = sorted(glob.glob(os.path.join(images_dir, "cam*_color.png")))
    images = {}
    for path in paths:
        m = re.search(r"cam(\d+)_color", os.path.basename(path))
        if not m:
            continue
        cam_id = int(m.group(1))
        img = cv2.imread(path)
        if img is None:
            print(f"  skip unreadable {path}")
            continue
        images[cam_id] = img
        print(f"  loaded cam {cam_id}: {path} {img.shape}")
    return images


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", required=True)
    parser.add_argument("--label", required=True, help="Reagent label to match")
    parser.add_argument("--category", default="white paper cup")
    parser.add_argument(
        "--grounding-url",
        default=os.environ.get("GROUNDING_URL", "http://127.0.0.1:5005"),
    )
    parser.add_argument("--out-dir", default=None,
                        help="Optional dir for annotated overlays")
    args = parser.parse_args()

    images = load_color_cams(args.images_dir)
    if not images:
        print("No cam*_color.png found")
        return 1

    client = GroundingClient(args.grounding_url)
    health = client.health()
    print(f"Grounding backend: {health.get('backend')}")

    instances = client.segment_instances(images, args.category)
    if not instances:
        print(f"No instances of {args.category!r}")
        return 1

    resolver = LabelResolver()
    masks, confidences, observed = resolver.resolve(images, instances, args.label)

    print(f"\nRequested: {args.label!r} (normalized {normalize_label(args.label)!r})")
    if not masks:
        print("FAIL: no camera matched that label")
        return 1

    out_dir = args.out_dir
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    for cam_id, mask in masks.items():
        lab = observed.get(cam_id)
        print(f"  cam {cam_id}: matched {lab!r} "
              f"score={confidences.get(cam_id)} "
              f"area={int(mask.sum())} "
              f"ok={labels_match(args.label, lab or '')}")
        if out_dir:
            vis = images[cam_id].copy()
            overlay = vis.copy()
            overlay[mask] = (0, 255, 0)
            vis = cv2.addWeighted(overlay, 0.45, vis, 0.55, 0)
            path = os.path.join(out_dir, f"match_cam{cam_id}.png")
            cv2.imwrite(path, vis)
            print(f"    wrote {path}")

    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
