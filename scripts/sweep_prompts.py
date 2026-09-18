"""
Find a SAM 3 prompt that actually segments an object, without using the robot.

Naming a new tool is guesswork — SAM 3 is open-vocabulary but not unlimited, and
a phrase it has no concept for returns nothing at all ("rectangular scoop" found
zero masks on all four cameras). Burning a robot run per guess is slow; this
tries a whole list against saved stills in one pass and ranks what works.

    source scripts/env.sh
    python scripts/capture_scene.py --label new_scoop        # once, with the
                                                             # tool on the bench
    python scripts/sweep_prompts.py --images-dir scene_captures/new_scoop_<ts>

By default it sweeps a built-in list of tool words. Add your own with --prompt
(repeatable) or replace the list entirely with --only.

A prompt is only useful if it finds the object on SEVERAL cameras and does not
also match the cups, so the report shows per-camera hits and mask sizes: a
"match" covering a quarter of the frame is the table, not a scoop.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robochem.vision.grounding_client import GroundingClient

#: Ordered roughly from most to least likely for a small printed hand tool.
DEFAULT_PROMPTS = [
    # plain nouns first — SAM 3 is strongest on common single words
    "spoon",
    "scoop",
    "spatula",
    "ladle",
    "shovel",
    "trowel",
    # material / colour qualifiers
    "white spoon",
    "plastic spoon",
    "white plastic spoon",
    "measuring spoon",
    "plastic scoop",
    "white scoop",
    "measuring scoop",
    # shape-led
    "square spoon",
    "rectangular spoon",
    "scoop with a handle",
    "spoon with a bent handle",
    # last resort: generic object words
    "white plastic tool",
    "3d printed part",
    "white object",
]


def load_color_images(images_dir: str):
    images = {}
    for path in sorted(glob.glob(os.path.join(images_dir, "cam*_color.png"))):
        m = re.search(r"cam(\d+)_color", os.path.basename(path))
        if not m:
            continue
        img = cv2.imread(path)
        if img is not None:
            images[int(m.group(1))] = img
    return images


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images-dir", required=True,
                    help="a scene_captures/<run> folder holding cam*_color.png")
    ap.add_argument("--grounding-url",
                    default=os.environ.get("GROUNDING_URL", "http://127.0.0.1:5005"))
    ap.add_argument("--prompt", action="append", default=[],
                    help="extra prompt to try (repeatable)")
    ap.add_argument("--only", action="append", default=[],
                    help="try ONLY these prompts (repeatable)")
    ap.add_argument("--conf", type=float, default=None,
                    help="confidence override passed to the service")
    ap.add_argument("--save-overlays", default=None,
                    help="directory to write mask overlays for prompts that hit")
    args = ap.parse_args()

    images = load_color_images(args.images_dir)
    if not images:
        print(f"No cam*_color.png under {args.images_dir}")
        return 1
    frame_px = next(iter(images.values())).shape[0] * next(iter(images.values())).shape[1]
    print(f"Loaded cameras {sorted(images)} from {args.images_dir}")

    prompts = args.only or (DEFAULT_PROMPTS + args.prompt)
    client = GroundingClient(url=args.grounding_url)

    rows = []
    for prompt in prompts:
        try:
            masks, scores = client.segment(images, prompt, conf=args.conf)
        except Exception as exc:
            print(f"  {prompt!r}: ERROR {exc}")
            continue

        if not masks:
            print(f"  {prompt:28} -> no masks")
            rows.append((prompt, 0, 0.0, 0.0))
            continue

        # Mask area as a fraction of frame: a huge "match" is the table.
        fracs = [float(m.sum()) / frame_px for m in masks.values()]
        mean_score = float(np.mean([s for s in scores.values() if s is not None])
                           or 0.0) if scores else 0.0
        print(f"  {prompt:28} -> {len(masks)}/{len(images)} cams, "
              f"score {mean_score:.2f}, mask {min(fracs) * 100:.1f}-"
              f"{max(fracs) * 100:.1f}% of frame")
        rows.append((prompt, len(masks), mean_score, float(np.mean(fracs))))

        if args.save_overlays:
            os.makedirs(args.save_overlays, exist_ok=True)
            safe = re.sub(r"[^a-z0-9]+", "_", prompt.lower()).strip("_")
            for cam, mask in masks.items():
                vis = images[cam].copy()
                vis[mask.astype(bool)] = (
                    0.5 * vis[mask.astype(bool)] + 0.5 * np.array([0, 0, 255])
                ).astype(np.uint8)
                cv2.imwrite(os.path.join(args.save_overlays,
                                         f"{safe}_cam{cam}.png"), vis)

    print("\n" + "=" * 68)
    print("Ranked (cameras found, then score). A good tool prompt hits most")
    print("cameras with a SMALL mask — a large mask is the table or a cup.")
    print("=" * 68)
    usable = [r for r in rows if r[1] > 0]
    for prompt, cams, score, frac in sorted(usable, key=lambda r: (-r[1], -r[2])):
        flag = "  <-- suspiciously large" if frac > 0.05 else ""
        print(f"  {cams}/{len(images)} cams  score {score:.2f}  "
              f"mask {frac * 100:5.2f}%  {prompt}{flag}")
    if not usable:
        print("  nothing matched. Try --prompt with your own wording, or check")
        print("  the tool is actually visible and unoccluded in these frames.")
        return 1

    if args.save_overlays:
        print(f"\nOverlays written to {args.save_overlays}/ — LOOK AT THEM before")
        print("trusting any prompt. A confident mask on the wrong object still")
        print("reports as a clean hit here.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
