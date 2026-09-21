"""
Capture colour + depth stills from the *simulated* cage cameras (2-5).

The sim counterpart of capture_scene.py, writing the same filenames into the
same place, so a simulated scene can be looked at -- or diffed against a real
capture -- with whatever already reads scene_captures/.

The cameras are the calibrated ones: each is placed at the measured
camera-to-world transform from calibration_out/ and rendered at the cage's
848x480 with the D435's 42.5 degree vertical field, so a simulated frame has
the same geometry as a real one.

Usage:
    sim_env/bin/python scripts/capture_sim_scene.py
    sim_env/bin/python scripts/capture_sim_scene.py --label spoon_swap
    sim_env/bin/python scripts/capture_sim_scene.py --cameras 2 4 --no-grid
    sim_env/bin/python scripts/capture_sim_scene.py --out diag_out/sim_cams
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robochem.sim import build_cell

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "scene_captures"
CAM_IDS = [2, 3, 4, 5]


def grid(frames, cols=2, pad=8):
    """Tile the per-camera frames into one contact sheet."""
    h, w = frames[0].shape[:2]
    rows = (len(frames) + cols - 1) // cols
    sheet = np.full((rows * h + (rows + 1) * pad,
                     cols * w + (cols + 1) * pad, 3), 32, np.uint8)
    for i, frame in enumerate(frames):
        r, c = divmod(i, cols)
        y, x = pad + r * (h + pad), pad + c * (w + pad)
        sheet[y:y + h, x:x + w] = frame
    return sheet


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--label", default="sim", help="Folder name suffix")
    parser.add_argument("--cameras", nargs="+", type=int, default=CAM_IDS)
    parser.add_argument("--out", default=None,
                        help="Write here instead of a timestamped "
                             "scene_captures/ folder")
    parser.add_argument("--calib-dir", default="calibration_out",
                        help="Where to read the camera extrinsics from")
    parser.add_argument("--granules", action="store_true",
                        help="Fill the reagent cups before capturing")
    parser.add_argument("--settle", type=float, default=0.5,
                        help="Seconds of physics to run before capturing, so "
                             "free props are resting rather than mid-drop")
    parser.add_argument("--no-grid", action="store_true")
    args = parser.parse_args()

    if args.out:
        out = Path(args.out)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = OUT_ROOT / f"{args.label}_{stamp}"
    out.mkdir(parents=True, exist_ok=True)

    cell = build_cell(viewer=False, realtime=False, granules=args.granules,
                      calib_dir=args.calib_dir, verbose=False)
    try:
        if args.settle > 0:
            cell.arm.settle(args.settle)

        frames = []
        for cam_id in args.cameras:
            if cam_id not in cell.vision.camera_ids:
                print(f"  cam{cam_id}: no extrinsics, skipping")
                continue

            rgb = cell.vision.render_rgb(cam_id)
            depth_m, _ = cell.vision._render(cam_id)
            color = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)       # match the cage
            depth = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)

            cv2.imwrite(str(out / f"cam{cam_id}_color.png"), color)
            cv2.imwrite(str(out / f"cam{cam_id}_depth.png"), depth)
            viz = cv2.applyColorMap(
                cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype("uint8"),
                cv2.COLORMAP_TURBO)
            cv2.imwrite(str(out / f"cam{cam_id}_depth_viz.png"), viz)
            frames.append(color)
            print(f"  wrote cam{cam_id}_color.png {color.shape}")

        if frames and not args.no_grid:
            cv2.imwrite(str(out / "grid.png"), grid(frames))
            print("  wrote grid.png")

        (out / "README.txt").write_text(
            "Simulated cage stills, rendered by robochem.sim.\n"
            f"Captured: {datetime.now().isoformat(timespec='seconds')}\n"
            f"Label: {args.label}\n"
            f"Extrinsics: {args.calib_dir}\n"
            "Cameras: MuJoCo cameras at the cage's measured extrinsics, "
            "848x480, fovy 42.5 (fx = fy = 617, cx = 424, cy = 240).\n"
            "Files:\n"
            "  cam{N}_color.png     - BGR colour frame\n"
            "  cam{N}_depth.png     - 16-bit depth (mm), no sensor dropout\n"
            "  cam{N}_depth_viz.png - colorized depth for quick viewing\n"
            "  grid.png             - all cameras on one sheet\n",
            encoding="utf-8")
    finally:
        cell.close()

    print(f"DONE {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
