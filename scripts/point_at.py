"""
Point the gripper at where perception thinks each object is, so a person can
see how far off that is.

For every name given, the cell locates it exactly as the skills do (a
container's rim centre from BaseSkill.locate_container, a tool's centroid),
then parks the fingertips straight down over that spot, a little above the
object, and waits. Look down past the fingers and read how far the true centre
of the object is from the fingertips: forward/back (toward or away from the
arm) and left/right (the arm's left and right). Press Enter for the next one.

Several objects on different parts of the bench tell whether the error is one
shift everywhere (calibration: one correction fixes every skill) or differs
per object (how each centre is estimated).

Moves are position-controlled and go up to a safe height before crossing the
bench. Hold the e-stop.

Usage:
    source scripts/env.sh
    python scripts/point_at.py "citric acid cup" "clear cup a" "plastic beaker" "larger spoon"
    python scripts/point_at.py --above 0.05 "baking soda cup"
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from robochem.vision import VisionSystem
from robochem.vision.bench_manifest import BENCHES
from robochem.skills.base_skill import to_rigid_transform
from robochem.skills.dump import DumpSkill

DOWN = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
SAFE_Z = 0.25


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="+", help="Bench objects to point at")
    parser.add_argument("--above", type=float, default=0.05,
                        help="Fingertips this far above the object's top, metres")
    parser.add_argument("--bench", choices=sorted(BENCHES), default="magic_beaker")
    parser.add_argument("--cameras", nargs="+", type=int, default=[2, 3, 4, 5])
    parser.add_argument("--grounding-url",
                        default=os.environ.get("GROUNDING_URL", "http://127.0.0.1:5005"))
    args = parser.parse_args()

    from frankapy import FrankaArm

    bench = BENCHES[args.bench]
    vision = VisionSystem(cameras={}, grounding_url=args.grounding_url, bench=bench)
    vision.object_localizer.camera_ids = list(args.cameras)
    fa = FrankaArm()
    # Any skill gives locate_container: the rim centre and top every skill aims at.
    probe = DumpSkill(fa, vision, {"workspace_min": [0.25, -0.40, -0.13], "table_z": 0.0})

    def go(xyz, seconds=5.0):
        pose = fa.get_pose()
        pose.translation = np.asarray(xyz, dtype=float)
        pose.rotation = DOWN
        fa.goto_pose(to_rigid_transform(pose), duration=seconds, use_impedance=False)
        got = np.asarray(fa.get_pose().translation, dtype=float)
        off = float(np.linalg.norm(got - np.asarray(xyz, dtype=float)))
        if off > 0.005:
            print(f"   (arm stopped {off * 1000:.1f} mm from the command)")

    rows = []
    fa.reset_joints()
    fa.close_gripper()
    for name in args.names:
        obj = bench.find(name)
        label = obj.name if obj is not None else name
        print(f"\n===== {label} =====")
        vision.clear_cache()
        located = probe.locate_container(label, force_refresh=True)
        if located is None:
            print("   not found; skipped")
            continue
        centre = np.asarray(located["rim_center"], dtype=float)
        top = float(located["top_z"])
        # The rim reads low on the cage (16 mm for the 29 mm dish): use the
        # bench's known height when there is one, so the fingers stay clear.
        known = None
        if obj is not None:
            known = obj.height or dict(obj.scoop).get("container_height")
        if known:
            top = max(top, float(known))
        hover = np.array([centre[0], centre[1], top + float(args.above)])
        print(f"   perceived centre ({centre[0]:.4f}, {centre[1]:+.4f}), top z {top:.4f} "
              f"[{located.get('rim_method', '')}; cams {located.get('cameras')}]")
        now = np.asarray(fa.get_pose().translation, dtype=float)
        go([now[0], now[1], SAFE_Z], 4.0)
        go([hover[0], hover[1], SAFE_Z], 5.0)
        go(hover, 4.0)
        input("   The fingertips are over the perceived centre. Read the offset to the real "
              "centre (forward/back, left/right), then press Enter... ")
        rows.append((label, centre))
        go([hover[0], hover[1], SAFE_Z], 4.0)

    fa.reset_joints()
    print("\nPerceived centres (base frame, metres):")
    for label, c in rows:
        print(f"   {label:24s} ({c[0]:.4f}, {c[1]:+.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
