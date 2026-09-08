"""
Teach / replay a pour *wrist orientation* from the live Franka.

This saves the end-effector rotation that pours correctly for the grasp you
are using (spout-aligned beaker, tip toward base, etc.). It is NOT a fixed
spot in the cage — pour will apply this orientation at whatever XYZ the arm
is at (or above the target cup).

1. Grasp the beaker the usual way, then tip it to the pour posture you want.
2. python scripts/teach_pour_pose.py --save
3. Pour skill loads that orientation on later runs.

Usage:
    source scripts/env.sh
    python scripts/teach_pour_pose.py --save
    python scripts/teach_pour_pose.py --print
"""

import argparse
import json
import os
import sys

import numpy as np
from frankapy import FrankaArm

DEFAULT_PATH = os.path.join(
    os.path.dirname(__file__), "..", "calibration_out", "taught_pour_pose.json"
)


def snapshot(fa: FrankaArm) -> dict:
    pose = fa.get_pose()
    joints = np.asarray(fa.get_joints(), dtype=float)
    R = np.asarray(pose.rotation, dtype=float)
    t = np.asarray(pose.translation, dtype=float)
    tip = R[:3, 2]
    tip_deg = float(np.degrees(np.arccos(np.clip(-tip[2], -1.0, 1.0))))
    return {
        "joints": joints.tolist(),
        "translation": t.tolist(),
        "rotation": R.tolist(),
        "gripper_width": float(fa.get_gripper_width()),
        "tip_from_vertical_deg": tip_deg,
        "tip_xy": tip[:2].tolist(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", default=DEFAULT_PATH)
    parser.add_argument("--save", action="store_true", help="Snapshot current pose")
    parser.add_argument("--goto", action="store_true", help="Replay saved joints")
    parser.add_argument("--print", action="store_true", dest="print_only",
                        help="Print current pose without saving")
    parser.add_argument("--duration", type=float, default=6.0)
    args = parser.parse_args()

    print("Connecting to FrankaArm...")
    fa = FrankaArm()

    if args.print_only or (not args.save and not args.goto):
        data = snapshot(fa)
        print(json.dumps(data, indent=2))
        if not args.save and not args.goto:
            print("\n(no --save/--goto; showing live pose only)")
            return 0

    if args.save:
        data = snapshot(fa)
        os.makedirs(os.path.dirname(os.path.abspath(args.path)) or ".", exist_ok=True)
        with open(args.path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Saved taught pour pose → {os.path.abspath(args.path)}")
        print(f"  joints: {np.round(data['joints'], 3)}")
        print(f"  xyz:    {np.round(data['translation'], 4)}")
        print(f"  tip:    {data['tip_from_vertical_deg']:.1f}° from vertical, "
              f"tip_xy={np.round(data['tip_xy'], 3)}")
        return 0

    if args.goto:
        with open(args.path) as f:
            data = json.load(f)
        joints = np.asarray(data["joints"], dtype=float)
        print(f"Going to taught joints {np.round(joints, 3)} "
              f"(duration={args.duration}s, cartesian impedance off)...")
        fa.goto_joints(joints, duration=float(args.duration), use_impedance=False)
        after = snapshot(fa)
        print(f"Arrived tip={after['tip_from_vertical_deg']:.1f}° "
              f"xyz={np.round(after['translation'], 4)}")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
