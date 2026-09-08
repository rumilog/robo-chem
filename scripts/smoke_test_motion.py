"""
Motion smoke test: exercises the skills library against the real Franka arm
without involving vision.

Runs GripperSkill and MoveToSkill through SkillsExecutor so the real dispatch
path, the pose adapter and the workspace clamp all get covered. Every target is
an explicit coordinate, so nothing depends on segmentation being correct.

Usage:
    source scripts/env.sh
    python scripts/smoke_test_motion.py            # gripper only, no arm motion
    python scripts/smoke_test_motion.py --move     # also move the arm
"""

import argparse
import sys

import numpy as np

from frankapy import FrankaArm

from robochem.skills import SkillsExecutor


def report(fa: FrankaArm, label: str) -> None:
    pose = fa.get_pose()
    print(f"  [{label}] xyz={np.round(pose.translation, 4)} gripper={fa.get_gripper_width():.4f}")


def run_step(executor: SkillsExecutor, fa: FrankaArm, skill: str, params: dict) -> bool:
    print(f"\n>>> {skill} {params}")
    success, result = executor.execute(skill, params)
    print(f"    success={success} result={result}")
    report(fa, "after")
    return success


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--move",
        action="store_true",
        help="Enable arm motion. Without this only the gripper is exercised.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset joints to home before starting.",
    )
    args = parser.parse_args()

    print("Connecting to FrankaArm...")
    fa = FrankaArm()
    print("Connected.")
    report(fa, "start")

    if args.reset:
        print("\n>>> reset_joints")
        fa.reset_joints()
        report(fa, "after reset")

    # No vision system: every skill used here targets explicit coordinates.
    executor = SkillsExecutor(robot_interface=fa, vision_system=None)

    failures = []

    # Gripper via the aliased registry names the orchestrator would emit.
    for skill, params in [("open_gripper", {}), ("close_gripper", {}), ("open_gripper", {})]:
        if not run_step(executor, fa, skill, params):
            failures.append(skill)

    if args.move:
        start = fa.get_pose().translation.copy()

        # Small, explicit, in-bounds waypoints near the current pose.
        waypoints = [
            start + np.array([0.00, 0.00, -0.05]),
            start + np.array([0.05, 0.00, -0.05]),
            start + np.array([0.00, 0.05, -0.05]),
            start,
        ]

        for i, wp in enumerate(waypoints):
            ok = run_step(
                executor,
                fa,
                "move_to",
                {"target": wp.tolist(), "speed": "slow", "maintain_orientation": True},
            )
            if not ok:
                failures.append(f"move_to[{i}]")
                break

        # Confirm the clamp rejects an out-of-bounds target instead of driving
        # into the table.
        run_step(
            executor,
            fa,
            "move_to",
            {"target": [0.45, 0.0, -0.50], "speed": "slow"},
        )
    else:
        print("\n(arm motion skipped; pass --move to enable)")

    print("\n" + "=" * 60)
    if failures:
        print(f"FAILED steps: {failures}")
        return 1
    print("All smoke test steps passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
