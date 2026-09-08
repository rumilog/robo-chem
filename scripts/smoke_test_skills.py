"""
Smoke-test a short skill chain with pauses between steps.

Default: locate plastic beaker → spout-aligned pick → (optional) pour into
white paper cup. Press Enter between steps so you can watch / abort.

Usage:
    source scripts/env.sh
    python scripts/smoke_test_skills.py
    python scripts/smoke_test_skills.py --pour
    python scripts/smoke_test_skills.py --source "plastic beaker" --target "white paper cup"
    python scripts/smoke_test_skills.py --no-pause   # run straight through
"""

import argparse
import sys

import numpy as np

from frankapy import FrankaArm

from robochem.skills import SkillsExecutor
from robochem.vision import VisionSystem


DEFAULT_CAMERAS = [2, 3, 4, 5]


def pause(enabled: bool, msg: str) -> None:
    if not enabled:
        print(f"\n--- {msg} ---")
        return
    input(f"\n[pause] {msg}  (Enter to continue, Ctrl-C to abort) ")


def run(executor: SkillsExecutor, skill: str, params: dict) -> bool:
    print(f"\n>>> {skill} {params}")
    ok, result = executor.execute(skill, params)
    print(f"    success={ok}")
    if isinstance(result, dict):
        for k in ("error", "spout_xy", "grasp_pose", "centroid", "gripper_width",
                  "poured_into", "tip_sign"):
            if k in result and result[k] is not None:
                v = result[k]
                if k == "grasp_pose" and hasattr(v, "__len__"):
                    print(f"    grasp xyz={np.round(np.asarray(v)[:3, 3], 4)}")
                else:
                    print(f"    {k}={v}")
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="plastic beaker",
                        help="Container to pick (spout-aligned if beaker)")
    parser.add_argument("--target", default="white paper cup",
                        help="Pour destination (only with --pour)")
    parser.add_argument("--pour", action="store_true",
                        help="After pick, pour into --target")
    parser.add_argument("--no-pause", action="store_true",
                        help="Do not wait for Enter between steps")
    parser.add_argument("--cameras", nargs="+", type=int, default=DEFAULT_CAMERAS)
    parser.add_argument("--grounding-url", default="http://127.0.0.1:5005")
    parser.add_argument("--z-offset", type=float, default=-0.01)
    parser.add_argument("--squeeze", type=float, default=0.008)
    parser.add_argument("--pitch-deg", type=float, default=25.0)
    parser.add_argument("--pour-angle", type=float, default=45.0)
    parser.add_argument(
        "--max-object-width",
        type=float,
        default=None,
        help="Optional cap on measured grasp width (m). Default: no cap.",
    )
    args = parser.parse_args()
    do_pause = not args.no_pause

    print(f"Cameras {args.cameras} (sequential capture)")
    vision = VisionSystem(
        cameras={},
        grounding_url=args.grounding_url,
    )
    vision.object_localizer.camera_ids = list(args.cameras)

    print("Connecting to FrankaArm...")
    fa = FrankaArm()
    executor = SkillsExecutor(robot_interface=fa, vision_system=vision)

    pause(do_pause, "open gripper + retract, then locate source")
    if not run(executor, "open_gripper", {}):
        return 1

    # Locate only (via pick's retract path is better; still show spout first).
    pause(do_pause, f"pick_up '{args.source}' with upright top grasp")
    pick_params = {
        "object_name": args.source,
        "grasp_type": "top",
        "z_offset": args.z_offset,
        "squeeze": args.squeeze,
    }
    if args.max_object_width is not None:
        pick_params["max_object_width"] = args.max_object_width
    print(f"\n>>> pick_up {pick_params}")
    ok, pick_result = executor.execute("pick_up", pick_params)
    print(f"    success={ok}")
    if not ok:
        print(f"    result={pick_result}")
        print("Pick failed — aborting smoke test")
        return 1
    if isinstance(pick_result, dict):
        for k in ("gripper_width", "centroid"):
            if pick_result.get(k) is not None:
                print(f"    {k}={pick_result[k]}")

    if not args.pour:
        pause(do_pause, "pick succeeded; done (pass --pour to continue into target)")
        return 0

    pause(do_pause, f"pour into '{args.target}' (taught orientation)")
    pour_params = {
        "target_container": args.target,
        "hold_duration": 1.0,
    }
    if not run(executor, "pour", pour_params):
        print("Pour failed")
        return 1

    print("\nSmoke test finished OK")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nAborted by user")
        sys.exit(130)
