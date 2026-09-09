"""
Main entry point for the autonomous chemistry stack.

Wires the Franka arm, the camera cage, the SAM 3 grounding service, the skills
library and the VLM orchestrator together, then runs a task.

This is the module setup.py's `robochem-run` console script points at.

Prerequisites (see scripts/env.sh):
  1. franka-interface / ROS running on the control PC
  2. the grounding service running in perception_env
  3. OPENAI_API_KEY set (for planning and scene description)

Usage:
    source scripts/env.sh
    python scripts/run_experiment.py --task "Pour the water into the beaker"
    python scripts/run_experiment.py --task "..." --dry-run   # plan only, no motion
    python scripts/run_experiment.py --skill pick_up --params '{"object_name":"white paper cup"}'
"""

import argparse
import json
import os
import sys

from frankapy import FrankaArm

from robochem.skills import SkillsExecutor
from robochem.vision import VisionSystem
from robochem.verification.chemistry_verifier import ChemistryVerifier
from robochem.utils.logging_utils import ExperimentLogger
from robochem.orchestrator.vlm_orchestrator import VLMOrchestrator

DEFAULT_CAMERAS = [2, 3, 4, 5]


def build_stack(args):
    """Open the cameras, connect the arm and assemble the subsystems."""
    # Do not hold four RealSense pipelines open. Sequential capture in the
    # localizer starts each camera, takes a frame, and stops it before the
    # next one. Opening all four at once has wedged this machine's USB bus.
    print(f"Cameras {args.cameras} will be captured one at a time")
    cameras = {}

    print(f"Connecting to grounding service at {args.grounding_url}...")
    vision = VisionSystem(
        cameras=cameras,
        grounding_url=args.grounding_url,
        consensus_tolerance=args.consensus_tolerance,
        max_object_extent=args.max_object_extent,
    )
    vision.object_localizer.camera_ids = list(args.cameras)

    robot = None
    if not args.dry_run:
        print("Connecting to FrankaArm...")
        robot = FrankaArm()
        if args.reset:
            print("Resetting to home...")
            robot.reset_joints()
            robot.open_gripper()

    print(f"Workspace Z floor: {args.workspace_min[2]:.3f} m "
          f"(min={args.workspace_min}, max={args.workspace_max})")
    skills = SkillsExecutor(
        robot_interface=robot,
        vision_system=vision,
        config={"workspace_min": args.workspace_min, "workspace_max": args.workspace_max},
    )

    return cameras, vision, skills, robot


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", help="Natural language task description")
    parser.add_argument("--instruction-image", help="Image containing written instructions")
    parser.add_argument("--skill", help="Run a single skill instead of a full task")
    parser.add_argument("--params", default="{}", help="JSON params for --skill")
    parser.add_argument("--cameras", nargs="+", type=int, default=DEFAULT_CAMERAS)
    parser.add_argument(
        "--grounding-url",
        default=os.environ.get("GROUNDING_URL", "http://127.0.0.1:5005"),
    )
    parser.add_argument("--log-dir", default=os.environ.get("ROBOCHEM_LOG_DIR", "experiments"))
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--reset", action="store_true", help="Home the arm before starting")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan and perceive but never command the arm",
    )
    parser.add_argument("--consensus-tolerance", type=float, default=0.05)
    parser.add_argument("--max-object-extent", type=float, default=0.35)
    parser.add_argument("--workspace-min", nargs=3, type=float, default=[0.25, -0.40, 0.015])
    parser.add_argument("--workspace-max", nargs=3, type=float, default=[0.75, 0.40, 0.70])
    args = parser.parse_args()

    if not (args.task or args.instruction_image or args.skill):
        parser.error("one of --task, --instruction-image or --skill is required")

    if not os.environ.get("OPENAI_API_KEY") and not args.skill:
        print("OPENAI_API_KEY is not set; planning will fail. Run 'source scripts/env.sh'.")
        return 1

    cameras, vision, skills, robot = build_stack(args)

    try:
        # Single-skill mode: useful for validating one skill on hardware
        # without involving the planner.
        if args.skill:
            params = json.loads(args.params)
            print(f"\nExecuting skill {args.skill!r} with {params}")
            if args.dry_run:
                print("(dry run: resolving perception only, not moving)")
                located = vision.locate(params.get("object_name", args.skill))
                print(json.dumps(
                    {k: (v.tolist() if hasattr(v, "tolist") else v)
                     for k, v in (located or {}).items() if k != "points"},
                    indent=2, default=str,
                ))
                return 0 if located else 1

            success, result = skills.execute(args.skill, params)
            print(f"\nsuccess={success}")
            print(json.dumps(result, indent=2, default=str))
            return 0 if success else 1

        # Full orchestrated task
        logger = ExperimentLogger(base_dir=args.log_dir)
        orchestrator = VLMOrchestrator(
            skills_executor=skills,
            vision_system=vision,
            verifier=ChemistryVerifier(),
            logger=logger,
        )

        result = orchestrator.run_task(
            task_description=args.task,
            instruction_image=args.instruction_image,
            max_retries=args.max_retries,
        )

        print("\n" + "=" * 60)
        print(json.dumps(result, indent=2, default=str)[:4000])
        return 0 if result.get("success") else 1

    finally:
        for cam in cameras.values():
            try:
                cam.stop_pipeline()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
