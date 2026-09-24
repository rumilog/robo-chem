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

With --sim the same skills run against the MuJoCo cell instead of the arm and
the cage, in a window you can watch. That route has no frankapy and no ROS, so
it runs out of perception_env rather than the robot venv:

    perception_env/bin/python scripts/run_experiment.py --sim \
        --skill pick_up --params '{"object_name":"plastic beaker","z_offset":0.02}'

--task runs the LLM agent pipeline (robochem.agents): it looks at the bench,
plans sub-tasks, picks a skill and its parameters for each one, executes them,
and replans when one fails. --dry-run does all of that except the executing, so
you can see what the model would do for the price of the tokens:

    perception_env/bin/python scripts/run_experiment.py --sim --dry-run \
        --task "Scoop citric acid into the white paper cup"
"""

import argparse
import ast
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robochem.skills import SkillsExecutor
from robochem.vision import VisionSystem
from robochem.verification.chemistry_verifier import ChemistryVerifier
from robochem.utils.logging_utils import ExperimentLogger
from robochem.agents import config as agent_config
from robochem.orchestrator.agent_orchestrator import AgentOrchestrator
from robochem.orchestrator.vlm_orchestrator import VLMOrchestrator

DEFAULT_CAMERAS = [2, 3, 4, 5]


def parse_params(raw: str) -> dict:
    """
    Turn ``--params`` into a dict, tolerating the shell it had to be typed in.

    Windows PowerShell 5.1 does not pass a JSON argument through: it eats the
    inner double quotes, and the CRT then splits what is left at the space in
    "plastic beaker", so the documented bash form arrives as two broken
    arguments and cannot be escaped around. Single quotes survive it intact, so
    a Python dict literal is accepted as well, and ``@file.json`` for anything
    long enough that quoting is not worth the argument.
    """
    raw = raw.strip()
    if raw.startswith("@"):
        raw = Path(raw[1:]).expanduser().read_text(encoding="utf-8").strip()

    try:
        params = json.loads(raw)
    except json.JSONDecodeError:
        try:
            params = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            raise SystemExit(
                f"--params is neither JSON nor a Python dict literal: {raw!r}\n"
                "In PowerShell the double quotes do not survive, so use single "
                "ones:\n"
                "  --params \"{'object_name':'plastic beaker','z_offset':0.02}\"\n"
                "or read them from a file: --params @params.json"
            )

    if not isinstance(params, dict):
        raise SystemExit(f"--params must be a mapping, got {type(params).__name__}")
    return params


def build_sim_stack(args):
    """
    Assemble the MuJoCo cell instead of the hardware one.

    Returns the same 4-tuple as :func:`build_stack` so main() does not care
    which cell it got. ``cameras`` is empty because the simulated cage is
    rendered on demand rather than held open.
    """
    from robochem.sim import build_cell

    print(f"Building the simulated cell (speed x{args.sim_speed}, "
          f"viewer={'on' if not args.no_viewer else 'off'})")
    cell = build_cell(
        viewer=not args.no_viewer,
        realtime=not args.sim_fast,
        speed=args.sim_speed,
        granules=args.sim_granules,
        tool_length=args.sim_tool_length,
        grasp_mode=args.sim_grasp_mode,
        calib_dir=args.sim_calib_dir,
        workspace_min=args.workspace_min,
        workspace_max=args.workspace_max,
    )
    if args.reset:
        cell.reset()
    return {}, cell.vision, cell.skills, cell.arm


def build_stack(args):
    """Open the cameras, connect the arm and assemble the subsystems."""
    from frankapy import FrankaArm

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
    parser.add_argument("--skill", action="append", metavar="NAME",
                        help="Run a skill instead of a full task. Repeat it to "
                             "run a sequence in ONE session, which is the only "
                             "way skills that hand off a held tool (pick_up "
                             "then stir) can work: a second invocation builds a "
                             "fresh cell with an empty gripper.")
    parser.add_argument("--params", action="append", metavar="JSON",
                        help="Params for --skill: JSON, a Python dict literal "
                             "(which is what survives PowerShell), or @file.json")
    parser.add_argument("--cameras", nargs="+", type=int, default=DEFAULT_CAMERAS)
    parser.add_argument(
        "--grounding-url",
        default=os.environ.get("GROUNDING_URL", "http://127.0.0.1:5005"),
    )
    parser.add_argument("--log-dir", default=os.environ.get("ROBOCHEM_LOG_DIR", "experiments"))
    parser.add_argument("--max-retries", type=int, default=3,
                        help="Corrective replanning attempts for --task")
    parser.add_argument("--legacy-planner", action="store_true",
                        help="Run --task through the older single-prompt "
                             "VLMOrchestrator instead of the agent pipeline")
    parser.add_argument("--verify", dest="verify", action="store_true", default=None,
                        help="Run the chemistry check after a --task run. On by "
                             "default on hardware, off in simulation, which has "
                             "no chemistry to check")
    parser.add_argument("--no-verify", dest="verify", action="store_false",
                        help="Skip the chemistry check after a --task run")
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

    sim = parser.add_argument_group("simulation")
    sim.add_argument("--sim", action="store_true",
                     help="Run against the MuJoCo cell instead of the robot")
    sim.add_argument("--sim-speed", type=float, default=1.0,
                     help="Playback multiplier; 2.0 halves every commanded duration")
    sim.add_argument("--sim-fast", action="store_true",
                     help="Step as fast as the machine allows instead of real time")
    sim.add_argument("--no-viewer", action="store_true",
                     help="Headless simulation (for tests and remote shells)")
    sim.add_argument("--sim-granules", action="store_true",
                     help="Put loose particles in the reagent cups so pours and "
                          "scoops move material")
    sim.add_argument("--sim-tool-length", type=float, default=0.0,
                     help="Length (m) of a fixed tool bolted to the hand")
    sim.add_argument("--sim-grasp-mode", choices=["magnet", "physics"], default="magnet",
                     help="Kinematic attach on close (default) or friction contacts")
    sim.add_argument("--sim-calib-dir", default="calibration_out",
                     help="Directory of realsense_cameraNw.npy to place the "
                          "simulated cage at. Point this at robomail's calib/ to "
                          "reproduce whatever extrinsics the robot is running")
    sim.add_argument("--sim-hold", type=float, default=None,
                     help="Seconds to keep the viewer open after the run "
                          "(default: until you close the window)")
    args = parser.parse_args()

    if not (args.task or args.instruction_image or args.skill):
        parser.error("one of --task, --instruction-image or --skill is required")

    # Validate the skill/params pairing here, not once the cell is up: on the
    # simulated path parser.error() past that point still runs the viewer
    # teardown, so the usage message scrolls away behind it.
    if args.skill and args.params and len(args.params) > len(args.skill):
        parser.error(f"got {len(args.params)} --params for {len(args.skill)} "
                     f"--skill; they are matched in order")

    # Reads the repo .env as well as the environment, so a simulated run out of
    # perception_env -- which never sources scripts/env.sh, and must not, since
    # that activates the robot venv -- still finds the key.
    if not args.skill and not agent_config.api_key():
        print("No API key. Set OPENAI_API_KEY, or put it in the repo's .env "
              "(openai_api_key=... is accepted). Planning cannot run without one.")
        return 1

    cameras, vision, skills, robot = (
        build_sim_stack(args) if args.sim else build_stack(args)
    )

    try:
        # Skill mode: useful for validating skills without involving the
        # planner. Several --skill flags run as a sequence against the one cell,
        # so a tool picked up by the first is still held by the next.
        if args.skill:
            raw = args.params or []
            # Skills with no --params of their own run on their defaults.
            plans = [(name, parse_params(raw[i] if i < len(raw) else "{}"))
                     for i, name in enumerate(args.skill)]

            if args.dry_run:
                print("(dry run: resolving perception only, not moving)")
                name, params = plans[0]
                located = vision.locate(params.get("object_name", name))
                print(json.dumps(
                    {k: (v.tolist() if hasattr(v, "tolist") else v)
                     for k, v in (located or {}).items() if k != "points"},
                    indent=2, default=str,
                ))
                return 0 if located else 1

            last = {}
            for step, (name, params) in enumerate(plans, 1):
                if len(plans) > 1:
                    print(f"\n--- step {step}/{len(plans)}: {name} ---")
                print(f"Executing skill {name!r} with {params}")
                success, last = skills.execute(name, params)
                print(f"success={success}")
                if not success:
                    print(json.dumps(last, indent=2, default=str))
                    print(f"\nStopped at step {step}/{len(plans)} ({name})")
                    return 1
            print()
            print(json.dumps(last, indent=2, default=str))
            return 0

        # Full orchestrated task
        if args.legacy_planner:
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

        # MuJoCo models geometry and contact, not chemistry -- its granules are
        # bouncy spheres -- so a colour-change check there fails for reasons that
        # have nothing to do with the plan, and each failure costs a replan.
        verify = args.verify if args.verify is not None else not args.sim
        orchestrator = AgentOrchestrator(
            skills_executor=skills,
            vision_system=vision,
            robot=robot,
            verifier=ChemistryVerifier() if verify else None,
            max_replans=args.max_retries,
            verify=verify,
            dry_run=args.dry_run,
            log_dir=args.log_dir,
        )

        outcome = orchestrator.run(
            task=args.task,
            instruction_image=args.instruction_image,
        )

        print("\n" + "=" * 60)
        print(json.dumps(outcome.as_dict(), indent=2, default=str)[:6000])
        return 0 if outcome.success else 1

    finally:
        if args.sim and not args.no_viewer:
            print("\nRun finished. Close the viewer window to exit"
                  + (f" (or wait {args.sim_hold:.0f}s)" if args.sim_hold else "") + ".")
            robot.hold(args.sim_hold if args.sim_hold is not None else 1e9)
        if args.sim:
            robot.close()
        for cam in cameras.values():
            try:
                cam.stop_pipeline()
            except Exception:
                pass


if __name__ == "__main__":
    sys.exit(main())
