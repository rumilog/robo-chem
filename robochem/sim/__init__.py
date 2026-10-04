"""
MuJoCo preview of the Franka Panda cell.

Lets every skill run -- and be watched -- without the arm or the camera cage.
The arm is mujoco_menagerie's vendor Panda model with the Franka Hand; the four
cage viewpoints are placed at the cell's measured extrinsics, so simulated point
clouds land where real ones do.

    from robochem.sim import build_cell

    cell = build_cell(viewer=True)
    cell.skills.execute("pick_up", {"object_name": "plastic beaker",
                                    "z_offset": 0.02, "squeeze": 0.013})
    cell.arm.hold()          # keep the window open

``cell.arm`` implements the same surface as ``frankapy.FrankaArm`` and
``cell.vision`` the same surface as ``VisionSystem``, so ``cell.skills`` is an
ordinary :class:`~robochem.skills.SkillsExecutor` with nothing simulated about
it -- the skills cannot tell the difference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .bench import Bench, Prop, default_bench, layout_violations, randomize_layout
from .lab import Lab
from .rigid_transform import RigidTransform, install as install_rigid_transform
from .scene import HOME_JOINTS, SimScene, build_scene, reset_scene
from .sim_arm import SimFrankaArm
from .sim_vision import SimVision

__all__ = [
    "Bench", "Prop", "default_bench", "layout_violations",
    "SimScene", "SimFrankaArm", "SimVision", "SimCell", "Lab",
    "build_scene", "build_cell", "reset_scene",
    "HOME_JOINTS", "RigidTransform", "install_rigid_transform",
]

# The README's validated commands run with a lower floor for the spoon; keep
# the same defaults here so a simulated run needs the same arguments.
DEFAULT_WORKSPACE_MIN = [0.25, -0.40, -0.13]
DEFAULT_WORKSPACE_MAX = [0.75, 0.40, 0.70]


@dataclass
class SimCell:
    """An assembled simulated cell: scene, arm, vision and the skills over them."""

    scene: SimScene
    arm: SimFrankaArm
    vision: SimVision
    skills: object
    lab: Optional[Lab] = None

    def reset(self):
        """Arm home, bench back on its marks, perception cache dropped."""
        self.arm.reset_scene()
        self.vision.clear_cache()
        if self.lab is not None:
            self.lab.reset()

    def close(self):
        """Shut the viewer window and the offscreen render context down."""
        self.vision.close()
        self.arm.close()


def build_cell(
    bench: Optional[Bench] = None,
    viewer: bool = True,
    realtime: bool = True,
    speed: float = 1.0,
    granules: bool = False,
    tool_length: float = 0.0,
    grasp_mode: str = "magnet",
    calib_dir: str = "calibration_out",
    workspace_min=None,
    workspace_max=None,
    noise_m: float = 0.002,
    verbose: bool = True,
    layout_seed: Optional[int] = None,
    lab: bool = True,
) -> SimCell:
    """
    Build a simulated cell with the skills wired over it.

    Args:
        bench: Props on the table (default: the README's validated layout)
        viewer: Open the interactive MuJoCo window
        realtime: Play motions at wall-clock speed
        speed: Multiplier on commanded durations
        granules: Loose particles in the reagent cups, so pours and scoops
            actually move material
        tool_length: Length of a fixed tool bolted to the hand, metres
        grasp_mode: ``"magnet"`` (kinematic) or ``"physics"``
        calib_dir: Where the measured camera-to-world .npy files live
        workspace_min/workspace_max: Clamp box handed to the skills
        noise_m: Depth noise, metres
        verbose: Print each command
        layout_seed: Move each container to a random spot near its usual one
            (``bench.randomize_layout``); the same seed gives the same bench.
            None keeps the fixed layout, which the tests rely on.
        lab: Keep the books on what the containers hold -- powder a scoop
            carries, liquid, the kit's chemistry -- and draw it
            (:class:`robochem.sim.lab.Lab`). After every skill the changes
            are printed and added to its result under ``sim_lab``.

    Returns:
        A :class:`SimCell`.
    """
    from robochem.skills import SkillsExecutor

    install_rigid_transform()

    if layout_seed is not None:
        from .scene import cameras_seeing, load_camera_extrinsics, synthetic_cage
        cams = load_camera_extrinsics(calib_dir) or synthetic_cage()
        # A container the cage sees from fewer than three cameras is one
        # occlusion away from not being found at all.
        bench, notes = randomize_layout(
            bench or Bench(), int(layout_seed),
            visible=lambda xy: cameras_seeing(cams, (xy[0], xy[1], 0.02)) >= 3)
        placed = ", ".join(f"{p.name} ({p.pos[0]:.3f}, {p.pos[1]:.3f})"
                           for p in bench.props if p.kind == "container")
        print(f"[sim] layout seed {layout_seed}: {placed}")
        for note in notes:
            print(f"[sim]   {note}")

    scene = build_scene(bench=bench, calib_dir=calib_dir,
                        tool_length=tool_length, granules=granules)
    arm = SimFrankaArm(scene, viewer=viewer, realtime=realtime, speed=speed,
                       grasp_mode=grasp_mode, verbose=verbose)
    vision = SimVision(scene, noise_m=noise_m)
    skills = SkillsExecutor(
        robot_interface=arm,
        vision_system=vision,
        config={
            "workspace_min": list(workspace_min or DEFAULT_WORKSPACE_MIN),
            "workspace_max": list(workspace_max or DEFAULT_WORKSPACE_MAX),
            # Where the table is, for skills that need a container's inside
            # floor: perception's base_z reads 3-6 mm high (scoop).
            "table_z": float(scene.bench.table_z),
        },
    )
    cell = SimCell(scene=scene, arm=arm, vision=vision, skills=skills)
    if lab and not granules:
        cell.lab = Lab(cell, verbose=verbose)
        cell.lab.start()
        _report_lab_after_each_skill(cell)
    return cell


def _report_lab_after_each_skill(cell: SimCell) -> None:
    """
    Wrap the executor so every skill ends with what it did to the contents.

    Printed, and kept in the result under ``sim_lab`` for the run's record.
    The planner never sees it: what it knows comes from the cameras, as on
    the robot.
    """
    execute = cell.skills.execute

    def wrapped(skill_name, params=None, *args, **kwargs):
        before = cell.lab.snapshot()
        ok, result = execute(skill_name, params, *args, **kwargs)
        notes = cell.lab.changes_since(before)
        if notes:
            print("[lab] " + "\n[lab] ".join(notes))
            if isinstance(result, dict):
                result = dict(result)
                result["sim_lab"] = notes
        return ok, result

    cell.skills.execute = wrapped
    cell.skills.sim_lab = cell.lab
