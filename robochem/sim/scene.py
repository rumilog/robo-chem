"""
Builds the MuJoCo scene: Panda + Franka Hand, the camera cage, and the bench.

The arm comes from mujoco_menagerie's ``franka_emika_panda`` MJCF (fetched and
cached by ``robot_descriptions``), so the link geometry, joint limits and the
Franka Hand are the vendor model rather than something hand-rolled. Everything
else -- the ``franka_tool`` TCP site, the four RealSense viewpoints, the props --
is added on top with ``MjSpec``.

Two models come out of this:

  ``model``     the full scene, stepped and rendered
  ``ik_model``  arm and hand only, used by the IK solver so that free-floating
                props can never be "solved" into position by the optimiser
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mujoco
import numpy as np

from .bench import Bench, Prop

# Franka Hand TCP. libfranka's default F_T_EE puts the tool frame 103.4 mm past
# the flange, which is the fingertip plane; the menagerie ``hand`` body already
# carries the -45 degree mounting rotation, so no extra rotation is needed here.
TOOL_OFFSET_Z = 0.1034

# frankapy's FC.HOME_JOINTS -- the pose fa.reset_joints() drives to. The
# menagerie keyframe is a different rest pose, so use the robot's.
HOME_JOINTS = np.array([0.0, -math.pi / 4, 0.0, -3 * math.pi / 4,
                        0.0, math.pi / 2, math.pi / 4])

ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
FINGER_JOINTS = ["finger_joint1", "finger_joint2"]

# The cage cameras stream colour at 848x480 with depth aligned to it. A 42.5
# degree vertical field gives fy ~= 617 at 480 rows -- what the D435 colour
# intrinsics report -- and, across an 848-wide frame, the 69 degree horizontal
# field of the real sensor. So a simulated camera sees what a real one sees, and
# deprojection runs on the same numbers.
DEFAULT_FOVY = 42.5
DEFAULT_CAMERA_IDS = (2, 3, 4, 5)

# OpenCV cameras look down +z with +y down; MuJoCo cameras look down -z with
# +y up. The calibration .npy files are OpenCV camera-to-world.
CV_TO_MJ = np.diag([1.0, -1.0, -1.0])

WALL_SEGMENTS = 16

# Prop meshes are named relative to the repo root, so a bench entry reads the
# same whatever directory the run was launched from.
REPO_ROOT = Path(__file__).resolve().parents[2]


def _panda_mjcf() -> str:
    from robot_descriptions import panda_mj_description

    return panda_mj_description.MJCF_PATH


def load_camera_extrinsics(
    calib_dir: str = "calibration_out",
    camera_ids=DEFAULT_CAMERA_IDS,
) -> Dict[int, np.ndarray]:
    """
    Read the cage's measured camera-to-world transforms.

    Placing the simulated cameras where the real ones were calibrated is what
    makes simulated point clouds land in the same part of the workspace as the
    real ones, so perception-dependent geometry behaves the same.

    Returns:
        Camera ID -> 4x4 OpenCV camera-to-world matrix. Cameras whose file is
        missing are skipped; an empty dict falls back to a synthetic ring.
    """
    base = Path(calib_dir)
    out = {}
    for cam_id in camera_ids:
        path = base / f"realsense_camera{cam_id}w.npy"
        if path.exists():
            out[cam_id] = np.load(path, allow_pickle=True).astype(float)
    return out


def synthetic_cage(camera_ids=DEFAULT_CAMERA_IDS, centre=(0.5, 0.0, 0.05),
                   radius=0.45, height=0.45) -> Dict[int, np.ndarray]:
    """Four cameras on a ring looking at ``centre``, for when no calibration exists."""
    centre = np.asarray(centre, float)
    out = {}
    for i, cam_id in enumerate(camera_ids):
        angle = math.pi / 4 + i * math.pi / 2
        eye = np.array([centre[0] + radius * math.cos(angle),
                        centre[1] + radius * math.sin(angle),
                        height])
        forward = centre - eye
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0, 0, 1.0])
        right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        T = np.eye(4)
        T[:3, :3] = np.column_stack([right, down, forward])   # OpenCV axes
        T[:3, 3] = eye
        out[cam_id] = T
    return out


@dataclass
class SimScene:
    """Compiled scene plus the lookups the arm and the vision stub need."""

    model: mujoco.MjModel
    data: mujoco.MjData
    ik_model: mujoco.MjModel
    ik_data: mujoco.MjData
    bench: Bench
    camera_extrinsics: Dict[int, np.ndarray]
    prop_bodies: Dict[str, int] = field(default_factory=dict)      # prop name -> body id
    prop_qpos: Dict[str, int] = field(default_factory=dict)        # prop name -> freejoint qpos adr
    grain_bodies: Dict[str, List[int]] = field(default_factory=dict)
    arm_dofs: np.ndarray = field(default_factory=lambda: np.arange(7))
    tool_site: int = 0
    ik_tool_site: int = 0
    camera_ids: List[int] = field(default_factory=list)

    def prop_pose(self, name: str) -> np.ndarray:
        """4x4 world pose of a prop's body."""
        bid = self.prop_bodies[name]
        out = np.eye(4)
        out[:3, :3] = self.data.xmat[bid].reshape(3, 3)
        out[:3, 3] = self.data.xpos[bid]
        return out


def _add_container(spec, prop: Prop, table_z: float):
    """A hollow open cup: a disc base ringed by thin wall panels."""
    body = spec.worldbody.add_body(
        name=prop.body, pos=[prop.pos[0], prop.pos[1], table_z + prop.height / 2]
    )
    body.add_freejoint()

    half_h = prop.height / 2
    inner = prop.radius - prop.wall
    body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[prop.radius, prop.wall, 0],
        pos=[0, 0, -half_h + prop.wall],
        rgba=list(prop.rgba),
        density=500,
    )
    # A ring of boxes stands in for a cylindrical shell, which MuJoCo has no
    # primitive for. 16 panels is enough that granules do not squeeze out.
    panel_half_y = prop.radius * math.tan(math.pi / WALL_SEGMENTS)
    for i in range(WALL_SEGMENTS):
        angle = 2 * math.pi * i / WALL_SEGMENTS
        quat = np.zeros(4)
        mujoco.mju_axisAngle2Quat(quat, np.array([0.0, 0.0, 1.0]), angle)
        body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[prop.wall, panel_half_y, half_h],
            pos=[(inner + prop.wall) * math.cos(angle),
                 (inner + prop.wall) * math.sin(angle), 0],
            quat=quat.tolist(),
            rgba=list(prop.rgba),
            density=500,
        )
    return body


def _add_rod(spec, prop: Prop, table_z: float):
    """A spoon: a flat handle with a shallow bowl at one end."""
    body = spec.worldbody.add_body(
        name=prop.body, pos=[prop.pos[0], prop.pos[1], table_z + prop.wall]
    )
    body.add_freejoint()
    body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[prop.height / 2, prop.wall * 2, prop.wall],
        pos=[0, 0, 0],
        rgba=list(prop.rgba),
        density=2000,
    )
    body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
        size=[prop.radius * 1.4, prop.radius, prop.wall * 1.5],
        pos=[prop.height / 2 + prop.radius, 0, 0],
        rgba=list(prop.rgba),
        density=2000,
    )
    return body


def _add_holder(spec, prop: Prop, table_z: float):
    """
    A fixture the stirrer stands in, so the tool is always presented upright.

    Welded to the bench: no freejoint, so the rod going in cannot shove it.

    The bore is a ring of panels rather than a subtracted cylinder, which
    MuJoCo has no primitive for, and the chamfer at the top is that ring
    repeated at widening radii. That lead-in is the whole reason a blind
    insertion works: it catches a rod that arrives up to ``funnel_radius`` off
    axis and walks it down to the bore, so the placement only has to be as
    accurate as the chamfer is wide, not as accurate as the bore is tight.
    """
    bore = prop.bore_radius or 0.0075
    outer = prop.radius
    segments = 16

    body = spec.worldbody.add_body(
        name=prop.body, pos=[prop.pos[0], prop.pos[1], table_z]
    )
    # No freejoint: static.

    def ring(z_lo, z_hi, inner, name):
        """An annulus of boxes between two radii, spanning a height band."""
        mid = (inner + outer) / 2
        radial = (outer - inner) / 2
        tangential = mid * math.tan(math.pi / segments)
        for i in range(segments):
            angle = 2 * math.pi * i / segments
            quat = np.zeros(4)
            mujoco.mju_axisAngle2Quat(quat, np.array([0.0, 0.0, 1.0]), angle)
            body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                name=f"{prop.body}_{name}{i}",
                size=[radial, tangential, (z_hi - z_lo) / 2],
                pos=[mid * math.cos(angle), mid * math.sin(angle),
                     (z_lo + z_hi) / 2],
                quat=quat.tolist(),
                rgba=list(prop.rgba),
                density=1250, condim=4, friction=[1.0, 0.01, 0.001], group=3,
                # Printed plastic on plastic is stiff. The default contact
                # squashes for millimetres before it pushes back, which turns
                # the seating event into a slow ramp that no force threshold can
                # separate from the rod brushing the bore on the way down.
                solref=[0.005, 1.0], solimp=[0.95, 0.99, 0.0005, 0.5, 2.0],
            )

    # Solid below the bore, so a dropped rod stops where the real one would.
    body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        name=prop.body + "_floor",
        size=[outer, prop.bore_bottom / 2, 0],
        pos=[0, 0, prop.bore_bottom / 2],
        rgba=list(prop.rgba),
        density=1250, condim=4, friction=[1.0, 0.01, 0.001], group=3,
    )
    ring(prop.bore_bottom, prop.funnel_bottom, bore, "bore")

    # The chamfer, as a short stack of ever-wider rings.
    steps = 4
    for k in range(steps):
        z_lo = prop.funnel_bottom + (prop.height - prop.funnel_bottom) * k / steps
        z_hi = prop.funnel_bottom + (prop.height - prop.funnel_bottom) * (k + 1) / steps
        inner = bore + (prop.funnel_radius - bore) * (k + 0.5) / steps
        ring(z_lo, z_hi, inner, f"chamfer{k}_")

    body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=prop.body + "_mesh",
        pos=list(prop.mesh_pos),
        quat=list(prop.mesh_quat),
        rgba=list(prop.rgba),
        contype=0, conaffinity=0, density=0,
        group=2,
    )
    return body


def _add_mesh_stirrer(spec, prop: Prop, table_z: float, bench=None):
    """
    A stirrer: a grasp cube on top of a rod, standing cube-up on the bench.

    The body origin is the cube's centre, so ``pick_up`` grasping the cube and
    ``stir`` measuring immersion from the rod tip are talking about the same
    point. Like the scoop, the mesh is visual only -- it is what the cameras
    reconstruct -- and contact comes from two primitives in group 3, which the
    renderer does not draw.

    The rod's flat tip is what the prop balances on, so it is a cylinder rather
    than a capsule: a rounded tip is a point contact and the stirrer falls over
    before the arm has left home.
    """
    cube = prop.cube_half or 0.010
    rest = cube + prop.rod_length          # rod tip on the table
    if prop.seated_in and bench is not None:
        holder = bench.find(prop.seated_in)
        if holder is None:
            raise ValueError(f"{prop.name!r} is seated_in {prop.seated_in!r}, "
                             f"which is not on the bench")
        # Head underside on the holder's top face, rod hanging in the bore.
        rest = holder.height + cube

    body = spec.worldbody.add_body(
        name=prop.body, pos=[prop.pos[0], prop.pos[1], table_z + rest]
    )
    body.add_freejoint()

    body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=prop.body + "_mesh",
        pos=list(prop.mesh_pos),
        quat=list(prop.mesh_quat),
        rgba=list(prop.rgba),
        contype=0, conaffinity=0, density=0,
        group=2,
    )

    # Named for the same reason the scoop's panels are: when the stirrer clips
    # the cup, "the rod" and "the cube" call for different fixes.
    body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        name=prop.body + "_cube",
        size=[cube, cube, cube],
        pos=[0, 0, 0],
        rgba=list(prop.rgba),
        density=1250,                      # printed PLA
        condim=4, friction=[1.2, 0.01, 0.001], group=3,
        solref=[0.005, 1.0], solimp=[0.95, 0.99, 0.0005, 0.5, 2.0],
    )
    body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        name=prop.body + "_rod",
        size=[prop.rod_radius, prop.rod_length / 2, 0],
        pos=[0, 0, -(cube + prop.rod_length / 2)],
        rgba=list(prop.rgba),
        density=1250,
        condim=4, friction=[1.2, 0.01, 0.001], group=3,
    )
    return body


def _add_mesh_scoop(spec, prop: Prop, table_z: float):
    """
    A prop whose visible shape is its CAD file.

    The mesh geom is what the cameras see and what perception reconstructs;
    it carries no contacts, because MuJoCo collides meshes by their convex
    hull and the hull of an open scoop is a solid block -- granules could
    never enter the bowl. Contact comes from primitives in group 3, which the
    renderer does not draw, so the segmentation buffer stays mesh-only: a
    floor and four walls around the bowl, plus a box for the handle the jaws
    close on.
    """
    bowl = np.asarray(prop.bowl_size, float)
    centre = np.asarray(prop.bowl_offset, float)
    wall = prop.wall / 2                       # panel half-thickness
    rest = -(centre[2] - bowl[2])              # body origin above the table

    body = spec.worldbody.add_body(
        name=prop.body, pos=[prop.pos[0], prop.pos[1], table_z + rest]
    )
    body.add_freejoint()

    body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=prop.body + "_mesh",
        pos=list(prop.mesh_pos),
        quat=list(prop.mesh_quat),
        rgba=list(prop.rgba),
        contype=0, conaffinity=0, density=0,   # seen, never touched
        group=2,
    )

    def panel(size, pos, name):
        # Named, because when a stroke clips the cup the useful question is
        # WHICH part of the tool touched: "the bowl's leading wall" says
        # shorten the sweep, "the crank" says the tilt is wrong.
        body.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=list(size),
                      name=f"{prop.body}_{name}",
                      pos=list(pos), rgba=list(prop.rgba), density=1200,
                      condim=4, friction=[1.2, 0.01, 0.001], group=3)

    # Handle, centred on the body origin so the grasp check and the jaws agree.
    panel([prop.height / 2, prop.wall, prop.wall], [0, 0, 0], "handle")
    # The crank between handle and bowl.
    crank_x = (prop.height / 2 + centre[0] - bowl[0]) / 2
    panel([(centre[0] - bowl[0] - prop.height / 2) / 2, prop.wall,
           abs(centre[2]) / 2], [crank_x + prop.height / 2, 0, centre[2] / 2],
          "crank")
    # Bowl floor and four walls.
    panel([bowl[0], bowl[1], wall], centre + [0, 0, -bowl[2] + wall], "bowl_floor")
    for sx, sy, name in ((1, 0, "bowl_lead"), (-1, 0, "bowl_back"),
                         (0, 1, "bowl_left"), (0, -1, "bowl_right")):
        size = [wall, bowl[1], bowl[2]] if sx else [bowl[0], wall, bowl[2]]
        panel(size, centre + [sx * (bowl[0] - wall), sy * (bowl[1] - wall), 0],
              name)
    return body


def _add_label(spec, prop: Prop, table_z: float):
    """The handwritten paper a reagent cup stands on."""
    body = spec.worldbody.add_body(
        name=prop.body + "_label", pos=[prop.pos[0], prop.pos[1], table_z + 0.0005]
    )
    body.add_geom(
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[0.055, 0.04, 0.0005],
        rgba=[0.96, 0.93, 0.82, 1.0],
        contype=0, conaffinity=0,
    )
    return body


def _grain_sites(inner: float, radius: float, rng) -> List[Tuple[float, float]]:
    """
    Where one layer of granules can sit without overlapping.

    A hex lattice with a little jitter, not random placement. Dropping grains at
    random positions packs them dense enough to intersect at spawn, and MuJoCo
    resolves that intersection by flinging them: a cup of 110 granules emptied
    itself across the bench and left perception unable to find anything. The
    lattice guarantees the gap, the jitter keeps the bed from looking machined.
    """
    pitch = 2.3 * radius
    usable = max(0.0, inner - radius)
    sites = []
    rows = int(usable * 2 / (pitch * 0.866)) + 1
    for row in range(-rows, rows + 1):
        y = row * pitch * 0.866
        offset = (pitch / 2) if row % 2 else 0.0
        cols = int(usable * 2 / pitch) + 1
        for col in range(-cols, cols + 1):
            x = col * pitch + offset
            if math.hypot(x, y) <= usable:
                sites.append((x + rng.uniform(-0.15, 0.15) * radius,
                              y + rng.uniform(-0.15, 0.15) * radius))
    rng.shuffle(sites)
    return sites


def _add_grains(spec, prop: Prop, table_z: float) -> List[str]:
    """Loose granules resting in a reagent cup, so pours and scoops show flow."""
    names = []
    rng = np.random.default_rng(abs(hash(prop.name)) % (2**32))
    inner = prop.radius - prop.wall * 2
    sites = _grain_sites(inner, prop.grain_radius, rng)
    if not sites:
        return names
    for i in range(prop.fill):
        x, y = sites[i % len(sites)]
        name = f"{prop.body}_grain{i}"
        body = spec.worldbody.add_body(
            name=name,
            pos=[prop.pos[0] + x, prop.pos[1] + y,
                 table_z + prop.wall * 2 + prop.grain_radius * 2
                 + 2.3 * prop.grain_radius * (i // len(sites))],
        )
        body.add_freejoint()
        body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[prop.grain_radius, 0, 0],
            rgba=list(prop.fill_rgba),
            density=1200,
            condim=4,
            friction=[1.2, 0.01, 0.001],
        )
        names.append(name)
    return names


def _add_cameras(spec, extrinsics: Dict[int, np.ndarray], fovy: float):
    for cam_id, T in sorted(extrinsics.items()):
        rot_mj = np.asarray(T[:3, :3], float) @ CV_TO_MJ
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, np.ascontiguousarray(rot_mj).reshape(9))
        spec.worldbody.add_camera(
            name=f"cam{cam_id}",
            pos=list(np.asarray(T[:3, 3], float)),
            quat=quat.tolist(),
            fovy=fovy,
        )


def _base_spec(tool_length: float = 0.0, tool_radius: float = 0.006):
    """
    Panda MJCF with the ``franka_tool`` site added, shared by both models.

    ``tool_length`` mounts a fixed tool on the hand (for previewing a bolted-on
    end effector). The default of 0 leaves the stock Franka Hand, because the
    lab's spoon is a prop the arm picks up rather than a fixture.
    """
    spec = mujoco.MjSpec.from_file(_panda_mjcf())

    # The real controller compensates gravity internally, so a commanded pose
    # is held rather than sagging under load. Without this the menagerie
    # position servos sit a few millimetres of joint error below every target
    # (~28 Nm on joint 2 against a 4500 gain) and the preview understates where
    # the TCP ends up. This must happen before compile: mjModel.ngravcomp is
    # counted there, and setting body_gravcomp afterwards is silently ignored.
    # Only the arm exists at this point; props are added later and still fall.
    for body in spec.bodies:
        body.gravcomp = 1.0

    hand = spec.body("hand")
    hand.add_site(name="franka_tool", pos=[0, 0, TOOL_OFFSET_Z + tool_length],
                  size=[0.005, 0, 0], rgba=[1, 0.3, 0, 0.6])
    if tool_length > 0:
        tool = hand.add_body(name="mounted_tool", pos=[0, 0, TOOL_OFFSET_Z])
        tool.add_geom(
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            size=[tool_radius, tool_length / 2, 0],
            pos=[0, 0, tool_length / 2],
            rgba=[0.8, 0.8, 0.85, 1.0],
            density=2000,
        )
    return spec


def build_scene(
    bench: Optional[Bench] = None,
    calib_dir: str = "calibration_out",
    camera_ids=DEFAULT_CAMERA_IDS,
    fovy: float = DEFAULT_FOVY,
    tool_length: float = 0.0,
    granules: bool = False,
    timestep: float = 0.002,
) -> SimScene:
    """
    Compile the full scene and the IK-only arm model.

    Args:
        bench: Props on the table (default: the README's validated layout)
        calib_dir: Where the measured camera-to-world .npy files live
        camera_ids: Which cage cameras to place
        fovy: Vertical field of view, degrees
        tool_length: Length of a fixed tool bolted to the hand, metres
        granules: Put loose particles in reagent cups so pours/scoops show flow
        timestep: Physics timestep

    Returns:
        A compiled :class:`SimScene`.
    """
    bench = bench or Bench()
    extrinsics = load_camera_extrinsics(calib_dir, camera_ids)
    if not extrinsics:
        print(f"[sim] No calibration in {calib_dir}; using a synthetic camera ring")
        extrinsics = synthetic_cage(camera_ids)

    ik_spec = _base_spec(tool_length)
    ik_model = ik_spec.compile()

    spec = _base_spec(tool_length)
    spec.option.timestep = timestep
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

    spec.worldbody.add_geom(
        name="table",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[1.5, 1.5, 0.05],
        pos=[0, 0, bench.table_z],
        rgba=list(bench.table_rgba),
    )
    spec.worldbody.add_light(pos=[0.5, 0, 1.6], dir=[0, 0, -1], castshadow=0)
    spec.worldbody.add_light(pos=[0.0, -0.8, 1.2], dir=[0.3, 0.6, -1], castshadow=0)

    _add_cameras(spec, extrinsics, fovy)

    for prop in bench.props:
        if prop.mesh:
            path = Path(prop.mesh)
            if not path.is_absolute():
                path = REPO_ROOT / path
            if not path.exists():
                raise FileNotFoundError(f"{prop.name!r} wants a mesh that is "
                                        f"not there: {path}")
            spec.add_mesh(name=prop.body + "_mesh", file=str(path),
                          scale=[prop.mesh_scale] * 3)

    grain_names: Dict[str, List[str]] = {}
    for prop in bench.props:
        if prop.label:
            _add_label(spec, prop, bench.table_z)
        if prop.kind == "holder":
            _add_holder(spec, prop, bench.table_z)
        elif prop.mesh and prop.kind == "stirrer":
            _add_mesh_stirrer(spec, prop, bench.table_z, bench)
        elif prop.mesh:
            _add_mesh_scoop(spec, prop, bench.table_z)
        elif prop.kind == "rod":
            _add_rod(spec, prop, bench.table_z)
        else:
            _add_container(spec, prop, bench.table_z)
        if granules and prop.fill:
            grain_names[prop.name] = _add_grains(spec, prop, bench.table_z)

    model = spec.compile()
    data = mujoco.MjData(model)

    def bid(name):
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)

    prop_bodies = {p.name: bid(p.body) for p in bench.props}
    # Static props are welded to the world and have no freejoint to address.
    prop_qpos = {
        p.name: int(model.jnt_qposadr[model.body_jntadr[prop_bodies[p.name]]])
        for p in bench.props
        if model.body_jntadr[prop_bodies[p.name]] >= 0
    }
    grain_bodies = {k: [bid(n) for n in v] for k, v in grain_names.items()}

    scene = SimScene(
        model=model,
        data=data,
        ik_model=ik_model,
        ik_data=mujoco.MjData(ik_model),
        bench=bench,
        camera_extrinsics=extrinsics,
        prop_bodies=prop_bodies,
        prop_qpos=prop_qpos,
        grain_bodies=grain_bodies,
        tool_site=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "franka_tool"),
        ik_tool_site=mujoco.mj_name2id(ik_model, mujoco.mjtObj.mjOBJ_SITE, "franka_tool"),
        camera_ids=sorted(extrinsics.keys()),
    )
    reset_scene(scene)
    return scene


def reset_scene(scene: SimScene) -> None:
    """Put the arm at frankapy's home pose and every prop back on its mark."""
    model, data = scene.model, scene.data
    mujoco.mj_resetData(model, data)
    data.qpos[:7] = HOME_JOINTS
    data.qpos[7:9] = 0.04                      # fingers open
    data.ctrl[:7] = HOME_JOINTS
    data.ctrl[7] = 255                         # gripper actuator: 255 = 80 mm
    mujoco.mj_forward(model, data)
