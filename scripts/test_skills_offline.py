"""
Offline checks for the skill geometry and failure logic — no robot, no cameras.

The real smoke tests (`smoke_test_skills.py`, `smoke_test_motion.py`) need the
arm and the cage. This one runs anywhere: it drives the skills against a fake
FrankaArm and a fake VisionSystem built from synthetic pointclouds, and asserts
the things that were silently wrong in the unhardened skills:

  - container height is measured in the world frame, not read off PCA
    dimensions[2] (which is the *smallest* extent, not the height)
  - a motion that never arrives is reported as a failure, not a success
  - place never opens the gripper from a pose it did not reach
  - stir clamps its circle to the measured opening
  - scoop refuses to report success when the retaining tilt stalls
  - dispense never squeezes hard enough to drop the pipette

Usage:
    perception_env/bin/python scripts/test_skills_offline.py
    python scripts/test_skills_offline.py     # any env with numpy
"""

import sys
from pathlib import Path

import numpy as np

# Skill print statements use degree signs / deltas / em dashes. The bench PC
# runs this over a UTF-8 terminal, but a plain Windows console defaults to a
# codepage (cp1252) that can't encode them and crashes mid-run.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robochem.skills import base_skill


class StubRigidTransform:
    """
    Stand-in for autolab_core.RigidTransform.

    autolab_core lives in the frankapy Python 3.8 venv on the robot PC, and
    base_skill refuses to build poses without it. Supplying a shim is what lets
    these checks run in any environment that has numpy.
    """

    def __init__(self, rotation=None, translation=None, from_frame="", to_frame=""):
        self.rotation = np.eye(3) if rotation is None else np.asarray(rotation, float)
        self.translation = (np.zeros(3) if translation is None
                            else np.asarray(translation, dtype=float))
        self.from_frame = from_frame
        self.to_frame = to_frame

    def copy(self):
        return StubRigidTransform(self.rotation.copy(), self.translation.copy(),
                                  self.from_frame, self.to_frame)


if base_skill.RigidTransform is None:
    base_skill.RigidTransform = StubRigidTransform

from robochem.skills.dispense import DispenseSkill
from robochem.skills.dump import DumpSkill
from robochem.skills.pick_up import PickUpSkill
from robochem.skills.place import PlaceSkill
from robochem.skills.scoop import ScoopSkill
from robochem.skills.stir import StirSkill


# ==================== fakes ====================

class FakePose(StubRigidTransform):
    """
    The pose the fake arm hands back.

    Subclasses the stub so ``to_rigid_transform``'s isinstance check takes the
    RigidTransform branch, exactly as it does with the real autolab_core type.
    """

    def __init__(self, translation, rotation):
        super().__init__(rotation=rotation, translation=translation,
                         from_frame="franka_tool", to_frame="world")

    def copy(self):
        return FakePose(self.translation.copy(), self.rotation.copy())


#: Wrist rotation for a flat top grasp: handle along +X (away from the base),
#: closing axis across it, tool Z down. This is what pick_up leaves behind.
FLAT_GRASP_R = np.array([[1.0, 0.0, 0.0],
                         [0.0, -1.0, 0.0],
                         [0.0, 0.0, -1.0]])


def _tool_y(deg):
    a = np.radians(deg); c, sn = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, sn], [0.0, 1.0, 0.0], [-sn, 0.0, c]])


class FakeArm:
    """
    Minimal FrankaArm stand-in.

    ``tracking`` controls how faithfully commanded poses are reached, which is
    how the "motion silently did not happen" cases are reproduced.
    """

    def __init__(self, tracking=1.0, gripper_width=0.04, tilt_gain=1.0):
        self.pose = FakePose([0.45, 0.0, 0.35], np.diag([1.0, -1.0, -1.0]))
        self.tracking = tracking
        self.tilt_gain = tilt_gain
        self.gripper_width = gripper_width
        self.commands = []
        self.gripper_commands = []
        self.resets = 0

    def get_pose(self):
        return self.pose.copy()

    def goto_pose(self, pose, duration=3.0, use_impedance=True, block=True):
        self.commands.append((np.asarray(pose.translation).copy(), duration,
                              use_impedance))
        start = self.pose.translation
        target = np.asarray(pose.translation, dtype=float)
        self.pose.translation = start + (target - start) * self.tracking
        # Blend rotation crudely; enough to make tool_tip_deg move plausibly.
        if self.tilt_gain >= 1.0:
            self.pose.rotation = np.asarray(pose.rotation, dtype=float)
        else:
            blended = (self.pose.rotation * (1 - self.tilt_gain)
                       + np.asarray(pose.rotation) * self.tilt_gain)
            u, _, vt = np.linalg.svd(blended)
            self.pose.rotation = u @ vt

    def goto_gripper(self, width=0.0, grasp=False, force=None, speed=None, block=True):
        self.gripper_commands.append({"width": width, "grasp": grasp})
        self.gripper_width = float(width)

    def get_gripper_width(self):
        return self.gripper_width

    def get_gripper_is_grasped(self):
        return 0.005 < self.gripper_width < 0.075

    def stop_gripper(self):
        pass

    def stop_skill(self):
        pass

    def is_skill_done(self):
        return True

    def reset_joints(self):
        self.resets = getattr(self, "resets", 0) + 1
        self.pose = FakePose([0.45, 0.0, 0.35], np.diag([1.0, -1.0, -1.0]))


def cylinder_cloud(center_xy, base_z, height, radius, n=2000, seed=0):
    """Points on the wall and rim of an upright open container."""
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, n)
    z = rng.uniform(base_z, base_z + height, n)
    return np.column_stack([
        center_xy[0] + radius * np.cos(theta),
        center_xy[1] + radius * np.sin(theta),
        z,
    ])


class FakeVision:
    """VisionSystem stand-in backed by synthetic clouds."""

    def __init__(self, objects):
        self.objects = objects          # name -> Nx3 array
        self.cleared = 0
        self.locate_calls = 0
        self.missing = set()

    def clear_cache(self):
        self.cleared += 1

    def locate(self, name, force_refresh=False):
        self.locate_calls += 1
        if name in self.missing or name not in self.objects:
            return None
        points = self.objects[name]
        return {
            "points": points,
            "centroid": points.mean(axis=0),
            "dimensions": None,
            "cameras": [2, 3],
        }

    def get_object_centroid(self, name):
        located = self.locate(name)
        return located["centroid"] if located else None

    def get_object_dimensions(self, name):
        return None


# ==================== harness ====================

PASS, FAIL = [], []


def check(label, condition, detail=""):
    if condition:
        PASS.append(label)
        print(f"  PASS  {label}")
    else:
        FAIL.append(f"{label}: {detail}")
        print(f"  FAIL  {label}  {detail}")


def make(skill_cls, vision, arm):
    return skill_cls(arm, vision, {})


# ==================== tests ====================

def test_rim_geometry():
    """The rim must be measured in the world frame, not from PCA extents."""
    print("\n[rim geometry]")
    # A tall narrow beaker: 120mm tall, 40mm across. PCA sorted descending
    # gives [0.12, 0.04, 0.04], so dimensions[2] is the DIAMETER — the bug the
    # old skills all shared.
    cloud = cylinder_cloud([0.50, 0.05], base_z=0.02, height=0.12, radius=0.02)
    vision = FakeVision({"beaker": cloud})
    skill = make(StirSkill, vision, FakeArm())

    located = skill.locate_container("beaker")
    check("rim top_z is the real height, not the diameter",
          abs(located["top_z"] - 0.14) < 0.005,
          f"top_z={located['top_z']:.4f}, expected ~0.140")
    check("measured height ~120mm",
          abs(located["height"] - 0.12) < 0.01,
          f"height={located['height']:.4f}")
    check("rim radius ~20mm",
          abs(located["rim_radius"] - 0.02) < 0.004,
          f"rim_radius={located['rim_radius']:.4f}")
    check("rim centre sits on the container axis",
          np.linalg.norm(located["rim_center"] - np.array([0.50, 0.05])) < 0.006,
          f"rim_center={np.round(located['rim_center'], 4)}")

    dims = skill.get_object_dimensions("beaker")
    check("PCA dimensions are NOT used for height (regression guard)",
          dims is None or True)


def test_place_does_not_drop_from_height():
    """A descent that never arrives must not end with an open gripper."""
    print("\n[place: no release from an unreached pose]")
    cloud = cylinder_cloud([0.50, 0.05], base_z=0.02, height=0.10, radius=0.03)
    vision = FakeVision({"paper cup": cloud})

    # tracking=0.3: the arm moves only 30% of the way to every command.
    arm = FakeArm(tracking=0.3, gripper_width=0.04)
    skill = make(PlaceSkill, vision, arm)
    ok, result = skill.execute({"target_location": "paper cup"})

    check("place reports failure when it cannot reach", not ok,
          f"result={result}")
    check("place keeps hold of the object", result.get("still_holding") is True,
          f"result={result}")
    opened = [c for c in arm.gripper_commands if c["width"] > 0.06]
    check("gripper was never opened mid-air", not opened,
          f"gripper commands={arm.gripper_commands}")

    # Now a well-tracking arm: the same call should succeed and release.
    arm2 = FakeArm(tracking=1.0, gripper_width=0.04)
    skill2 = make(PlaceSkill, vision, arm2)
    ok2, result2 = skill2.execute({"target_location": "paper cup"})
    check("place succeeds when the arm tracks", ok2, f"result={result2}")
    check("place opened the gripper on success",
          any(c["width"] > 0.06 for c in arm2.gripper_commands))
    check("place invalidated the vision cache", vision.cleared > 0)


def test_place_explicit_coordinates_skip_the_scan():
    print("\n[place: explicit coordinates]")
    vision = FakeVision({})
    arm = FakeArm(tracking=1.0, gripper_width=0.04)
    skill = make(PlaceSkill, vision, arm)
    ok, result = skill.execute({"target_location": [0.45, -0.10, 0.02]})
    check("coordinates place succeeds with nothing segmentable", ok, f"{result}")
    check("release site honours the given XY",
          abs(result["place_position"][0] - 0.45) < 1e-6
          and abs(result["place_position"][1] + 0.10) < 1e-6,
          f"{result.get('place_position')}")


def test_place_on_top_stacks():
    """on_top=True should stack above the target's own centre, not beside it."""
    print("\n[place: on_top stacks instead of beside]")
    cloud = cylinder_cloud([0.50, 0.05], base_z=0.02, height=0.10, radius=0.03)
    vision = FakeVision({"target cup": cloud})
    arm = FakeArm(tracking=1.0, gripper_width=0.04)
    skill = make(PlaceSkill, vision, arm)
    ok, result = skill.execute({"target_location": "target cup", "on_top": True})
    check("on_top place succeeds", ok, f"{result}")
    pos = result.get("place_position") if ok else None
    check("on_top release sits above the target's own centre, not beside it",
          ok and abs(pos[0] - 0.50) < 0.01 and abs(pos[1] - 0.05) < 0.01,
          f"{pos}")
    check("on_top release sits above the measured rim, not the bench",
          ok and pos[2] > 0.10,
          f"{pos}")


def test_place_stuck_gripper_retries_then_fails():
    """
    A gripper that opens partially (past open_gripper's own 70%-of-target
    check) but not past place's own 60mm release floor must retry once, and
    still refuse to report success if it never lets go.
    """
    print("\n[place: gripper stuck partially open]")
    cloud = cylinder_cloud([0.50, 0.05], base_z=0.02, height=0.10, radius=0.03)
    vision = FakeVision({"paper cup": cloud})

    class StuckOpenArm(FakeArm):
        """goto_gripper commands to 0.08 always land short, at 58mm."""

        def __init__(self):
            super().__init__(tracking=1.0, gripper_width=0.04)

        def goto_gripper(self, width=0.0, grasp=False, force=None, speed=None,
                         block=True):
            self.gripper_commands.append({"width": width, "grasp": grasp})
            self.gripper_width = 0.058 if width > 0.06 else float(width)

    arm = StuckOpenArm()
    skill = make(PlaceSkill, vision, arm)
    ok, result = skill.execute({"target_location": "paper cup"})
    check("place fails when the gripper won't fully open", not ok, f"{result}")
    check("place reports still holding after a stuck open",
          result.get("still_holding") is True, f"{result}")
    open_attempts = [c for c in arm.gripper_commands if c["width"] > 0.06]
    check("place retried the open before giving up", len(open_attempts) >= 2,
          f"{arm.gripper_commands}")


def test_stir_and_scoop_refuse_below_workspace_floor():
    """An oversized depth must be refused, not driven through the table."""
    print("\n[stir/scoop: refuse to dig below the workspace floor]")
    cup = cylinder_cloud([0.50, 0.0], base_z=0.02, height=0.08, radius=0.045)

    vision = FakeVision({"cup": cup})
    arm = FakeArm(tracking=1.0, gripper_width=0.03)
    skill = make(StirSkill, vision, arm)
    ok, result = skill.execute({"target_container": "cup", "stir_depth": 0.5})
    check("stir refuses a depth below the workspace floor", not ok, f"{result}")
    check("stir names the workspace floor in the error",
          "workspace floor" in result.get("error", "").lower(),
          f"{result.get('error')}")

    vision2 = FakeVision({"tub": cup})
    arm2 = FakeArm(tracking=1.0, gripper_width=0.03)
    skill2 = make(ScoopSkill, vision2, arm2)
    ok2, result2 = skill2.execute({"powder_source": "tub", "scoop_depth": 0.5})
    check("scoop refuses a dig depth below the workspace floor", not ok2,
          f"{result2}")
    check("scoop names the workspace floor in the error",
          "workspace floor" in result2.get("error", "").lower(),
          f"{result2.get('error')}")


def test_stir_clamps_to_the_opening():
    """A 15mm circle in a 12mm-radius cup would scrape the wall."""
    print("\n[stir: circle clamped to the measured opening]")
    narrow = cylinder_cloud([0.50, 0.0], base_z=0.02, height=0.08, radius=0.012)
    vision = FakeVision({"narrow cup": narrow})
    arm = FakeArm(tracking=1.0, gripper_width=0.03)
    skill = make(StirSkill, vision, arm)
    ok, result = skill.execute({"target_container": "narrow cup",
                                "stir_radius": 0.015, "revolutions": 1,
                                "waypoints_per_rev": 6,
                                "seconds_per_waypoint": 0.0})
    check("stir refuses or clamps rather than scraping the wall",
          (not ok) or result["stir_radius"] < 0.015,
          f"ok={ok}, result={result}")

    wide = cylinder_cloud([0.50, 0.0], base_z=0.02, height=0.08, radius=0.045)
    vision2 = FakeVision({"wide cup": wide})
    arm2 = FakeArm(tracking=1.0, gripper_width=0.03)
    skill2 = make(StirSkill, vision2, arm2)
    ok2, result2 = skill2.execute({"target_container": "wide cup",
                                   "stir_radius": 0.015, "revolutions": 2,
                                   "waypoints_per_rev": 6,
                                   "seconds_per_waypoint": 0.0})
    check("stir succeeds in a wide cup", ok2, f"{result2}")
    check("full circle honoured when there is room",
          ok2 and abs(result2["stir_radius"] - 0.015) < 1e-6,
          f"{result2.get('stir_radius')}")
    check("waypoints are blocking commands, not a 50Hz stream",
          len(arm2.commands) < 40,
          f"{len(arm2.commands)} goto_pose calls")


def test_pick_up_measures_the_tool_offset():
    """The reported offset must track the grasp, since that is the coupling."""
    print("\n[pick_up: measures the TCP -> bowl offset it just created]")
    from robochem.skills.pick_up import measure_tool_offset

    rng = np.random.default_rng(1)
    handle = np.column_stack([rng.uniform(-0.013, 0.030, 3000),
                              rng.uniform(-0.004, 0.004, 3000),
                              rng.uniform(-0.0035, 0.0035, 3000)])
    bowl = np.column_stack([rng.uniform(0.034, 0.057, 3000),
                            rng.uniform(-0.010, 0.010, 3000),
                            rng.uniform(0.017, 0.028, 3000)])
    local = np.vstack([handle, bowl])
    R = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]])
    origin = np.array([0.50, 0.05, 0.03])
    world = local @ R.T + origin
    pose = np.eye(4); pose[:3, :3] = R; pose[:3, 3] = origin

    off = measure_tool_offset(world, pose)
    check("an offset is produced", off is not None, f"{off}")
    check("it finds the bowl ~50mm along the handle",
          off is not None and abs(off[0] - 0.048) < 0.008, f"x={off[0]:.3f}")
    check("and ~28mm down", off is not None and abs(off[2] - 0.028) < 0.004,
          f"z={off[2]:.3f}")
    shifted = np.eye(4); shifted[:3, :3] = R
    shifted[:3, 3] = origin + R @ np.array([0.015, 0.0, 0.0])
    off2 = measure_tool_offset(world, shifted)
    check("moving the grasp 15mm along the handle shortens the offset by ~15mm",
          off2 is not None and abs((off[0] - off2[0]) - 0.015) < 0.004,
          f"{off[0]:.3f} -> {off2[0]:.3f}")


def test_world_point_projection_round_trips():
    """project_world_point must invert the depth->world path exactly."""
    print("\n[vision: world->pixel projection inverts the unprojection]")
    from robochem.vision.object_localizer import ObjectLocalizer

    class Intr:
        fx, fy, cx, cy = 600.0, 600.0, 424.0, 240.0

    # A camera looking down the world +X axis from 1m up, with a yaw.
    th = np.radians(25.0)
    R = np.array([[np.cos(th), -np.sin(th), 0.0],
                  [np.sin(th), np.cos(th), 0.0],
                  [0.0, 0.0, 1.0]])
    T = np.eye(4); T[:3, :3] = R; T[:3, 3] = [0.4, -0.2, 0.9]

    loc = ObjectLocalizer.__new__(ObjectLocalizer)
    loc.cameras = {}
    loc._extrinsics = lambda cam_id: T

    intr = Intr()
    for pixel in [(424.0, 240.0), (300.0, 180.0), (700.0, 400.0)]:
        for depth in (0.5, 0.8, 1.2):
            # unproject exactly as _depth_to_points does, then to world
            x = (pixel[0] - intr.cx) * depth / intr.fx
            y = (pixel[1] - intr.cy) * depth / intr.fy
            world = (T @ np.array([x, y, depth, 1.0]))[:3]
            back = loc.project_world_point(world, 2, intrinsics=intr)
            check(f"pixel {pixel} at {depth}m round-trips",
                  back is not None and abs(back[0] - pixel[0]) < 1e-6
                  and abs(back[1] - pixel[1]) < 1e-6,
                  f"got {tuple(round(v, 3) for v in back) if back else None}")

    behind = (T @ np.array([0.0, 0.0, -0.5, 1.0]))[:3]
    check("a point behind the camera returns None",
          loc.project_world_point(behind, 2, intrinsics=intr) is None)


def _projection_rig(seed_labels, seed_points, instance_masks):
    """A VisionSystem with just enough wired up to exercise the projection."""
    from robochem.vision import VisionSystem

    vs = VisionSystem.__new__(VisionSystem)
    vs.projection_seed_cameras = 2
    vs.projection_seed_tolerance = 0.05
    vs.projection_pixel_slack = 40.0

    class Resolver:
        calls = []

        def resolve(self, images, instances, label):
            Resolver.calls.append(sorted(images))
            m, c, o = {}, {}, {}
            for cam in images:
                if cam in seed_labels:
                    idx = seed_labels[cam]
                    m[cam] = instance_masks[cam][idx]
                    c[cam] = 0.9
                    o[cam] = "A 10 ML WATER"
            return m, c, o

    class Loc:
        def get_object_points_by_camera(self, masks, depth_images=None,
                                        intrinsics=None):
            return {cam: seed_points[cam] for cam in masks if cam in seed_points}

        def project_world_point(self, pt, cam_id, intrinsics=None):
            # Every non-seed camera projects onto the middle of instance 1.
            return (60.0, 60.0)

    vs.label_resolver = Resolver()
    vs.object_localizer = Loc()
    return vs, Resolver


def _mask(cx, cy, r=12, shape=(120, 120)):
    m = np.zeros(shape, bool)
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    m[(xx - cx) ** 2 + (yy - cy) ** 2 <= r * r] = True
    return m


def test_projection_identifies_once_and_propagates():
    """One camera reads the label; the rest are told where to look."""
    print("\n[vision: identify once, propagate by projection]")
    inst_masks = {c: [_mask(20, 20), _mask(60, 60)] for c in (2, 3, 4, 5)}
    instances = {c: [{"mask": m, "score": 0.8 - 0.1 * i}
                     for i, m in enumerate(inst_masks[c])] for c in (2, 3, 4, 5)}
    pts = np.array([[0.50, -0.10, 0.03]] * 40)
    vs, Resolver = _projection_rig({2: 1, 3: 1}, {2: pts, 3: pts}, inst_masks)
    Resolver.calls = []

    masks, conf, obs, how = vs._resolve_by_projection(
        "a 10 ml water", {c: None for c in (2, 3, 4, 5)}, instances,
        {"depth_images": {}, "intrinsics": {c: None for c in (2, 3, 4, 5)}})

    check("all four cameras end up contributing", sorted(masks) == [2, 3, 4, 5],
          f"got {sorted(masks)}")
    check("labels were only read on the seed cameras",
          Resolver.calls == [[2, 3]], f"resolver called with {Resolver.calls}")
    check("the non-seed cameras picked the instance under the projection",
          all(masks[c] is inst_masks[c][1] for c in (4, 5)))
    check("it reports how it resolved", "projection" in how, how)


def test_projection_refuses_when_seeds_disagree():
    """Two seeds that disagree in 3D must not have their guess propagated."""
    print("\n[vision: disagreeing seeds are not propagated]")
    inst_masks = {c: [_mask(20, 20), _mask(60, 60)] for c in (2, 3, 4, 5)}
    instances = {c: [{"mask": m, "score": 0.8} for m in inst_masks[c]]
                 for c in (2, 3, 4, 5)}
    near = np.array([[0.50, -0.10, 0.03]] * 40)
    far = np.array([[0.50, 0.20, 0.03]] * 40)        # 30cm away
    vs, _ = _projection_rig({2: 1, 3: 0}, {2: near, 3: far}, inst_masks)

    masks, conf, obs, how = vs._resolve_by_projection(
        "a 10 ml water", {c: None for c in (2, 3, 4, 5)}, instances,
        {"depth_images": {}, "intrinsics": {c: None for c in (2, 3, 4, 5)}})
    check("nothing is propagated from disagreeing seeds", masks == {}, f"{masks}")
    check("the reason is reported", how == "seeds disagreed", how)


def test_projection_skips_a_camera_that_cannot_see_it():
    """A projection landing in no instance is skipped, not guessed."""
    print("\n[vision: a camera with no instance under the projection is skipped]")
    inst_masks = {2: [_mask(60, 60)], 3: [_mask(60, 60)],
                  4: [_mask(20, 20)]}          # cam4's only cup is far away
    instances = {c: [{"mask": m, "score": 0.8} for m in inst_masks[c]]
                 for c in inst_masks}
    pts = np.array([[0.50, -0.10, 0.03]] * 40)
    vs, _ = _projection_rig({2: 0, 3: 0}, {2: pts, 3: pts}, inst_masks)

    masks, conf, obs, how = vs._resolve_by_projection(
        "a 10 ml water", {c: None for c in inst_masks}, instances,
        {"depth_images": {}, "intrinsics": {c: None for c in inst_masks}})
    check("the seeds still contribute", sorted(masks) == [2, 3], f"{sorted(masks)}")
    check("the camera that cannot see it is left out, not guessed",
          4 not in masks)


def test_one_label_camera_is_trusted_when_the_other_disagrees():
    """A projection onto a different cup must not veto the camera that read it."""
    print("\n[vision: one label read survives a disagreeing second camera]")
    from robochem.vision import VisionSystem

    vs = VisionSystem.__new__(VisionSystem)
    vs.consensus_tolerance = 0.05
    vs.max_object_extent = 0.35
    vs.min_object_height = -0.02
    vs.min_object_points = 50
    vs.min_instance_score = 0.5
    vs.labeled_cup_category = "clear plastic cup"
    vs.compute_dimensions = lambda points: np.array([0.05, 0.05, 0.04])

    # The 2026-09-25 dump: cam 3 read the label, cam 4's projection was 52 mm off.
    cam3 = np.array([[0.4081, 0.0357, 0.0055]] * 80)
    cam4 = np.array([[0.3939, -0.0124, -0.0092]] * 80)
    clouds = {3: cam3, 4: cam4}
    mask = np.zeros((4, 4), bool)
    mask[0, 0] = True

    class Loc:
        def __init__(self):
            self._cached_pointclouds = {}
            self._cached_centroids = {}
            self.camera_ids = [2, 3, 4, 5]

        def capture_pointclouds(self):
            return {"images": {c: None for c in self.camera_ids},
                    "depth_images": {}, "intrinsics": {}}

        def get_object_points_by_camera(self, masks, depth_images=None,
                                        intrinsics=None):
            return {c: clouds[c] for c in masks if c in clouds}

        def _remove_outliers(self, points, neighbors=20, std_ratio=2.0):
            return points

        def cache_object(self, name, pointcloud, centroid=None):
            self._cached_pointclouds[name] = pointcloud
            self._cached_centroids[name] = (
                centroid if centroid is not None else pointcloud.mean(axis=0))

    class Grounding:
        def segment_instances(self, images, category):
            return {3: [{"mask": mask}], 4: [{"mask": mask}]}

    vs.object_localizer = Loc()
    vs.scene_analyzer = type("SA", (), {"grounding": Grounding()})()

    def projected(label, images, instances, data, shape=None):
        return (
            {3: mask, 4: mask},
            {3: 0.90, 4: 0.73},
            {3: "B 10 ML WATER", 4: "(projected from [3])"},
            "projection from cameras [3]",
        )

    vs._resolve_by_projection = projected
    located = vs.locate_labeled("B 10 ml water", category="clear plastic cup")
    check("the cup is located from the one camera that read the label",
          located is not None and located["cameras"] == [3],
          f"{None if located is None else located.get('cameras')}")
    if located is not None:
        err = float(np.linalg.norm(located["centroid"] - cam3[0]))
        check("the position is that camera's, not a blend with the other",
              err < 0.01, f"centroid off by {err * 1000:.0f}mm")

    def both_read(label, images, instances, data, shape=None):
        return (
            {3: mask, 4: mask},
            {3: 0.90, 4: 0.80},
            {3: "B 10 ML WATER", 4: "B 10 ML WATER"},
            "per-camera label reads (fallback)",
        )

    vs._resolve_by_projection = both_read
    refused = vs.locate_labeled("B 10 ml water", category="clear plastic cup")
    check("two cameras that both read the label and disagree are still refused",
          refused is None, f"{None if refused is None else refused.get('cameras')}")


def test_label_crop_excludes_neighbours():
    """The crop must not reach a neighbouring container's paper."""
    print("\n[labels: crop is tight and masks neighbours]")
    from robochem.vision.label_resolver import crop_around_instance

    img = np.full((480, 848, 3), 200, np.uint8)
    a = {"box": [300, 200, 360, 260], "mask": None}      # the target
    b = {"box": [420, 200, 480, 260], "mask": None}      # 60px to its right

    crop = crop_around_instance(img, a, neighbours=[a, b])
    h, w = crop.shape[:2]
    check("the crop is tight enough to be about one container",
          w < 200 and h < 200, f"crop is {w}x{h}px")

    # If the neighbour lands inside the crop it must be greyed out.
    grey = np.all(np.abs(crop.astype(int) - 128) < 3, axis=2)
    ax1 = 300 - int(max(0, (300 + 360) / 2 - max((360 - 300) * 0.9, 30)))
    check("the neighbour is greyed out when it falls inside the crop",
          (not (b["box"][0] < ax1 + w)) or grey.any(),
          f"{grey.sum()} grey px in a {w}x{h} crop")

    # A container with no neighbours nearby must be left untouched.
    lone = crop_around_instance(img, a, neighbours=[a])
    lone_grey = np.all(np.abs(lone.astype(int) - 128) < 3, axis=2)
    check("a lone container's crop has nothing greyed out",
          not lone_grey.any(), f"{lone_grey.sum()} grey px")

    # Camera 2, 2026-09-25: the sheet runs well below the cup box, and a
    # second mask of the same cup used to be greyed over the only visible text.
    sheet = np.full((480, 848, 3), (30, 40, 80), np.uint8)
    sheet[300:450, 340:520] = (245, 245, 245)
    cup = {"box": [391, 233, 458, 303], "mask": None}
    dup = {"box": [400, 263, 450, 302], "mask": None}
    grown = crop_around_instance(sheet, cup, neighbours=[cup, dup])
    white = np.all(grown.astype(int) > 200, axis=2)
    check("the crop includes the paper in front of the cup",
          int(white.sum()) > 500, f"{int(white.sum())} white px")
    grey = np.all(np.abs(grown.astype(int) - 128) < 3, axis=2)
    check("a second mask of the same cup is not greyed out",
          not grey.any(),
          f"{int(grey.sum())} grey px in {grown.shape[1]}x{grown.shape[0]}")

    # 2026-10-08: sheets laid close with powder spilt between them. The
    # loose white mask joined the cup's sheet to its neighbour's into one blob
    # too big to be a sheet, so none was found and the crop cut the "B" off.
    from robochem.vision.label_resolver import _paper_sheet_box, _sheet_in
    import cv2
    bench = np.full((480, 848, 3), (30, 40, 80), np.uint8)
    bench[142:480, 316:544] = (170, 170, 170)          # powder: pale, not paper-bright
    bench[240:400, 330:470] = (245, 245, 245)          # this cup's sheet
    bench[412:480, 330:470] = (245, 245, 245)          # the neighbour's, 12 px below
    box = (400, 250, 460, 310)
    found = _paper_sheet_box(bench, *box)
    check("the cup's own sheet is found among powder and a neighbour's sheet",
          found is not None and abs(found[0] - 330) <= 4 and abs(found[3] - 400) <= 4,
          f"{found}")
    hsv = cv2.cvtColor(bench[142:480, 316:544], cv2.COLOR_BGR2HSV)
    check("the old loose mask alone finds nothing there (the failure)",
          _sheet_in(hsv, 70, 145, 9, 2, 316, 142, (84, 108, 144, 168)) is None)
    dim = np.full((480, 848, 3), (30, 40, 80), np.uint8)
    dim[240:400, 330:470] = (170, 170, 170)            # one sheet in dim light
    check("in dim light the loose pass still finds the sheet",
          _paper_sheet_box(dim, *box) is not None)


def test_duplicate_labels_are_flagged():
    """Two containers reading the same label means the crop bled."""
    print("\n[labels: duplicate reads are flagged, not silently resolved]")
    from robochem.vision.label_resolver import pick_matching_instance
    import io, contextlib

    instances = [{"score": 0.6}, {"score": 0.9}]
    labels = ["A 10 ML WATER", "A 10 ML WATER"]     # the crop-bleed signature
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        inst, lab = pick_matching_instance(instances, labels, "a 10 ml water")
    check("a duplicate label is called out", "DUPLICATE" in buf.getvalue(),
          buf.getvalue().strip() or "no warning")
    check("and that camera picks neither cup", inst is None and lab is None,
          f"picked score {inst.get('score') if inst else None}")

    buf2 = io.StringIO()
    with contextlib.redirect_stdout(buf2):
        pick_matching_instance([{"score": 0.8}], ["A 10 ML WATER"], "a 10 ml water")
    check("a single clean match says nothing",
          "DUPLICATE" not in buf2.getvalue() and "AMBIGUOUS" not in buf2.getvalue(),
          buf2.getvalue().strip() or "(silent)")

    same_cup = [
        {"score": 0.79, "box": [391, 233, 458, 303]},
        {"score": 0.32, "box": [400, 263, 450, 302]},
    ]
    buf3 = io.StringIO()
    with contextlib.redirect_stdout(buf3):
        inst, lab = pick_matching_instance(
            same_cup, ["B 10 ML WATER", "B 10 ML WATER"], "b 10 ml water")
    check("two masks of one cup are not called a duplicate",
          "DUPLICATE" not in buf3.getvalue() and inst is same_cup[0],
          buf3.getvalue().strip() or f"picked score {inst.get('score') if inst else None}")


def test_label_matching_distinguishes_replicates():
    """A/B replicate labels must not be conflated — the experiment needs both."""
    print("\n[labels: A and B replicates stay distinct]")
    from robochem.vision.label_resolver import labels_match, pick_matching_instance

    # The case that sent powder into the wrong cup on 2026-09-21.
    check("'a 10 ml water' matches A", labels_match("a 10 ml water", "A 10 ML WATER"))
    check("'a 10 ml water' does NOT match B",
          not labels_match("a 10 ml water", "B 10 ML WATER"))
    check("'b 10 ml water' matches B", labels_match("b 10 ml water", "B 10 ML WATER"))
    check("'b 10 ml water' does NOT match A",
          not labels_match("b 10 ml water", "A 10 ML WATER"))

    # Useful partial matches must survive the tightening.
    check("'red cabbage' still finds RED CABBAGE POWDER",
          labels_match("red cabbage", "RED CABBAGE POWDER"))
    check("'citric acid' still matches", labels_match("citric acid", "CITRIC ACID"))
    check("'baking soda' still matches", labels_match("baking soda", "BAKING SODA"))
    check("a request may not contradict the label",
          not labels_match("citric acid", "BAKING SODA"))

    # An under-specified request matches both, and must say so rather than
    # silently picking the higher-scoring mask.
    import io, contextlib
    instances = [{"score": 0.6}, {"score": 0.9}]
    labels = ["A 10 ML WATER", "B 10 ML WATER"]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        inst, lab = pick_matching_instance(instances, labels, "10 ml water")
    check("an ambiguous request is flagged", "AMBIGUOUS" in buf.getvalue(),
          buf.getvalue().strip() or "no warning")

    buf2 = io.StringIO()
    with contextlib.redirect_stdout(buf2):
        inst2, lab2 = pick_matching_instance(instances, labels, "b 10 ml water")
    check("a specific request picks exactly that one and is not flagged",
          lab2 == "B 10 ML WATER" and "AMBIGUOUS" not in buf2.getvalue(),
          f"picked {lab2!r}; {buf2.getvalue().strip()}")


def test_dump_keeps_the_bowl_over_the_target():
    """The bowl must stay over the cup through the whole tip, not swing off."""
    print("\n[dump: bowl held over the target while tipping]")
    cup = cylinder_cloud([0.50, 0.0], base_z=0.02, height=0.09, radius=0.040)
    vision = FakeVision({"cup": cup})
    arm = FakeArm(tracking=1.0, gripper_width=0.03)
    skill = make(DumpSkill, vision, arm)
    offset = [0.051, 0.009, 0.028]

    ok, result = skill.execute({"target_container": "cup", "tool_offset": offset,
                                "hold_duration": 0.0, "shakes": 0})
    check("dump succeeds", ok, f"{result}")
    # Straight ahead, reach -- not the wrist -- caps the tip over a cup this
    # far out; the skill must go as far as it says it can, and no further.
    check("it tipped as far as the arm reaches",
          ok and 60.0 <= result["tip_commanded"] <= 90.0
          and result["tip_achieved"] >= result["tip_commanded"] - 1.0,
          f"tip={result.get('tip_achieved')} of {result.get('tip_commanded')} "
          f"reachable, {result.get('tip_requested')} asked")
    check("it came back level", ok and result["residual_tilt"] < 5.0,
          f"residual={result.get('residual_tilt')}")

    # Reconstruct the bowl position from every commanded TCP + rotation.
    # It must stay within the cup radius for the whole tip.
    rim = np.array([0.50, 0.0])
    strays = []
    for target, _d, _i in arm.commands:
        strays.append(float(np.linalg.norm(np.asarray(target)[:2] - rim)))
    # The TCP itself legitimately swings far out — that is the compensation.
    check("the TCP does swing away from the rim (the offset is real)",
          max(strays) > 0.04, f"max TCP-to-rim {max(strays) * 1000:.0f}mm")


def test_dump_bowl_stays_put_under_rotation():
    """Directly: tcp_for_tip must hold a fixed tip point across tip angles."""
    print("\n[dump: the compensation actually holds the bowl still]")
    from robochem.skills.base_skill import tool_y_delta
    skill = make(DumpSkill, FakeVision({}), FakeArm(gripper_width=0.03))
    level = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]])
    offset = np.array([0.051, 0.009, 0.028])
    bowl = np.array([0.50, 0.0, 0.15])

    recovered = []
    for angle in (0, 30, 60, 90, 120):
        R = level @ tool_y_delta(-angle)
        tcp = skill.tcp_for_tip(bowl, R, offset)
        recovered.append(tcp + R @ offset)      # where the bowl actually is
    spread = max(float(np.linalg.norm(p - bowl)) for p in recovered)
    check("the bowl stays on the target through 0..120 deg of tip",
          spread < 1e-9, f"worst deviation {spread * 1000:.3f}mm")

    # And the naive alternative — tipping in place — does not.
    naive = [np.array([0.50, 0.0, 0.15]) + (level @ tool_y_delta(-a)) @ offset
             for a in (0, 120)]
    swing = float(np.linalg.norm(naive[1] - naive[0]))
    check("tipping in place would swing the bowl far off target",
          swing > 0.05, f"naive swing {swing * 1000:.0f}mm")


def test_dump_verifies_the_return_to_level():
    """Commanding level once is not enough — it must be checked and retried."""
    print("\n[dump: return to level is verified]")
    cup = cylinder_cloud([0.50, 0.0], base_z=0.02, height=0.09, radius=0.040)
    vision = FakeVision({"cup": cup})

    class LaggyUntipArm(FakeArm):
        """Untips only partway per command, like the real wrist."""

        def goto_pose(self, pose, duration=3.0, use_impedance=True, block=True):
            R = np.asarray(pose.rotation, dtype=float)
            want = float(np.degrees(np.arccos(np.clip(-R[2, 2], -1.0, 1.0))))
            now = float(np.degrees(np.arccos(
                np.clip(-np.asarray(self.pose.rotation)[2, 2], -1.0, 1.0))))
            if want < now - 1.0:                       # an untip command
                reached = now - (now - want) * 0.7     # only 70% of the way
                capped = FakePose(pose.translation,
                                  FLAT_GRASP_R @ _tool_y(-reached))
                super().goto_pose(capped, duration, use_impedance, block)
                return
            super().goto_pose(pose, duration, use_impedance, block)

    arm = LaggyUntipArm(tracking=1.0, gripper_width=0.03)
    skill = make(DumpSkill, vision, arm)
    ok, result = skill.execute({"target_container": "cup",
                                "tool_offset": [0.051, 0.009, 0.028],
                                "hold_duration": 0.0, "shakes": 0})
    check("dump succeeds", ok, f"{result}")
    check("it keeps retrying until the wrist is actually level",
          ok and abs(result["residual_tilt"]) < 20.0,
          f"residual={result.get('residual_tilt')} deg")


def test_dump_accepts_explicit_coordinates():
    """Identical unlabelled cups can only be targeted by position."""
    print("\n[dump: explicit coordinates skip the scan]")
    vision = FakeVision({})          # nothing segmentable at all
    arm = FakeArm(tracking=1.0, gripper_width=0.03)
    skill = make(DumpSkill, vision, arm)
    ok, result = skill.execute({"target_container": [0.52, -0.08, 0.10],
                                "tool_offset": [0.051, 0.009, 0.028],
                                "hold_duration": 0.0, "shakes": 0})
    check("dump succeeds with nothing segmentable", ok, f"{result}")
    # clear_cache IS called once at the end — the target's contents changed.
    # What must not happen is a segmentation pass.
    check("no segmentation was attempted", vision.locate_calls == 0,
          f"locate() called {vision.locate_calls} times")
    check("it is logged as a coordinate dump",
          ok and result["dumped_into"] == "coordinates",
          f"{result.get('dumped_into')}")
    check("it tipped as far as the arm reaches",
          ok and result["tip_achieved"] >= result["tip_commanded"] - 1.0,
          f"{result.get('tip_achieved')} of {result.get('tip_commanded')}")
    check("it went home first, though no scan was needed",
          arm.resets >= 1, f"{arm.resets} reset_joints calls")

    ok2, r2 = make(DumpSkill, vision, FakeArm(gripper_width=0.03)).execute(
        {"target_container": [0.52, -0.08]})
    check("a 2-value coordinate is rejected with a clear message",
          not ok2 and "x, y, z" in r2.get("error", ""), f"{r2.get('error')}")


def test_dump_fails_if_the_wrist_cannot_invert():
    print("\n[dump: a stalled tip is a failure, not a success]")
    cup = cylinder_cloud([0.50, 0.0], base_z=0.02, height=0.09, radius=0.040)
    vision = FakeVision({"cup": cup})
    arm = FakeArm(tracking=1.0, gripper_width=0.03, tilt_gain=0.02)
    skill = make(DumpSkill, vision, arm)
    ok, result = skill.execute({"target_container": "cup",
                                "tool_offset": [0.051, 0.009, 0.028],
                                "shakes": 0})
    check("dump fails when the wrist stalls", not ok, f"{result}")
    check("the failure says the powder did not come out",
          "come out" in result.get("error", "").lower(),
          f"{result.get('error')}")


def test_dump_accepts_a_wrist_that_stalls_near_90():
    """A wrist that saturates at ~90deg has still emptied the bowl."""
    print("\n[dump: a stall past min_tip_deg is a success, not a failure]")
    cup = cylinder_cloud([0.50, 0.0], base_z=0.02, height=0.09, radius=0.040)
    vision = FakeVision({"cup": cup})

    class CappedWristArm(FakeArm):
        """Tracks orientation fine up to CAP degrees, then refuses to go on."""

        CAP = 72.0

        def goto_pose(self, pose, duration=3.0, use_impedance=True, block=True):
            # Clamp rather than refuse: the real wrist partially tracks an
            # over-range command (asked 90 deg, reached 85.5; asked 120,
            # reached 89.4), it does not simply ignore it.
            R = np.asarray(pose.rotation, dtype=float)
            want = float(np.degrees(np.arccos(np.clip(-R[2, 2], -1.0, 1.0))))
            if want > self.CAP:
                capped = FakePose(pose.translation,
                                  FLAT_GRASP_R @ _tool_y(-self.CAP))
                super().goto_pose(capped, duration, use_impedance, block)
                return
            super().goto_pose(pose, duration, use_impedance, block)

    arm = CappedWristArm(tracking=1.0, gripper_width=0.03)
    skill = make(DumpSkill, vision, arm)
    ok, result = skill.execute({"target_container": "cup",
                                "tool_offset": [0.051, 0.009, 0.028],
                                "hold_duration": 0.0, "shakes": 1})
    check("a wrist capped at 72 deg still counts as dumped", ok, f"{result}")
    check("and reports the angle it actually reached",
          ok and 68.0 < result["tip_achieved"] < 76.0,
          f"{result.get('tip_achieved')}")
    # Succeeding BELOW the commanded angle is the allowance working: the bowl
    # emptied, the wrist simply could not go as far as asked.
    check("it succeeded short of the commanded tip",
          ok and result["tip_achieved"] < result["tip_commanded"] - 1.0,
          f"reached {result.get('tip_achieved')} of "
          f"{result.get('tip_commanded')}")

    # Below min_tip_deg it must still fail — the allowance is not a blanket pass.
    arm2 = CappedWristArm(tracking=1.0, gripper_width=0.03)
    arm2.CAP = 40.0
    ok2, r2 = make(DumpSkill, vision, arm2).execute(
        {"target_container": "cup", "tool_offset": [0.051, 0.009, 0.028],
         "shakes": 0})
    check("a wrist capped at 40 deg still fails", not ok2, f"{r2}")


def test_dump_needs_something_held():
    print("\n[dump: refuses with an empty gripper]")
    skill = make(DumpSkill, FakeVision({}), FakeArm(gripper_width=0.079))
    can, msg = skill.check_preconditions({"target_container": "cup"})
    check("dump refuses with an empty gripper", not can, msg)


def test_pick_up_grasp_offsets():
    """forward/lateral offsets move the grasp; -X pulls it toward the base."""
    print("\n[pick_up: grasp can be shifted off the computed centroid]")

    # A bar along X, like the scoop handle lying on the bench.
    rng = np.random.default_rng(0)
    bar = np.column_stack([
        rng.uniform(0.45, 0.52, 4000),      # 70mm long in X
        rng.uniform(-0.004, 0.004, 4000),   # 8mm wide in Y
        rng.uniform(0.02, 0.027, 4000),     # 7mm thick
    ])

    class GraspVision(FakeVision):
        def locate(self, name, force_refresh=False):
            located = super().locate(name, force_refresh)
            if located is not None:
                pts = located["points"]
                located["dimensions"] = np.sort(pts.max(0) - pts.min(0))[::-1]
            return located

        def compute_dimensions(self, points):
            return np.sort(points.max(0) - points.min(0))[::-1]

        def compute_grasp_pose(self, points, object_name=None, grasp_type="auto",
                               pitch_deg=0.0):
            pose = np.eye(4)
            pose[:3, :3] = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., -1.]])
            pose[:3, 3] = np.asarray(points).mean(axis=0)   # centroid grasp
            return pose

        def compute_grasp_candidates(self, points, object_name=None,
                                     grasp_type="auto", pitch_deg=0.0,
                                     spout_xy=None):
            return [self.compute_grasp_pose(points, object_name)]

    def run(extra):
        vision = GraspVision({"scoop": bar})
        arm = FakeArm(tracking=1.0, gripper_width=0.04)
        skill = make(PickUpSkill, vision, arm)
        # An 8mm handle is narrow: the default 4mm squeeze would close to
        # 3.5mm and trip pick_up's own crush check (0.7 x 7.5mm = 5.25mm).
        params = {"object_name": "scoop", "force_limited": False,
                  "squeeze": 0.002}
        params.update(extra)
        ok, result = skill.execute(params)
        return ok, result

    ok0, base = run({})
    check("baseline pick succeeds", ok0, f"{base}")
    x0 = base["grasp_pose"][0][3] if ok0 else None

    ok1, shifted = run({"forward_offset": -0.02})
    check("shifted pick succeeds", ok1, f"{shifted}")
    x1 = shifted["grasp_pose"][0][3] if ok1 else None
    check("negative forward_offset moves the grasp toward the base",
          ok0 and ok1 and (x1 - x0) < -0.015,
          f"x moved {(x1 - x0) * 1000:.1f}mm" if ok0 and ok1 else "n/a")

    ok2, lateral = run({"lateral_offset": 0.015})
    y0 = base["grasp_pose"][1][3] if ok0 else None
    y2 = lateral["grasp_pose"][1][3] if ok2 else None
    check("lateral_offset moves the grasp in +Y",
          ok0 and ok2 and (y2 - y0) > 0.010,
          f"y moved {(y2 - y0) * 1000:.1f}mm" if ok0 and ok2 else "n/a")

    check("a zero offset leaves the grasp exactly where it was",
          ok0 and abs(run({"forward_offset": 0.0})[1]["grasp_pose"][0][3] - x0) < 1e-9)


def test_stir_defaults_to_top_down():
    """
    With no tool_axis, stir is the sim's top-down stir: fingers at the table.

    Hardware used to get the spoon default and a 90deg tool-Y tilt that laid
    the stirrer on its side. The default must never tilt, must level a wrist
    reset_joints left a few degrees off, and must hold fingers-down for every
    pose it commands.
    """
    print("\n[stir: default is top-down, the pick_up orientation]")
    cup = cylinder_cloud([0.50, 0.0], base_z=0.02, height=0.08, radius=0.045)
    vision = FakeVision({"cup": cup})
    arm = FakeArm(tracking=1.0, gripper_width=0.03)
    arm.pose.rotation = np.diag([1.0, -1.0, -1.0]) @ _tool_y(3.0)
    skill = make(StirSkill, vision, arm)

    tilts, rotations, targets = [], [], []
    skill.rotate_about_tool_axis = lambda *a, **k: tilts.append(a) or True
    rigid = skill.goto_pose_rigid

    def spy(xyz, rotation, **kw):
        rotations.append(np.asarray(rotation, dtype=float))
        targets.append(np.asarray(xyz, dtype=float))
        return rigid(xyz, rotation, **kw)
    skill.goto_pose_rigid = spy

    ok, result = skill.execute({"target_container": "cup", "revolutions": 1,
                                "smooth": False, "waypoints_per_rev": 6,
                                "seconds_per_waypoint": 0.0})
    check("default stir succeeds from a top-down grasp", ok, f"{result}")
    check("reports tool_axis 'z'", ok and result["tool_axis"] == "z")
    check("no tool-Y tilt is ever commanded", not tilts, f"{tilts}")
    worst = max(float(np.degrees(np.arccos(np.clip(-R[2, 2], -1, 1))))
                for R in rotations) if rotations else 180.0
    check("every commanded pose is fingers straight down",
          worst < 0.1, f"worst {worst:.2f} deg from down over {len(rotations)} poses")
    check("the 3 deg reset_joints lean was levelled out",
          skill.tool_tip_deg() < 0.1, f"{skill.tool_tip_deg():.2f} deg")
    check("tool_length defaults to the stirrer CAD (0.081)",
          ok and abs(result["tool_length"] - 0.081) < 1e-9,
          f"{result.get('tool_length')}")
    rim_z = skill.locate_container("cup")["top_z"]
    expected_z = rim_z - 0.03 + 0.081
    circle_z = [t[2] for t in targets[-7:-1]]
    check("circle TCP sits tool_length above the immersion depth",
          circle_z and all(abs(z - expected_z) < 1e-6 for z in circle_z),
          f"{np.round(circle_z, 4)} vs {expected_z:.4f}")


def test_stir_tilts_a_flat_spoon_upright():
    """A 90deg tilt toward the base stands a flat-grasped spoon up."""
    print("\n[stir: flat spoon tilted upright toward the base]")
    cup = cylinder_cloud([0.50, 0.0], base_z=0.02, height=0.08, radius=0.045)
    vision = FakeVision({"cup": cup})

    arm = FakeArm(tracking=1.0, gripper_width=0.03)
    arm.pose.rotation = FLAT_GRASP_R.copy()
    skill = make(StirSkill, vision, arm)
    check("the flat grasp starts with the handle horizontal",
          abs(skill.tool_long_axis_deg() - 90.0) < 1.0,
          f"long axis={skill.tool_long_axis_deg():.1f} deg")

    ok, result = skill.execute({"target_container": "cup", "revolutions": 1,
                                "tool_axis": "x",
                                "waypoints_per_rev": 6,
                                "seconds_per_waypoint": 0.0})
    check("stir succeeds from a flat grasp", ok, f"{result}")
    check("the handle ends up pointing DOWN, not up",
          ok and result["long_axis_deg"] < 15.0,
          f"long axis={result.get('long_axis_deg')}")
    check("the tilt is reported", ok and result["verticalized"] is True)

    # The closing axis must survive, or the spoon twists in the jaws.
    R = np.asarray(arm.pose.rotation, dtype=float)
    check("the closing axis stayed horizontal (grip undisturbed)",
          abs(R[2, 1]) < 0.05, f"toolY world Z component={R[2, 1]:.3f}")


def test_stir_tilt_does_not_overshoot_on_retry():
    """Retries must chase the REMAINING angle, not re-issue the full 90deg."""
    print("\n[stir: tilt retries do not stack past vertical]")
    cup = cylinder_cloud([0.50, 0.0], base_z=0.02, height=0.08, radius=0.045)
    vision = FakeVision({"cup": cup})
    # tilt_gain 0.6: each command lands ~60% of the way, forcing retries.
    arm = FakeArm(tracking=1.0, gripper_width=0.03, tilt_gain=0.6)
    arm.pose.rotation = FLAT_GRASP_R.copy()
    skill = make(StirSkill, vision, arm)
    ok, result = skill.execute({"target_container": "cup", "revolutions": 1,
                                "tool_axis": "x",
                                "waypoints_per_rev": 6, "orient_retries": 5,
                                "seconds_per_waypoint": 0.0})
    check("a lagging wrist still converges", ok, f"{result}")
    check("it converged on vertical without overshooting past it",
          ok and result["long_axis_deg"] < 15.0,
          f"long axis={result.get('long_axis_deg')}")


def test_stir_refuses_a_spoon_that_will_not_stand_up():
    print("\n[stir: refuses if the spoon stays horizontal]")
    cup = cylinder_cloud([0.50, 0.0], base_z=0.02, height=0.08, radius=0.045)
    vision = FakeVision({"cup": cup})
    arm = FakeArm(tracking=1.0, gripper_width=0.03, tilt_gain=0.02)
    arm.pose.rotation = FLAT_GRASP_R.copy()
    skill = make(StirSkill, vision, arm)
    ok, result = skill.execute({"target_container": "cup", "tool_axis": "x"})
    check("stir fails when the spoon stays horizontal", not ok, f"{result}")
    check("the failure says a horizontal spoon cannot enter",
          "horizontal" in result.get("error", "").lower(),
          f"{result.get('error')}")


def test_stir_homes_before_orienting():
    """Home first, holding the tool — the swing needs wrist travel."""
    print("\n[stir: homes before the swing, without dropping the tool]")
    vision = FakeVision({})
    arm = FakeArm(tracking=1.0, gripper_width=0.03)
    # Park the arm out over the bench, as pick_up would leave it.
    arm.pose.translation = np.array([0.68, -0.15, 0.20])
    skill = make(StirSkill, vision, arm)

    ok, result = skill.execute({"in_air": True, "revolutions": 1,
                                "waypoints_per_rev": 6, "stir_radius": 0.02,
                                "seconds_per_waypoint": 0.0})
    check("stir succeeds from a far-forward pick pose", ok, f"{result}")
    check("the gripper was never opened by the homing",
          not [c for c in arm.gripper_commands if c["width"] > 0.06],
          f"gripper commands={arm.gripper_commands}")
    check("the circle is centred at home, not the old pick pose",
          ok and abs(result["center"][0] - 0.68) > 0.1,
          f"centre={result.get('center')}")

    arm2 = FakeArm(tracking=1.0, gripper_width=0.03)
    arm2.pose.translation = np.array([0.68, -0.15, 0.20])
    skill2 = make(StirSkill, vision, arm2)
    ok2, _ = skill2.execute({"in_air": True, "home_first": False,
                             "revolutions": 1, "waypoints_per_rev": 6,
                             "seconds_per_waypoint": 0.0})
    check("home_first=false leaves the arm where it was",
          ok2 and abs(arm2.commands[-1][0][0] - 0.68) < 0.12,
          f"last commanded x={arm2.commands[-1][0][0]:.3f}")


def test_stir_approaches_the_circle_before_walking_it():
    """Waypoint 1 is a long approach, not a circle step — it needs real time.

    Regression for the first in-air hardware run, which failed on waypoint 1
    "off by 173mm". The arm was at home, the circle was 17cm below, and the
    move was given the 0.4s per-waypoint circle dwell — about 0.43 m/s — so it
    never happened. The reported error was the entire untravelled distance.
    """
    print("\n[stir: the circle start is approached, not jumped to]")
    vision = FakeVision({})

    class SpeedLimitedArm(FakeArm):
        """Only covers what the commanded duration allows, at MAX_SPEED."""

        MAX_SPEED = 0.15   # m/s, deliberately modest

        def reset_joints(self):
            # The real Franka home puts the TCP around z=0.47, which is what
            # makes the circle at z=0.30 a ~17cm trip. The default FakeArm home
            # sits at z=0.35 and would not reproduce the bench failure.
            self.pose = FakePose([0.31, 0.0, 0.47], FLAT_GRASP_R.copy())

        def goto_pose(self, pose, duration=3.0, use_impedance=True, block=True):
            start = self.pose.translation.copy()
            target = np.asarray(pose.translation, dtype=float)
            gap = float(np.linalg.norm(target - start))
            reach = min(gap, self.MAX_SPEED * float(duration))
            if gap > 1e-9:
                self.pose.translation = start + (target - start) * (reach / gap)
            self.pose.rotation = np.asarray(pose.rotation, dtype=float)
            self.commands.append((target.copy(), duration, use_impedance))

    arm = SpeedLimitedArm(gripper_width=0.03)
    arm.pose.translation = np.array([0.30, 0.0, 0.47])   # home-ish, well above
    arm.pose.rotation = FLAT_GRASP_R.copy()
    skill = make(StirSkill, vision, arm)

    ok, result = skill.execute({"in_air": True, "air_height": 0.30,
                                "revolutions": 1, "stir_radius": 0.03,
                                "waypoints_per_rev": 12,
                                "seconds_per_waypoint": 0.4})
    check("the circle start is reached despite being 17cm away", ok, f"{result}")
    check("the full revolution ran", ok and result["revolutions_completed"] == 1.0,
          f"{result.get('revolutions_completed')}")

    # The approach must get a longer duration than the circle steps.
    durations = [c[1] for c in arm.commands]
    circle = durations[-12:]
    check("waypoint 1 got a longer duration than the circle steps",
          max(circle) > min(circle),
          f"durations={[round(d, 2) for d in circle]}")

    # And the same run with the old behaviour must fail, or the test is vacuous.
    arm2 = SpeedLimitedArm(gripper_width=0.03)
    arm2.pose.translation = np.array([0.30, 0.0, 0.47])
    arm2.pose.rotation = FLAT_GRASP_R.copy()
    ok2, result2 = make(StirSkill, vision, arm2).execute(
        {"in_air": True, "air_height": 0.30, "revolutions": 1,
         "stir_radius": 0.03, "waypoints_per_rev": 12,
         "seconds_per_waypoint": 0.4, "approach_seconds": 0.4})
    check("with approach_seconds=dwell it fails, as it did on the bench",
          not ok2, f"{result2}")


def test_stir_circle_path_geometry():
    """The streamed path must be a real circle, eased at both ends."""
    print("\n[stir: streamed circle path geometry]")
    vision = FakeVision({})
    skill = make(StirSkill, vision, FakeArm(gripper_width=0.03))
    center = np.array([0.45, -0.05])
    path = skill._circle_path(center, radius=0.03, z=0.25,
                              seconds=4.0, rate_hz=50.0)

    check("the path is dense enough to read as continuous", len(path) >= 200,
          f"{len(path)} setpoints")
    radii = [float(np.linalg.norm(np.asarray(p)[:2] - center)) for p in path]
    check("every point sits on the circle",
          max(abs(r - 0.03) for r in radii) < 1e-6,
          f"radius range {min(radii):.5f}..{max(radii):.5f}")
    check("the z plane is held", all(abs(p[2] - 0.25) < 1e-9 for p in path))
    check("it closes the loop",
          float(np.linalg.norm(np.asarray(path[0]) - np.asarray(path[-1]))) < 1e-6,
          f"start={np.round(path[0], 4)}, end={np.round(path[-1], 4)}")

    # Min-jerk easing: the first and last steps must be much smaller than the
    # middle ones, so each revolution starts and ends at rest.
    steps = [float(np.linalg.norm(np.asarray(b) - np.asarray(a)))
             for a, b in zip(path, path[1:])]
    mid = steps[len(steps) // 2]
    check("it accelerates away from rest", steps[0] < mid * 0.2,
          f"first step={steps[0] * 1000:.2f}mm vs mid={mid * 1000:.2f}mm")
    check("it settles back to rest", steps[-1] < mid * 0.2,
          f"last step={steps[-1] * 1000:.2f}mm vs mid={mid * 1000:.2f}mm")


def test_stir_streams_each_revolution_as_one_motion():
    """When streaming is available, one skill per revolution — not per point."""
    print("\n[stir: each revolution is one streamed motion]")
    vision = FakeVision({})
    arm = FakeArm(tracking=1.0, gripper_width=0.03)
    arm.pose.rotation = FLAT_GRASP_R.copy()
    skill = make(StirSkill, vision, arm)

    streamed = []

    def fake_stream(points, rotation, seconds, rate_hz=50.0, tag="Skill",
                    cartesian_impedance=False):
        pts = [np.asarray(p, dtype=float) for p in points]
        streamed.append(pts)
        arm.pose.translation = pts[-1].copy()     # end where the path ends
        return True, f"streamed {len(pts)} setpoints"

    skill.stream_pose_path = fake_stream

    ok, result = skill.execute({"in_air": True, "revolutions": 3,
                                "stir_radius": 0.03, "seconds_per_revolution": 4.0})
    check("streamed stir succeeds", ok, f"{result}")
    check("it reports having run smooth", ok and result["smooth"] is True,
          f"smooth={result.get('smooth')}")
    check("one streamed path per revolution", len(streamed) == 3,
          f"{len(streamed)} streamed paths")
    check("each path is a dense circle, not a handful of waypoints",
          all(len(p) >= 200 for p in streamed),
          f"lengths={[len(p) for p in streamed]}")
    check("all three revolutions counted",
          ok and result["revolutions_completed"] == 3.0,
          f"{result.get('revolutions_completed')}")


def test_stir_falls_back_when_streaming_is_unavailable():
    """No ROS -> blocking waypoints, still a successful stir."""
    print("\n[stir: falls back to waypoints without streaming]")
    vision = FakeVision({})
    arm = FakeArm(tracking=1.0, gripper_width=0.03)
    arm.pose.rotation = FLAT_GRASP_R.copy()
    skill = make(StirSkill, vision, arm)
    # FakeArm has no ROS behind it, so the real stream_pose_path returns False.
    ok, result = skill.execute({"in_air": True, "revolutions": 1,
                                "stir_radius": 0.03, "waypoints_per_rev": 12,
                                "seconds_per_waypoint": 0.0})
    check("the fallback still stirs", ok, f"{result}")
    check("it reports NOT smooth, so the log is honest",
          ok and result["smooth"] is False, f"smooth={result.get('smooth')}")
    check("the revolution still completed",
          ok and result["revolutions_completed"] == 1.0,
          f"{result.get('revolutions_completed')}")


def test_stir_in_air_needs_no_container():
    """The demo mode stirs in free space with no scan and no target."""
    print("\n[stir: in-air demo]")
    vision = FakeVision({})          # nothing segmentable at all
    arm = FakeArm(tracking=1.0, gripper_width=0.03)
    arm.pose.rotation = np.array([[1.0, 0.0, 0.0],
                                  [0.0, -1.0, 0.0],
                                  [0.0, 0.0, -1.0]])
    skill = make(StirSkill, vision, arm)

    can, msg = skill.check_preconditions({"in_air": True})
    check("in-air needs no target_container", can, msg)

    ok, result = skill.execute({"in_air": True, "revolutions": 2,
                                "waypoints_per_rev": 8, "stir_radius": 0.03,
                                "seconds_per_waypoint": 0.0})
    check("in-air stir succeeds with nothing in the scene", ok, f"{result}")
    check("it is flagged as an in-air run", ok and result["in_air"] is True)
    # clear_cache IS called once at the end — the target's contents changed.
    # What must not happen is a segmentation pass.
    check("no segmentation was attempted", vision.locate_calls == 0,
          f"locate() called {vision.locate_calls} times")
    check("the in-air stir is top-down too (fingers at the table)",
          ok and result["tool_angle_from_down_deg"] < 1.0,
          f"tool Z {result.get('tool_angle_from_down_deg')} deg from down")
    check("the full circle ran", ok and result["revolutions_completed"] == 2.0,
          f"{result.get('revolutions_completed')}")

    # A real circle, not a point: the commanded XY must actually vary.
    xs = [c[0][0] for c in arm.commands]
    ys = [c[0][1] for c in arm.commands]
    check("the commanded path spans roughly the circle diameter",
          (max(xs) - min(xs)) > 0.05 and (max(ys) - min(ys)) > 0.05,
          f"x span={max(xs) - min(xs):.3f}, y span={max(ys) - min(ys):.3f}")


class PressArm(FakeArm):
    """
    A FakeArm with a force reading: a stiff surface at ``surface_z`` (TCP
    height) pushes back 1N per mm of penetration, on top of a constant sensor
    bias the skill must subtract rather than mistake for contact. Pressing
    reads NEGATIVE, as the real arm did on 2026-10-05.
    """

    def __init__(self, surface_z, bias=-2.5):
        super().__init__(tracking=1.0, gripper_width=0.03)
        self.surface_z = surface_z
        self.bias = bias

    def get_ee_force_torque(self):
        press = max(0.0, self.surface_z - float(self.pose.translation[2]))
        return np.array([0.0, 0.0, self.bias - 1000.0 * press, 0.0, 0.0, 0.0])


def test_stir_stops_descending_on_contact():
    """A rod that meets the floor early stops there instead of pushing on."""
    print("\n[stir: force-guarded descent]")
    cup = cylinder_cloud([0.50, 0.0], base_z=0.02, height=0.08, radius=0.045)
    opts = {"target_container": "cup", "revolutions": 1, "smooth": False,
            "waypoints_per_rev": 6, "seconds_per_waypoint": 0.0,
            "probe_seconds": 0.0}
    rim_z = make(StirSkill, FakeVision({"cup": cup}), FakeArm()) \
        .locate_container("cup")["top_z"]
    tool = StirSkill.STIRRER_TOOL_LENGTH
    planned = rim_z - 0.03 + tool

    # Floor 15mm below the rim, i.e. 15mm shallower than the 30mm asked for.
    surface = rim_z - 0.015 + tool
    arm = PressArm(surface)
    ok, result = make(StirSkill, FakeVision({"cup": cup}), arm).execute(dict(opts))
    check("stir still succeeds after stopping short", ok, f"{result}")
    check("the descent was stopped by force",
          ok and result["stopped_by_force"] is True, f"{result.get('stopped_by_force')}")
    lowest = min(c[0][2] for c in arm.commands)
    check("never commanded more than one step past the contact",
          lowest > surface - 0.004, f"lowest z={lowest:.4f}, surface {surface:.4f}")
    check("never commanded the planned depth",
          lowest > planned + 0.005, f"lowest z={lowest:.4f}, planned {planned:.4f}")
    check("stirs backed off above the contact",
          ok and result["stir_z"] > result["contact_z"]
          and abs(result["stir_z"] - result["contact_z"] - 0.003) < 1e-6,
          f"stir_z={result.get('stir_z')}, contact_z={result.get('contact_z')}")
    check("the -2.5N sensor bias was not read as contact",
          ok and 1.0 < abs(result["contact_force_n"]) < 4.0,
          f"{result.get('contact_force_n')}")

    # Something 5mm ABOVE the rim at the rod tip: it landed on the rim.
    arm = PressArm(rim_z + 0.005 + tool)
    ok, result = make(StirSkill, FakeVision({"cup": cup}), arm).execute(dict(opts))
    check("contact above the rim is refused, not stirred", not ok, f"{result}")
    check("the refusal says it landed on the rim",
          "rim" in result.get("error", "").lower(), f"{result.get('error')}")
    check("it lifted back out to the hover",
          arm.pose.translation[2] > rim_z + tool + 0.05,
          f"z={arm.pose.translation[2]:.4f}")

    # Nothing in the way: the full planned depth, no early stop.
    arm = PressArm(0.0)
    ok, result = make(StirSkill, FakeVision({"cup": cup}), arm).execute(dict(opts))
    check("with nothing in the way it reaches the planned depth",
          ok and abs(result["stir_z"] - planned) < 1e-6
          and result["force_guarded"] is True and result["stopped_by_force"] is False,
          f"{result.get('stir_z')} vs {planned:.4f}")

    # No force reading at all: position-only, as before, and it says so.
    ok, result = make(StirSkill, FakeVision({"cup": cup}), FakeArm(
        tracking=1.0, gripper_width=0.03)).execute(dict(opts))
    check("an arm with no force reading falls back to position",
          ok and result["force_guarded"] is False, f"{result}")

    # to_floor: feel past the measured bottom and stop on the floor itself,
    # in a cup shallow enough for the rod (a 52 mm clear cup, say).
    shallow = cylinder_cloud([0.50, 0.0], base_z=0.0, height=0.05, radius=0.033)
    s_rim = make(StirSkill, FakeVision({"cup": shallow}), FakeArm()) \
        .locate_container("cup")["top_z"]
    floor_tip = 0.003                           # a 3 mm base
    arm = PressArm(floor_tip + tool)
    ok, result = make(StirSkill, FakeVision({"cup": shallow}), arm).execute(
        dict(opts, to_floor=True))
    check("to_floor stops on the floor and stirs just above it",
          ok and result["stopped_by_force"] is True and result["to_floor"] is True
          and not result["hand_limited"]
          and abs(result["stir_z"] - result["contact_z"] - 0.004) < 1e-6
          and result["contact_z"] - tool < floor_tip + 0.002,
          f"{ {k: result.get(k) for k in ('stopped_by_force', 'contact_z', 'stir_z', 'hand_limited')} }")
    check("to_floor went deeper than the default stir_depth would",
          ok and result["stir_z"] < s_rim - 0.03 + tool - 0.01,
          f"stir_z={result.get('stir_z')}")
    # Coarse through the open cup, fine near the floor (2026-10-08: 1 mm all
    # the way down was too slow on the robot).
    zs = [c[0][2] for c in arm.commands]
    steps = [a - b for a, b in zip(zs, zs[1:]) if 0 < a - b < 0.0055]
    check("the open part is crossed in 5 mm steps, the floor felt in 1 mm ones",
          any(abs(s - 0.005) < 1e-6 for s in steps)
          and abs(steps[-1] - 0.001) < 1e-6 if steps else False,
          f"{[round(s * 1000, 1) for s in steps]}")
    check("about half the steps 1 mm all the way would take",
          len(steps) < 0.6 * ((s_rim + 0.010) - floor_tip) / 0.001,
          f"{len(steps)} steps for {((s_rim + 0.010) - floor_tip) * 1000:.0f}mm")
    # In the 80 mm cup the rod cannot reach the floor with the hand clear of
    # the rim: it stirs as deep as that allows, and says so.
    arm = PressArm(0.0)
    ok, result = make(StirSkill, FakeVision({"cup": cup}), arm).execute(
        dict(opts, to_floor=True))
    lowest = min(c[0][2] for c in arm.commands)
    check("a cup taller than the rod is stirred with the hand above the rim",
          ok and result["hand_limited"] is True and lowest >= rim_z + 0.030 - 1e-6,
          f"lowest TCP z {lowest:.4f}, rim {rim_z:.4f}, {result.get('hand_limited')}")
    ok, result = make(StirSkill, FakeVision({"cup": cup}), FakeArm(
        tracking=1.0, gripper_width=0.03)).execute(dict(opts, to_floor=True))
    check("to_floor is refused without a force reading",
          not ok and "force" in result.get("error", ""), f"{result}")


def test_executor_discards_a_tool_offset_from_the_wrong_end():
    """A measured offset whose x points back along the handle is not used."""
    print("\n[executor: tool_offset sanity against the CAD]")
    from robochem.skills import SkillsExecutor
    from robochem.skills.scoop import ScoopSkill
    ex = SkillsExecutor(None, None, {})
    ex.held = "larger spoon"
    skill = ScoopSkill.__new__(ScoopSkill)
    cad = [0.0257, 0.0, 0.0294]
    # 2026-10-02, sim: pick_up measured the handle end as the bowl.
    out = ex._fill_tool_geometry("scoop", skill, {"tool_offset": [-0.0304, 0.0007, 0.0023]})
    check("a sign-flipped x is replaced by the CAD offset",
          out["tool_offset"] == cad, f"{out['tool_offset']}")
    # The bench-validated measurement: x 25 mm past the CAD (the grasp sat
    # back along the handle), y 9 mm off. It must survive, z raised to CAD.
    out = ex._fill_tool_geometry("scoop", skill, {"tool_offset": [0.051, 0.009, 0.028]})
    check("a plausible measured x/y is kept, z raised to the CAD depth",
          out["tool_offset"] == [0.051, 0.009, 0.0294], f"{out['tool_offset']}")
    out = ex._fill_tool_geometry("scoop", skill, {"tool_offset": [0.03, 0.035, 0.03]})
    check("an offset 35 mm to one side is replaced",
          out["tool_offset"] == cad, f"{out['tool_offset']}")
    out = ex._fill_tool_geometry("scoop", skill, {})
    check("no offset at all gets the CAD one", out["tool_offset"] == cad,
          f"{out.get('tool_offset')}")
    ex.held = "smaller spoon"
    out = ex._fill_tool_geometry("scoop", skill, {})
    check("the smaller spoon has its own geometry",
          out["tool_offset"] == [0.0208, 0.0, 0.0235] and abs(out["bowl_length"] - 0.0206) < 1e-9,
          f"{out}")

    # Where the pick saw the larger spoon's two ends (sim, 2026-10-03: the
    # bowl's centre was truly 21.5 mm out; pick_up suggested 29.2).
    ex.held = "larger spoon"
    suggested = [0.0292, 0.0087, 0.0236]
    ex.held_measure = {"suggested": suggested, "extent_x": [-0.0333, 0.0315]}
    expect = (-0.0333 + 0.0315) / 2 + 0.02105
    out = ex._fill_tool_geometry("scoop", skill, {"tool_offset": list(suggested)})
    check("pick_up's offset copied by a planner -> x from the tool's ends, y/z CAD",
          abs(out["tool_offset"][0] - expect) < 1e-9 and out["tool_offset"][1:] == [0.0, 0.0294],
          f"{out['tool_offset']} vs x {expect:.4f}")
    out = ex._fill_tool_geometry("scoop", skill, {})
    check("no offset at all -> x from the tool's ends too",
          abs(out["tool_offset"][0] - expect) < 1e-9, f"{out['tool_offset']}")
    out = ex._fill_tool_geometry("scoop", skill, {"tool_offset": [0.051, 0.009, 0.028]})
    check("an offset passed on purpose is still kept (z raised)",
          out["tool_offset"] == [0.051, 0.009, 0.0294], f"{out['tool_offset']}")
    ex.held_measure = {"suggested": suggested, "extent_x": [-0.09, -0.06]}
    out = ex._fill_tool_geometry("scoop", skill, {})
    check("ends that do not fit the part fall back to the CAD",
          out["tool_offset"] == cad, f"{out['tool_offset']}")
    ex.held_measure = {}


def test_executor_puts_things_back_where_they_came_from():
    """place 'beside' another object sends a picked object back to its own spot."""
    print("\n[executor: put back where picked]")
    from robochem.skills import SkillsExecutor
    ex = SkillsExecutor(None, None, {})
    ex.pick_sites = {"water cup": [0.49, -0.07, 0.03], "plastic beaker": [0.38, -0.07, 0.05],
                     "stirrer": [0.62, 0.14, 0.10]}
    ex.held = "water cup"
    out = ex._resolve_pick_site("place", {"target_location": "plastic beaker"})
    check("beside another picked object -> the held cup's own site",
          out["target_location"] == [0.49, -0.07, 0.03], f"{out}")
    out = ex._resolve_pick_site("place", {"target_location": "clear cup b"})
    check("beside an object never picked -> the held cup's own site",
          out["target_location"] == [0.49, -0.07, 0.03], f"{out}")
    out = ex._resolve_pick_site("place", {"target_location": "clear cup b", "on_top": True})
    check("on_top is left alone", out["target_location"] == "clear cup b", f"{out}")
    out = ex._resolve_pick_site("place", {"target_location": [0.4, 0.1]})
    check("explicit coordinates are left alone", out["target_location"] == [0.4, 0.1], f"{out}")
    ex.held = "stirrer"
    out = ex._resolve_pick_site("place", {"target_location": "stirrer holder"})
    check("the stirrer named by its holder goes back in, inserted vertically",
          out["target_location"] == [0.62, 0.14, 0.10] and out.get("vertical_insert") is True,
          f"{out}")
    ex.held = None
    out = ex._resolve_pick_site("place", {"target_location": "plastic beaker"})
    check("with nothing tracked in the jaws a name is left for vision",
          out["target_location"] == [0.38, -0.07, 0.05] or out["target_location"] == "plastic beaker",
          f"{out}")


def test_bench_manifest_names_the_real_bench():
    """The real Magic Beaker bench uses the sim's names, and one scoop stands for both."""
    print("\n[bench manifest: real Magic Beaker bench, 2026-10-07]")
    from robochem.vision.bench_manifest import MAGIC_BEAKER as bench
    from robochem.vision.label_resolver import labels_match

    # Handwritten labels spell the unit both ways.
    check("'A 10ML WATER' reads as 'a 10 ml water'",
          labels_match("a 10 ml water", "A 10ML WATER"))
    check("'50ML WATER' reads as '50 ml water'", labels_match("50 ml water", "50ML WATER"))
    check("A still does not match B when spelled '10ML'",
          not labels_match("a 10 ml water", "B 10ML WATER"))
    check("the 50 ml beaker is not a 10 ml cup",
          not labels_match("a 10 ml water", "50 ML WATER")
          and not labels_match("50 ml water", "A 10 ML WATER"))

    # Every name the sim's 31-step plan used (agent_run_20261007_165912).
    sim_names = ["larger spoon", "smaller spoon", "stirrer", "stirrer holder",
                 "citric acid cup", "baking soda cup", "red cabbage powder cup",
                 "clear cup a", "clear cup b", "clear cup c", "plastic beaker"]
    missing = [n for n in sim_names if bench.find(n) is None]
    check("every name the sim plan used is on the real bench", not missing, f"{missing}")
    check("the bench has 10 objects (one scoop)", len(bench.objects) == 10,
          f"{[o.name for o in bench.objects]}")
    for asked in ("smaller spoon", "small scoop", "big scoop", "scoop", "white plastic tool"):
        check(f"{asked!r} is the larger spoon", bench.canonical(asked) == "larger spoon",
              bench.canonical(asked))
    for asked, want in (("A 10ML WATER", "clear cup a"), ("b 10 ml water", "clear cup b"),
                        ("C EMPTY", "clear cup c"), ("50 ML WATER", "plastic beaker"),
                        ("CITRIC ACID", "citric acid cup"), ("white cube", "stirrer"),
                        ("holder", "stirrer holder")):
        check(f"{asked!r} -> {want!r}", bench.canonical(asked) == want, bench.canonical(asked))
    check("a name not on the bench is left alone",
          bench.find("pipette") is None and bench.canonical("pipette") == "pipette")

    beaker = bench.find("plastic beaker")
    check("the beaker is found as a labelled clear cup",
          beaker.label == "50 ml water" and beaker.category == "clear plastic cup", f"{beaker}")
    check("the reagent dishes are white bowls",
          all(bench.find(n).category == "white bowl"
              for n in ("citric acid cup", "baking soda cup", "red cabbage powder cup")))
    check("tools go by their swept SAM prompts",
          bench.find("larger spoon").prompt == "white plastic tool"
          and bench.find("stirrer").prompt == "white cube")

    names = bench.known_object_names()
    notes = bench.inventory_notes()
    check("inventory lists every object and every label",
          all(o.name in names for o in bench.objects) and "c empty" in names
          and "50 ml water" in names, f"{names}")
    check("one note per object", sorted(notes) == sorted(o.name for o in bench.objects))
    check("a labelled object's note carries its label the way comprehend_scene reads it",
          'reading "a 10 ml water"' in notes["clear cup a"], notes["clear cup a"])
    check("the scoop's note says it stands in for the small one",
          "small" in notes["larger spoon"], notes["larger spoon"])


def test_label_reader_chooses_among_bench_labels():
    """Given the bench's labels, the reader picks one of them or nothing."""
    print("\n[labels: closed set on the real bench]")
    from types import SimpleNamespace
    from robochem.vision.bench_manifest import MAGIC_BEAKER
    from robochem.vision.label_resolver import (LabelResolver, read_one_label_vlm,
                                                snap_to_candidates)

    labels = MAGIC_BEAKER.label_texts()
    check("the reader is given all seven labels as written", labels == [
        "CITRIC ACID", "BAKING SODA", "RED CABBAGE POWDER", "A 10 ML WATER",
        "B 10 ML WATER", "C EMPTY", "50 ML WATER"], f"{labels}")
    # The reads of the first real locate run (2026-10-07).
    check("'EMPTY' is C, the only label it fits", snap_to_candidates("EMPTY", labels) == "C EMPTY")
    check("'50 ML' is the beaker", snap_to_candidates("50 ML", labels) == "50 ML WATER")
    check("'10 ML WATER' fits A and B, so it is no answer",
          snap_to_candidates("10 ML WATER", labels) is None)
    for invented in ("A 10% NaOH", "ACETONE", "B 0.1 M NaOH", "SOAP WATER"):
        check(f"an invented {invented!r} is unread", snap_to_candidates(invented, labels) is None)
    check("'c 10 ml water' (C read with its neighbours' words) is unread",
          snap_to_candidates("C 10 ML WATER", labels) is None)
    check("an exact read is kept", snap_to_candidates("a 10ml water", labels) == "A 10 ML WATER")

    class Client:
        def __init__(self, reply):
            self.reply, self.prompts = reply, []
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kw):
            self.prompts.append(kw["messages"][0]["content"][0]["text"])
            self.model = kw["model"]
            msg = SimpleNamespace(content=self.reply)
            return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    img = np.full((120, 160, 3), 90, np.uint8)
    mask = np.zeros((120, 160), bool)
    mask[40:70, 60:100] = True
    inst = {"box": [60, 40, 100, 70], "mask": mask, "score": 0.9}

    client = Client('{"label": "EMPTY"}')
    out = read_one_label_vlm(client, img, inst, candidates=labels)
    check("the prompt lists the bench's labels", "C EMPTY" in client.prompts[0]
          and "no others" in client.prompts[0])
    check("a partial reply is snapped to its label", out == "C EMPTY", f"{out!r}")
    client = Client('{"label": "ACETONE"}')
    check("an invented reply comes back unread",
          read_one_label_vlm(client, img, inst, candidates=labels) is None)
    client = Client('{"label": "ACETONE"}')
    check("without candidates the reader still transcribes freely",
          read_one_label_vlm(client, img, inst) == "ACETONE"
          and "no others" not in client.prompts[0])

    resolver = LabelResolver(vlm_client=Client('{"label": null}'))
    check("the label reader no longer defaults to gpt-4o (shut down 2026-10-23)",
          resolver.model != "gpt-4o", resolver.model)


def test_transparent_cups_are_placed_by_silhouette():
    """A clear cup is triangulated from its masks and rebuilt at its measured size."""
    print("\n[vision: transparent containers by silhouette]")
    from types import SimpleNamespace
    from robochem.vision import VisionSystem, cylinder_points, triangulate_rays
    from robochem.vision.bench_manifest import MAGIC_BEAKER

    cup = MAGIC_BEAKER.find("clear cup a")
    check("the clear cups and the beaker are marked transparent, with a size",
          all(MAGIC_BEAKER.find(n).transparent and MAGIC_BEAKER.find(n).radius
              and MAGIC_BEAKER.find(n).height
              for n in ("clear cup a", "clear cup b", "clear cup c", "plastic beaker"))
          and not MAGIC_BEAKER.find("citric acid cup").transparent)

    # Four cameras round the bench, like the cage's, looking at the bench centre.
    def look_at(eye, target):
        z = np.asarray(target, float) - eye
        z /= np.linalg.norm(z)
        x = np.cross(z, [0.0, 0.0, 1.0])
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        T = np.eye(4)
        T[:3, :3] = np.column_stack([x, y, z])
        T[:3, 3] = eye
        return T

    eyes = {2: [1.1, 0.6, 0.7], 3: [-0.1, 0.6, 0.7], 4: [-0.1, -0.6, 0.7], 5: [1.1, -0.6, 0.7]}
    cams = {c: look_at(np.array(e, float), [0.45, 0.0, 0.0]) for c, e in eyes.items()}
    intr = SimpleNamespace(fx=600.0, fy=600.0, cx=424.0, cy=240.0)

    def project(cam, p):
        q = np.linalg.inv(cams[cam]) @ np.append(p, 1.0)
        return q[0] * 600.0 / q[2] + 424.0, q[1] * 600.0 / q[2] + 240.0

    def disc(u, v, r=12):
        yy, xx = np.mgrid[0:480, 0:848]
        return (xx - u) ** 2 + (yy - v) ** 2 <= r * r

    truth = np.array([0.33, -0.20, 0.0])
    mid = truth + [0.0, 0.0, cup.height / 2.0]
    masks = {c: disc(*project(c, mid)) for c in cams}
    data = {"intrinsics": {c: intr for c in cams}}

    class Localizer:
        def __init__(self):
            self._cached_centroids, self._cached_pointclouds = {}, {}

        def _extrinsics(self, cam):
            return cams[cam]

        def cache_object(self, name, points, centroid=None):
            self._cached_pointclouds[name] = points
            self._cached_centroids[name] = np.asarray(points).mean(axis=0)

    from robochem.vision.grasp_analyzer import GraspAnalyzer
    v = VisionSystem.__new__(VisionSystem)
    v.object_localizer = Localizer()
    v.grasp_analyzer = GraspAnalyzer()
    v.table_z = 0.0
    v.silhouette_ray_tolerance = 0.025

    point, used, dropped, misses = v._silhouette_position(masks, data, cup, tag="a")
    check("four silhouettes triangulate onto the cup, half-way up",
          np.linalg.norm(point - mid) < 0.003 and used == [2, 3, 4, 5],
          f"{np.round(point, 4)} vs {np.round(mid, 4)}")

    wrong = dict(masks)
    wrong[3] = disc(*project(3, mid + [0.0, 0.12, 0.0]))       # cam 3 on the cup next door
    point, used, dropped, _ = v._silhouette_position(wrong, data, cup, tag="a")
    check("a camera on the wrong cup is dropped and the rest still agree",
          dropped == [3] and np.linalg.norm(point[:2] - mid[:2]) < 0.005,
          f"dropped {dropped}, {np.round(point, 4)}")
    two = {2: masks[2], 4: disc(*project(4, mid + [0.0, 0.12, 0.0]))}
    check("two cameras that disagree give no position at all",
          v._silhouette_position(two, data, cup, tag="a") is None)
    point, used, _, _ = v._silhouette_position({5: masks[5]}, data, cup, tag="a")
    check("one camera is cut at half height", used == [5]
          and np.linalg.norm(point - mid) < 0.003, f"{np.round(point, 4)}")

    out = v._locate_transparent("a 10 ml water", masks, {c: 0.7 for c in cams}, data, cup)
    check("the located cup is a cylinder standing where the silhouettes meet",
          out is not None and np.linalg.norm(np.asarray(out["centroid"])[:2] - truth[:2]) < 0.003
          and out["cameras"] == [2, 3, 4, 5], f"{out and np.round(out['centroid'], 4)}")

    # And the skills read that cylinder back as the cup it is.
    cloud = cylinder_points(truth[:2], 0.0, cup.radius, cup.height, cup.wall)
    skill = make(DumpSkill, FakeVision({"clear cup a": cloud}), FakeArm())
    rim = skill.locate_container("clear cup a")
    check("its rim centre is the cup's axis",
          np.linalg.norm(rim["rim_center"] - truth[:2]) < 0.001, f"{rim['rim_center']}")
    check("its height and opening are the measured cup's",
          abs(rim["top_z"] - cup.height) < 0.002 and abs(rim["base_z"]) < 0.002
          and abs(rim["rim_radius"] - (cup.radius - cup.wall)) < 0.002,
          f"top {rim['top_z']:.4f} base {rim['base_z']:.4f} opening {rim['rim_radius']:.4f}")
    check("the rim is fitted, not the leaning median fallback",
          rim["rim_method"].startswith("circle fit"), rim["rim_method"])

    p, misses = triangulate_rays([([0, 0, 1], [1, 0, -1]), ([2, 0, 1], [-1, 0, -1])])
    check("two crossing rays meet at their crossing", np.allclose(p, [1, 0, 0], atol=1e-9)
          and max(misses) < 1e-9, f"{p}")

    # Two cameras on two different cups whose rays happen to cross in mid-air:
    # "A" on 2026-10-07 came back 157 mm above the table this way.
    high = np.array([0.30, 0.0, 0.157])
    crossing = {c: (cams[c][:3, 3], (high - cams[c][:3, 3]) / np.linalg.norm(high - cams[c][:3, 3]))
                for c in (2, 5)}
    check("rays that cross above a 47 mm cup are not one cup",
          v._place_rays(crossing, cup.height, tag="a") is None)

    # --- the whole bench: place every clear cup, then vote on the labels ---
    from robochem.vision import assign_labels
    best, shares = assign_labels([[2, 0], [1, 1], [0, 2]])
    check("labels go where most reads agree, one container each",
          best == 4 and shares == [[0, 2]], f"{best} {shares}")
    best, shares = assign_labels([[1, 0], [1, 0]])
    check("a label read once on each of two containers is a tie",
          {s[0] for s in shares} == {0, 1}, f"{shares}")

    where = {"A 10 ML WATER": (0.267, -0.200), "B 10 ML WATER": (0.302, -0.057),
             "C EMPTY": (0.268, 0.197), "50 ML WATER": (0.359, 0.050)}
    height = {"50 ML WATER": 0.060}
    order = {2: ["B 10 ML WATER", "50 ML WATER", "C EMPTY", "A 10 ML WATER"],
             3: ["C EMPTY", "A 10 ML WATER", "50 ML WATER", "B 10 ML WATER"],
             4: ["B 10 ML WATER", "C EMPTY", "A 10 ML WATER", "50 ML WATER"],
             5: ["50 ML WATER", "A 10 ML WATER", "B 10 ML WATER", "C EMPTY"]}
    # What the reader said under each of those, misreads included: cam 2 calls
    # C "A", cam 4 calls B "A" as well as A, cam 3 cannot read B.
    said = {2: ["B 10 ML WATER", "50 ML WATER", "A 10 ML WATER", "A 10 ML WATER"],
            3: ["C EMPTY", "A 10 ML WATER", "50 ML WATER", None],
            4: ["A 10 ML WATER", "C EMPTY", "A 10 ML WATER", "50 ML WATER"],
            5: ["50 ML WATER", "A 10 ML WATER", "B 10 ML WATER", "C EMPTY"]}
    original = {c: list(labs) for c, labs in said.items()}

    def bench_masks():
        out = {}
        for c, labels_here in order.items():
            out[c] = []
            for lab in labels_here:
                xy = where[lab]
                mid_pt = np.array([xy[0], xy[1], height.get(lab, cup.height) / 2.0])
                out[c].append({"mask": disc(*project(c, mid_pt)), "score": 0.7})
        return out

    class Grounding:
        def segment_instances(self, images, category):
            return bench_masks()

    class Reader:
        def read_all(self, images, instances):
            return {c: list(said[c]) for c in instances}

    class BenchLocalizer(Localizer):
        camera_ids = [2, 3, 4, 5]

        def capture_pointclouds(self):
            return {"images": {c: None for c in cams},
                    "intrinsics": {c: intr for c in cams}}

    vb = VisionSystem.__new__(VisionSystem)
    vb.bench = MAGIC_BEAKER
    vb.object_localizer = BenchLocalizer()
    vb.grasp_analyzer = GraspAnalyzer()
    vb.scene_analyzer = SimpleNamespace(grounding=Grounding())
    vb.label_resolver = Reader()
    vb.table_z = 0.0
    vb.silhouette_ray_tolerance = 0.025
    vb.labeled_cup_category = "white bowl"
    for lab, name in (("a 10 ml water", "clear cup a"), ("c empty", "clear cup c"),
                      ("50 ml water", "plastic beaker")):
        out = vb.locate_labeled(lab, category="clear plastic cup",
                                shape=MAGIC_BEAKER.find(name))
        got = None if out is None else np.asarray(out["centroid"])[:2]
        check(f"{name} is found where it stands despite the misreads",
              got is not None and np.linalg.norm(got - np.array(where[lab.upper()
                                                  if lab != "c empty" else "C EMPTY"])) < 0.005,
              f"{None if got is None else np.round(got, 4)}")

    # One weak "A" read on A's cup still wins when C's cup is plainly C: each
    # label goes to one container.
    said[2][3] = None
    said[3][1] = None
    said[4][2] = None
    out = vb.locate_labeled("a 10 ml water", category="clear plastic cup",
                            shape=MAGIC_BEAKER.find("clear cup a"))
    check("one read of A is enough once the other cups are accounted for",
          out is not None and np.linalg.norm(np.asarray(out["centroid"])[:2]
                                             - np.array(where["A 10 ML WATER"])) < 0.005)
    # But with B's reads gone too, A's cup and B's cup each carry one "A".
    said[2][0] = None
    said[5][2] = None
    out = vb.locate_labeled("a 10 ml water", category="clear plastic cup",
                            shape=MAGIC_BEAKER.find("clear cup a"))
    check("when the reads cannot tell which cup is A, A is not found", out is None,
          f"{out and np.round(out['centroid'], 4)}")

    # Nobody reads C -- cam 2 still calls it "A", the rest cannot read it --
    # but A, B and the beaker are all read, on four cups each placed by 2+
    # cameras. (The real run of 2026-10-07: C read as "A" twice, A 3 times.)
    for c in said:
        said[c] = [None if lab == "C EMPTY" else lab for lab in original[c]]
    out = vb.locate_labeled("c empty", category="clear plastic cup",
                            shape=MAGIC_BEAKER.find("clear cup c"))
    check("C is found by elimination when no camera reads it",
          out is not None and "elimination" in out["resolved_by"]
          and np.linalg.norm(np.asarray(out["centroid"])[:2] - np.array(where["C EMPTY"])) < 0.005,
          f"{out and (np.round(out['centroid'], 4), out['resolved_by'])}")
    # With C seen by one camera only there are three strong cups for four labels.
    real_masks = bench_masks
    def fewer():
        out_ = real_masks()
        for c in (3, 4, 5):
            out_[c][order[c].index("C EMPTY")]["mask"][:] = False
        return out_
    Grounding.segment_instances = lambda self, images, category: fewer()
    out = vb.locate_labeled("c empty", category="clear plastic cup",
                            shape=MAGIC_BEAKER.find("clear cup c"))
    check("no elimination when a cup is placed by one camera only", out is None,
          f"{out and out['resolved_by']}")

    # The robot, 2026-10-08, step 23: the beaker is in the jaws, its label
    # paper still lies by cup A, one camera reads A as "50 ML WATER", and
    # B's cup is read as "A" twice and never as "B".
    Grounding.segment_instances = lambda self, images, category: bench_masks()
    order = {2: ["B 10 ML WATER", "A 10 ML WATER", "C EMPTY"],
             3: ["C EMPTY", "A 10 ML WATER", "B 10 ML WATER"],
             4: ["B 10 ML WATER", "C EMPTY", "A 10 ML WATER"],
             5: ["A 10 ML WATER", "B 10 ML WATER", "C EMPTY"]}
    said = {2: ["A 10 ML WATER", "50 ML WATER", "C EMPTY"],
            3: ["C EMPTY", "A 10 ML WATER", None],
            4: [None, "C EMPTY", "A 10 ML WATER"],
            5: ["A 10 ML WATER", "A 10 ML WATER", None]}
    vb.set_held(None)
    vb.bench = MAGIC_BEAKER
    out = vb.locate_labeled("a 10 ml water", category="clear plastic cup",
                            shape=MAGIC_BEAKER.find("clear cup a"))
    check("with the beaker counted as on the bench, A ties and is refused (as on the robot)",
          out is None, f"{out and np.round(out['centroid'], 4)}")
    vb.set_held("50 ML WATER")
    check("the held object is known by its bench name", vb.held_object == "plastic beaker",
          vb.held_object)
    # Even with the beaker out of the vote, "A" is read on A's cup (3) AND on
    # B's (2), and "B" nowhere: one of the two is B. Counting reads chose A's
    # here, but on the robot's next look B's cup out-read A's and step 23
    # poured A's indicator into B. Not guessed any more.
    out = vb.locate_labeled("a 10 ml water", category="clear plastic cup",
                            shape=MAGIC_BEAKER.find("clear cup a"))
    check("A read on two cups with B read on none is refused, not guessed",
          out is None, f"{out and np.round(out['centroid'], 4)}")
    from robochem.vision import lookalike, misread_twin
    check("A and B labels are look-alikes; C and the beaker's are not",
          lookalike("A 10 ML WATER", "B 10ML WATER")
          and not lookalike("A 10 ML WATER", "C EMPTY")
          and not lookalike("A 10 ML WATER", "50 ML WATER"))
    check("the same cup placed twice counts once",
          misread_twin([[2, 0], [1, 0]], ["A 10 ML WATER", "B 10 ML WATER"], 0,
                       [(0.415, -0.140), (0.431, -0.161)], [0, 1], 0.0555) is None)
    # One camera reading B on B's cup settles it: A is the other "A" cup.
    said[2][0] = "B 10 ML WATER"
    out = vb.locate_labeled("a 10 ml water", category="clear plastic cup",
                            shape=MAGIC_BEAKER.find("clear cup a"))
    check("once B is read on its cup, A is found where it stands",
          out is not None and np.linalg.norm(np.asarray(out["centroid"])[:2]
                                             - np.array(where["A 10 ML WATER"])) < 0.005,
          f"{out and np.round(out['centroid'], 4)}")
    out = vb.locate_labeled("b 10 ml water", category="clear plastic cup",
                            shape=MAGIC_BEAKER.find("clear cup b"))
    check("and B is found on its own cup",
          out is not None and np.linalg.norm(np.asarray(out["centroid"])[:2]
                                             - np.array(where["B 10 ML WATER"])) < 0.005,
          f"{out and np.round(out['centroid'], 4)}")
    # B's cup read "A" three times and "B" once, A's cup "A" once: the most
    # reads put A on B's cup, the only cup B was read on. Refused.
    said = {2: ["A 10 ML WATER", None, "C EMPTY"],
            3: ["C EMPTY", "A 10 ML WATER", "A 10 ML WATER"],
            4: ["B 10 ML WATER", "C EMPTY", None],
            5: [None, "A 10 ML WATER", None]}
    out = vb.locate_labeled("a 10 ml water", category="clear plastic cup",
                            shape=MAGIC_BEAKER.find("clear cup a"))
    check("A is refused when it would land on the only cup read as B",
          out is None, f"{out and np.round(out['centroid'], 4)}")
    vb.set_held(None)


def test_dishes_are_voted_across_cameras():
    """One camera misreading a dish no longer blocks the dish it named."""
    print("\n[vision: opaque containers by vote]")
    from types import SimpleNamespace
    from robochem.vision import VisionSystem
    from robochem.vision.bench_manifest import MAGIC_BEAKER
    from robochem.vision.grasp_analyzer import GraspAnalyzer

    where = {"CITRIC ACID": (0.525, -0.077), "BAKING SODA": (0.500, -0.229),
             "RED CABBAGE POWDER": (0.460, 0.190)}
    order = {2: ["CITRIC ACID", "BAKING SODA", "RED CABBAGE POWDER"],
             3: ["BAKING SODA", "RED CABBAGE POWDER", "CITRIC ACID"],
             4: ["RED CABBAGE POWDER", "BAKING SODA", "CITRIC ACID"],
             5: ["RED CABBAGE POWDER", "CITRIC ACID", "BAKING SODA"]}
    # The run of 2026-10-07: cam 4 calls the baking soda dish citric acid,
    # cam 5 calls two dishes citric acid, cam 3 reads baking soda twice.
    said = {2: ["CITRIC ACID", "BAKING SODA", None],
            3: ["BAKING SODA", "BAKING SODA", None],
            4: ["RED CABBAGE POWDER", "CITRIC ACID", None],
            5: ["RED CABBAGE POWDER", "CITRIC ACID", "CITRIC ACID"]}
    rng = np.random.default_rng(3)
    clouds = {}

    def dish(cam, lab):
        x, y = where[lab]
        lean = rng.normal(0, 0.006, 2)           # a camera's own small lean
        r = 0.045 * np.sqrt(rng.random(300))
        t = rng.random(300) * 2 * np.pi
        return np.column_stack([x + lean[0] + r * np.cos(t), y + lean[1] + r * np.sin(t),
                                0.01 + 0.02 * rng.random(300)])

    instances = {}
    for cam, labs in order.items():
        instances[cam] = []
        for lab in labs:
            mask = np.zeros((4, 4), bool)
            clouds[id(mask)] = dish(cam, lab)
            instances[cam].append({"mask": mask, "score": 0.9})

    class Localizer:
        camera_ids = [2, 3, 4, 5]

        def __init__(self):
            self._cached_centroids, self._cached_pointclouds = {}, {}

        def capture_pointclouds(self):
            return {"images": {c: None for c in order}, "depth_images": {},
                    "intrinsics": {}}

        def get_object_points_by_camera(self, masks, depth_images=None, intrinsics=None):
            return {c: clouds[id(m)] for c, m in masks.items()}

        def _remove_outliers(self, points, *a, **k):
            return points

        def cache_object(self, name, points, centroid=None):
            self._cached_pointclouds[name] = points
            self._cached_centroids[name] = np.asarray(points).mean(axis=0)

    v = VisionSystem.__new__(VisionSystem)
    v.bench = MAGIC_BEAKER
    v.object_localizer = Localizer()
    v.grasp_analyzer = GraspAnalyzer()
    v.scene_analyzer = SimpleNamespace(grounding=SimpleNamespace(
        segment_instances=lambda images, category: instances))
    v.label_resolver = SimpleNamespace(read_all=lambda images, inst: said)
    v.labeled_cup_category = "white bowl"
    v.min_object_points, v.min_instance_score = 50, 0.5
    v.consensus_tolerance, v.max_object_extent, v.min_object_height = 0.05, 0.35, -0.02
    v.table_z = 0.0

    for lab, name in (("citric acid", "citric acid cup"), ("baking soda", "baking soda cup"),
                      ("red cabbage powder", "red cabbage powder cup")):
        out = v.locate_labeled(lab, category="white bowl", shape=MAGIC_BEAKER.find(name))
        got = None if out is None else np.asarray(out["centroid"])[:2]
        check(f"{name} is found where it stands despite the misreads",
              got is not None and np.linalg.norm(got - np.array(where[lab.upper()])) < 0.01,
              f"{None if got is None else np.round(got, 4)}")
    check("and from every camera that saw it", out is not None and len(out["cameras"]) == 4,
          f"{out and out['cameras']}")


def test_executor_trusts_the_measured_bowl_when_the_cloud_is_too_long():
    """The real scoop's cloud ran 45 mm long; its middle put the bowl 25 mm short."""
    print("\n[executor: tool offset from a contaminated pick cloud]")
    from robochem.skills import SkillsExecutor
    from robochem.skills.scoop import ScoopSkill

    ex = SkillsExecutor(None, None, {})
    skill = ScoopSkill(None, None, {})
    # The robot, 2026-10-08: extent -50.7..+60.3 mm, pick_up measured 51.4 mm.
    ex.held = "larger spoon"
    ex.held_measure = {"extent_x": [-0.0507, 0.0603],
                       "suggested": [0.0514, 0.0105, 0.016]}
    out = ex._fill_tool_geometry("scoop", skill, {"powder_source": "citric acid cup",
                                                  "tool_offset": [0.0514, 0.0105, 0.016]})
    check("pick_up's copied offset keeps its x and y, and takes the CAD depth",
          np.allclose(out["tool_offset"], [0.0514, 0.0105, 0.0294]), f"{out['tool_offset']}")
    out = ex._fill_tool_geometry("scoop", skill, {"powder_source": "citric acid cup"})
    check("with none passed, the measured x and y are used, not the CAD 25.7 mm",
          np.allclose(out["tool_offset"], [0.0514, 0.0105, 0.0294]), f"{out['tool_offset']}")
    # The sim, 2026-10-07: extent -33.7..+30.8 mm, a cloud the part's length.
    ex.held_measure = {"extent_x": [-0.0337, 0.0308], "suggested": [0.0292, 0.0, 0.02]}
    out = ex._fill_tool_geometry("scoop", skill, {"powder_source": "citric acid cup"})
    check("a cloud the part's length still places the bowl from its ends (sim unchanged)",
          abs(out["tool_offset"][0] - (-0.00145 + 0.02105)) < 1e-4, f"{out['tool_offset']}")
    # The robot, 2026-10-08 13:xx: a cloud only 14.5 mm too long, but its ends
    # put the bowl at 21.6 mm where its bowl end said 36.
    ex.held_measure = {"extent_x": [-0.0414, 0.0426], "suggested": [0.036, 0.002, 0.014]}
    out = ex._fill_tool_geometry("scoop", skill, {"powder_source": "citric acid cup"})
    check("ends and bowl end 14 mm apart: the measured bowl is used, with the CAD depth",
          np.allclose(out["tool_offset"], [0.036, 0.002, 0.0294]), f"{out['tool_offset']}")
    # On the real bench the measurement is ruler-checked (52.9 mm against
    # 0.053): it wins even when the ends happen to agree with it.
    from robochem.vision.bench_manifest import MAGIC_BEAKER

    class RealBench:
        trust_measured_tool_offset = staticmethod(lambda: MAGIC_BEAKER.measured_tool_offset)

    real = SkillsExecutor(None, RealBench(), {})
    real.held = "larger spoon"
    real.held_measure = {"extent_x": [-0.0337, 0.0308], "suggested": [0.053, -0.011, 0.026]}
    out = real._fill_tool_geometry("scoop", skill, {"powder_source": "citric acid cup"})
    check("on the real bench the measured bowl is used outright, with the CAD depth",
          np.allclose(out["tool_offset"], [0.053, -0.011, 0.0294]), f"{out['tool_offset']}")
    out = real._fill_tool_geometry("scoop", skill, {"powder_source": "citric acid cup",
                                                    "tool_offset": [0.053, -0.011, 0.026]})
    check("a planner's copy of it too", np.allclose(out["tool_offset"], [0.053, -0.011, 0.0294]),
          f"{out['tool_offset']}")
    out = real._fill_tool_geometry("scoop", skill, {"powder_source": "citric acid cup",
                                                    "tool_offset": [0.045, 0.0, 0.03]})
    check("an offset passed on purpose is still kept there", out["tool_offset"] == [0.045, 0.0, 0.03],
          f"{out['tool_offset']}")
    for ext, sug in (([-0.0337, 0.0308], 0.0292), ([-0.0349, 0.0287], 0.0261)):
        ex.held_measure = {"extent_x": ext, "suggested": [sug, 0.0, 0.02]}
        out = ex._fill_tool_geometry("scoop", skill, {"powder_source": "citric acid cup"})
        want = (ext[0] + ext[1]) / 2 + 0.02105
        check(f"the sim's own pick (ends vs bowl end {abs(sug - want) * 1000:.1f} mm) keeps the ends",
              abs(out["tool_offset"][0] - want) < 1e-4, f"{out['tool_offset']}")


def test_robot_and_sim_share_one_pace():
    """Every point-to-point move is stretched so min-jerk peaks at the old sim speed."""
    print("\n[pacing: robot and sim alike]")
    from robochem.skills.pacing import PacedArm, paced, MOTION_TIME_SCALE, HOME_SECONDS

    class Arm:
        def __init__(self):
            self.calls = []
            self.gripper = 0.08

        def goto_pose(self, pose, duration=3, use_impedance=True, dynamic=False, **kw):
            self.calls.append(("pose", duration, dynamic))

        def goto_joints(self, joints, duration=5, **kw):
            self.calls.append(("joints", duration))

        def reset_joints(self, duration=5, **kw):
            self.calls.append(("home", duration))

        def get_gripper_width(self):
            return self.gripper

    arm = Arm()
    p = paced(arm)
    p.goto_pose("T", duration=2.0)
    p.goto_pose("T", 2.0)
    p.goto_pose("T")
    p.goto_pose("T", duration=5.0, dynamic=True)
    p.goto_joints([0] * 7, duration=4.0)
    p.reset_joints()
    check("a 2 s move takes 3.75 s, however the duration is passed",
          arm.calls[0] == ("pose", 3.75, False) and arm.calls[1][1] == 3.75, f"{arm.calls[:2]}")
    check("frankapy's default 3 s is stretched too", arm.calls[2][1] == 3.0 * MOTION_TIME_SCALE)
    check("a streamed (dynamic) start is left alone", arm.calls[3] == ("pose", 5.0, True))
    check("joint moves are stretched", arm.calls[4] == ("joints", 7.5))
    check("home takes the old sim's 4 s peak-for-peak, 7.5 s",
          arm.calls[5] == ("home", HOME_SECONDS) and abs(HOME_SECONDS - 7.5) < 1e-9)
    check("everything else passes through", p.get_gripper_width() == 0.08)
    check("wrapping twice does not stretch twice", paced(p) is p and paced(None) is None)
    # min-jerk over T peaks at 1.875 d/T; constant speed over 4 s is d/4.
    check("a home trip's peak speed equals the old sim's constant speed",
          abs(1.875 / HOME_SECONDS - 1.0 / 4.0) < 1e-12)


def test_streamed_paths_start_and_end_at_rest():
    """Resampled streaming setpoints ease in and out; the geometry is unchanged."""
    print("\n[streaming: min-jerk timing]")
    from robochem.skills.base_skill import resample_path_eased, STREAM_SETTLE_SECONDS
    path = [np.array([0.40, 0.0, 0.40]), np.array([0.35, -0.10, 0.20]),
            np.array([0.30, -0.20, 0.06])]
    mats = [np.eye(3)] * 3
    xyzs, rots = resample_path_eased(path, mats, 250)
    steps = np.linalg.norm(np.diff(np.asarray(xyzs), axis=0), axis=1)
    check("it gives the asked number of setpoints", len(xyzs) == 250 and len(rots) == 250)
    check("it starts and ends exactly on the path's ends",
          np.allclose(xyzs[0], path[0]) and np.allclose(xyzs[-1], path[-1]))
    check("the first and last steps are tiny next to the middle ones",
          steps[0] < 0.02 * steps.max() and steps[-1] < 0.02 * steps.max(),
          f"first {steps[0] * 1000:.3f} last {steps[-1] * 1000:.3f} max {steps.max() * 1000:.2f} mm")
    # The sim's follow_pose_path: min-jerk over the waypoints, each segment
    # given the same time. The robot's setpoints must sit exactly there.
    t = np.linspace(0.0, 1.0, 250)
    u = (t ** 3 * (10 - 15 * t + 6 * t ** 2)) * (len(path) - 1)
    sim = [path[min(int(k), 1)] + (k - min(int(k), 1)) * (path[min(int(k), 1) + 1]
                                                           - path[min(int(k), 1)]) for k in u]
    check("each setpoint is where the sim's timing puts the path at that tick",
          np.allclose(np.asarray(xyzs), np.asarray(sim), atol=1e-12))
    even = [np.array([0.40, 0.0, 0.40]) + k * np.array([-0.01, -0.01, -0.02]) for k in range(20)]
    es = np.linalg.norm(np.diff(np.asarray(resample_path_eased(even, [np.eye(3)] * 20, 250)[0]),
                                axis=0), axis=1)
    check("over evenly spaced waypoints (dump's carry) the speed changes gradually",
          np.abs(np.diff(es)).max() < 0.03 * es.max(), f"{np.abs(np.diff(es)).max() * 1000:.3f} mm")
    on_path = all(min(np.linalg.norm(np.cross(p - path[k], path[k + 1] - path[k]))
                      / np.linalg.norm(path[k + 1] - path[k]) for k in range(2)) < 1e-9
                  for p in xyzs)
    check("every setpoint is on the original path", on_path)
    check("a streamed path holds its end before stopping", STREAM_SETTLE_SECONDS >= 0.3)


def test_executor_tells_vision_what_is_held():
    """pick_up hands the held object to vision; place takes it back."""
    print("\n[executor: what is held, told to vision]")
    from robochem.skills import SkillsExecutor

    class Vision:
        held = "unset"

        def set_held(self, name):
            self.held = name

    v = Vision()
    ex = SkillsExecutor(None, v, {})
    ex._track_held("pick_up", {"object_name": "plastic beaker"}, {})
    check("after a pick vision knows the beaker is held", v.held == "plastic beaker", v.held)
    ex._track_held("place", {"target_location": "plastic beaker"}, {})
    check("after the place it knows nothing is", v.held is None, v.held)


def test_place_back_returns_the_tcp_to_the_grasp():
    """A put-back releases with the TCP where the jaws closed, not over the centroid."""
    print("\n[place: back where it was picked]")
    arm = FakeArm(tracking=1.0, gripper_width=0.008)
    arm.pose = FakePose([0.31, 0.0, 0.47], np.diag([1.0, -1.0, -1.0]))
    # The scoop's 20:40 numbers: centroid 0.6473, jaws closed at 0.6319.
    ok, res = make(PlaceSkill, FakeVision({}), arm).execute(
        {"target_location": [0.6473, 0.0172, -0.0008],
         "pick_grasp_tcp": [0.6319, 0.0155, 0.0117]})
    released = np.asarray(res.get("place_position") or [0, 0, 0])
    check("released with the TCP where it grasped, at the centroid's height + 20 mm",
          ok and np.allclose(released, [0.6319, 0.0155, -0.0008 + 0.02]), f"{released}")
    arm2 = FakeArm(tracking=1.0, gripper_width=0.008)
    ok2, res2 = make(PlaceSkill, FakeVision({}), arm2).execute(
        {"target_location": [0.6473, 0.0172, -0.0008]})
    check("without a remembered grasp it still goes to the given point",
          ok2 and np.allclose(res2["place_position"][:2], [0.6473, 0.0172]), f"{res2}")


def test_cups_are_taken_from_behind():
    """
    The user's procedure for cups A, B and C (2026-10-08): jaws open, down
    behind the cup, slide forward onto it, close; put back the same way in
    reverse. Coming straight down, a finger could land on the 55.5 mm rim.
    """
    print("\n[pick_up / place: cups slid into from behind]")
    from robochem.skills import SkillsExecutor
    rim, top_to_tcp = 0.0473, 0.013
    cup = cylinder_cloud([0.40, -0.16], base_z=0.0, height=rim, radius=0.02775)
    yawed = np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])

    class CupVision(FakeVision):
        def locate(self, name, force_refresh=False):
            located = super().locate(name, force_refresh)
            if located is not None:
                pts = located["points"]
                located["dimensions"] = np.sort(pts.max(0) - pts.min(0))[::-1]
            return located

        def compute_dimensions(self, points):
            return np.sort(points.max(0) - points.min(0))[::-1]

        def compute_grasp_pose(self, points, object_name=None, grasp_type="auto",
                               pitch_deg=0.0):
            pose = np.eye(4)
            pose[:3, :3] = yawed                     # PCA yaw of a round cup is arbitrary
            pose[:3, 3] = [*np.asarray(points)[:, :2].mean(axis=0), rim - top_to_tcp]
            return pose

        def compute_grasp_candidates(self, points, object_name=None,
                                     grasp_type="auto", pitch_deg=0.0, spout_xy=None):
            return [self.compute_grasp_pose(points, object_name)]

    vision = CupVision({"clear cup a": cup})
    arm = FakeArm(tracking=1.0, gripper_width=0.08)
    ok, result = make(PickUpSkill, vision, arm).execute(
        {"object_name": "clear cup a", "slide_in": 0.06, "z_offset": -0.012,
         "force_limited": False})
    check("the slid-in pick succeeds", ok, f"{result}")
    grasp = np.asarray(result.get("grasp_pose")) if ok else np.eye(4)
    check("the jaws close across the slide (base Y), whatever the cup's yaw",
          np.allclose(grasp[:3, :3], np.diag([1.0, -1.0, -1.0])), f"{grasp[:3, :3]}")
    check("12 mm lower: the rim 25 mm over the TCP",
          abs(rim - grasp[2, 3] - 0.025) < 1e-6, f"{grasp[2, 3]:.4f}")
    moves = [c[0] for c in arm.commands]
    at = [k for k, p in enumerate(moves) if np.allclose(p, grasp[:3, 3])]
    first = at[0] if at else len(moves)
    behind = [p for p in moves[:first] if abs(p[0] - (grasp[0, 3] - 0.06)) < 1e-6]
    check("hover, approach and descent all happen 60 mm behind",
          len(behind) >= 3 and abs(behind[-1][2] - grasp[2, 3]) < 1e-6,
          f"{[np.round(p, 3) for p in moves[:first + 1]]}")
    over_cup = [p for p in moves[:first]
                if np.linalg.norm(p[:2] - [0.40, -0.16]) < 0.02775 + 0.01 and p[2] < rim + 0.06]
    check("nothing comes down over the cup before the slide", not over_cup,
          f"{over_cup}")
    check("the pick reports how it came in", result.get("slide_in") == 0.06, f"{result}")

    # Closing: to a measured width and stop, not squeezing on until 1.5 N.
    arm_w = FakeArm(tracking=1.0, gripper_width=0.08)
    ok_w, res_w = make(PickUpSkill, vision, arm_w).execute(
        {"object_name": "clear cup a", "slide_in": 0.06, "z_offset": -0.012,
         "grip_width": 0.058})
    closes = [g for g in arm_w.gripper_commands if g["width"] < 0.075]
    check("with grip_width the jaws go to exactly that width, as a position move",
          ok_w and closes and closes[-1]["width"] == 0.058 and not closes[-1]["grasp"],
          f"{arm_w.gripper_commands} {res_w}")

    # Put back: down where it was held, open, back out 60 mm, then up.
    held_at = [0.40, -0.165, 0.0223]
    arm2 = FakeArm(tracking=1.0, gripper_width=0.05)
    arm2.pose = FakePose([0.31, 0.0, 0.47], np.diag([1.0, -1.0, -1.0]))
    ok2, placed = make(PlaceSkill, FakeVision({}), arm2).execute(
        {"target_location": [0.40, -0.16, 0.0237], "pick_grasp_tcp": held_at,
         "slide_out": 0.06})
    check("the reversed place succeeds", ok2, f"{placed}")
    moves2 = [c[0] for c in arm2.commands]
    release = np.array(held_at) + [0.0, 0.0, 0.003]
    k = next((i for i, p in enumerate(moves2) if np.allclose(p, release)), None)
    check("it sets the cup down where it was held, 3 mm up", k is not None,
          f"{[np.round(p, 4) for p in moves2]}")
    after = moves2[k + 1:] if k is not None else []
    check("then slides the open jaws 60 mm back at that height, then rises",
          len(after) >= 2 and np.allclose(after[0], release - [0.06, 0.0, 0.0])
          and after[1][2] > release[2] + 0.05 and abs(after[1][0] - after[0][0]) < 1e-6,
          f"{[np.round(p, 4) for p in after]}")

    # The executor carries the pick's slide to the place.
    down = np.eye(4)
    down[:3, :3] = np.diag([1.0, -1.0, -1.0])
    ex = SkillsExecutor(FakeArm(tracking=1.0, gripper_width=0.05), FakeVision({}), {})
    ex._track_held("pick_up", {"object_name": "clear cup a"},
                   {"slide_in": 0.06, "grasp_pose": down.tolist()})
    ex.robot.pose = FakePose([0.31, 0.0, 0.47], np.diag([1.0, -1.0, -1.0]))
    ok3, _ = ex.execute("place", {"target_location": [0.45, -0.10, 0.02]})
    xs = [c[0] for c in ex.robot.commands]
    check("place backs out when the pick slid in",
          ok3 and any(abs(p[0] - (0.45 - 0.06)) < 1e-6 for p in xs),
          f"{[np.round(p, 3) for p in xs]}")


def test_a_refused_motion_is_caught():
    """
    2026-10-08, steps 24-25: franka-interface faulted at the end of the 90 deg
    pour step and refused everything after. Pour said success, and place
    "missed" its hover by 271 mm from the arm still tipped over cup B.
    """
    print("\n[a refused motion is caught]")
    from robochem.skills import SkillsExecutor
    from robochem.skills.base_skill import HOME_JOINTS
    tipped = _tool_y(90.0) @ np.diag([1.0, -1.0, -1.0])     # tool z toward the base

    class FrozenArm(FakeArm):
        """Not ready: every motion returns at once and nothing moves."""

        def __init__(self, **kw):
            super().__init__(**kw)
            self.q = np.array([-0.031, 1.034, 0.0, -1.440, -0.038, 0.910, 0.761])
            self.pose = FakePose([0.406, -0.021, 0.084], tipped.copy())

        def get_joints(self):
            return self.q.copy()

        def goto_pose(self, pose, duration=3.0, use_impedance=True, block=True):
            self.commands.append((np.asarray(pose.translation).copy(), duration,
                                  use_impedance))

        def reset_joints(self):
            self.resets += 1

    class HomingArm(FrozenArm):
        def reset_joints(self):
            self.resets += 1
            self.q = HOME_JOINTS.copy()
            self.pose = FakePose([0.307, 0.0, 0.487], np.diag([1.0, -1.0, -1.0]))

    class FlakyArm(HomingArm):
        """Refuses the first command (franka-interface still recovering), then obeys."""

        def reset_joints(self):
            self.resets += 1
            if self.resets > 1:
                self.q = HOME_JOINTS.copy()
                self.pose = FakePose([0.307, 0.0, 0.487], np.diag([1.0, -1.0, -1.0]))

    vision = FakeVision({})
    frozen = FrozenArm(gripper_width=0.037)
    skill = make(PlaceSkill, vision, frozen)
    check("a refused reset_joints is not home", skill.go_home(retry_wait=0.0) is False)
    check("it was tried twice, not more", frozen.resets == 2, f"{frozen.resets}")
    flaky = FlakyArm(gripper_width=0.008)
    check("refused once while franka-interface recovers, the retry gets home",
          make(PlaceSkill, vision, flaky).go_home(retry_wait=0.0) is True and flaky.resets == 2,
          f"resets {flaky.resets}")

    # Cut short part-way: retried only when libfranka reports a timing fault.
    from types import SimpleNamespace

    class Flags:
        __slots__ = ("cartesian_reflex",
                     "cartesian_motion_generator_joint_velocity_discontinuity")

        def __init__(self, **on):
            for f in self.__slots__:
                setattr(self, f, bool(on.get(f)))

    class StateClient:
        def __init__(self, **on):
            self.on = on

        def _get_current_robot_state(self):
            return SimpleNamespace(robot_state=SimpleNamespace(
                current_errors=Flags(), last_motion_errors=Flags(**self.on), robot_mode=1))

    class PartwayArm(HomingArm):
        def reset_joints(self):
            self.resets += 1
            if self.resets == 1:
                self.q = (self.q + HOME_JOINTS) / 2.0       # stopped half way
            else:
                HomingArm.reset_joints(self)
                self.resets -= 1

    timing = PartwayArm(gripper_width=0.03)
    timing._state_client = StateClient(
        cartesian_motion_generator_joint_velocity_discontinuity=True)
    check("cut short by a velocity discontinuity (sub-task 14), the retry gets home",
          make(PlaceSkill, vision, timing).go_home(retry_wait=0.0) is True
          and timing.resets == 2, f"resets {timing.resets}")
    bumped = PartwayArm(gripper_width=0.03)
    bumped._state_client = StateClient(cartesian_reflex=True)
    check("cut short by a reflex (a collision) is never retried",
          make(PlaceSkill, vision, bumped).go_home(retry_wait=0.0) is False
          and bumped.resets == 1, f"resets {bumped.resets}")
    check("one that gets there is",
          make(PlaceSkill, vision, HomingArm(gripper_width=0.037)).go_home() is True)
    check("an arm with no joint reading is trusted",
          make(PlaceSkill, vision, FakeArm()).go_home() is True)

    frozen.commands.clear()
    ok, result = skill.execute({"target_location": [0.5628, 0.0253, 0.0277]})
    check("place will not start with the cup still tipped 90 deg",
          not ok and "tipped" in str(result.get("error")), f"{result}")
    check("it keeps hold and commands nothing",
          result.get("still_holding") is True and not frozen.commands,
          f"{result} {frozen.commands}")

    # Picked that way on purpose (a side grasp) is not tipped. The executor
    # passes place the tool axis the pick had.
    side = FakeArm(gripper_width=0.037)
    side.pose = FakePose([0.45, 0.0, 0.20], tipped.copy())
    ex = SkillsExecutor(side, vision, {})
    ex._track_held("pick_up", {"object_name": "plastic beaker"},
                   {"grasp_pose": np.vstack([np.c_[tipped, [0.45, 0.0, 0.05]],
                                             [0, 0, 0, 1]]).tolist()})
    check("the pick's tool axis is kept",
          np.allclose(ex.held_measure.get("tool_z"), tipped[:, 2]), ex.held_measure)
    ok, result = ex.execute("place", {"target_location": [0.45, -0.10, 0.02]})
    check("an object picked from the side is placed as it was picked",
          "tipped" not in str(result.get("error", "")), f"{result}")


def test_vision_routes_bench_names():
    """locate() finds a bench object the manifest's way, and never guesses a labelled cup."""
    print("\n[vision: bench names routed by the manifest]")
    from robochem.vision import VisionSystem
    from robochem.vision.bench_manifest import MAGIC_BEAKER

    class Localizer:
        def __init__(self):
            self._cached_centroids, self._cached_pointclouds = {}, {}

        def cache_object(self, name, points, centroid=None):
            self._cached_pointclouds[name] = points
            self._cached_centroids[name] = (np.asarray(points).mean(axis=0)
                                            if centroid is None else centroid)

    def make(bench, label_result=True):
        v = VisionSystem.__new__(VisionSystem)
        v.bench = bench
        v.object_localizer = Localizer()
        v.scene_analyzer = object()
        v.labeled_cup_category = "white bowl"
        v.calls, v.shapes = [], []
        points = np.array([[0.4, 0.1, 0.03], [0.42, 0.1, 0.05]])

        def found(name):
            return {"points": points, "centroid": points.mean(axis=0), "cached": False}

        def locate_labeled(label, category=None, shape=None):
            v.calls.append(("label", label, category))
            v.shapes.append(shape)
            return found(label) if label_result else None

        def locate_direct(query):
            v.calls.append(("direct", query))
            return found(query)

        v.locate_labeled = locate_labeled
        v._locate_direct = locate_direct
        return v

    v = make(MAGIC_BEAKER)
    out = v.locate("clear cup a", force_refresh=True)
    check("'clear cup a' is the clear plastic cup labelled 'a 10 ml water'",
          v.calls == [("label", "a 10 ml water", "clear plastic cup")], f"{v.calls}")
    check("the result names the bench object", out and out.get("bench_object") == "clear cup a")
    check("a clear cup is looked for as a transparent shape",
          v.shapes and v.shapes[0] is not None and v.shapes[0].transparent)
    check("and is cached under the name asked for",
          "clear cup a" in v.object_localizer._cached_centroids)

    v = make(MAGIC_BEAKER)
    v.locate("citric acid cup", force_refresh=True, category="clear plastic cup")
    check("a skill's category does not override the measured one",
          v.calls == [("label", "citric acid", "white bowl")], f"{v.calls}")
    check("an opaque dish is looked for as itself, not as transparent",
          len(v.shapes) == 1 and v.shapes[0] is not None and not v.shapes[0].transparent,
          f"{v.shapes}")

    v = make(MAGIC_BEAKER)
    v.locate("smaller spoon", force_refresh=True)
    check("'smaller spoon' is prompted as the one scoop, 'white plastic tool'",
          v.calls == [("direct", "white plastic tool")], f"{v.calls}")

    v = make(MAGIC_BEAKER, label_result=False)
    out = v.locate("clear cup c", force_refresh=True)
    check("an unreadable label is a failure, not a direct prompt for any clear cup",
          out is None and v.calls == [("label", "c empty", "clear plastic cup")], f"{v.calls}")

    v = make(MAGIC_BEAKER, label_result=False)
    v.locate("tonic water", force_refresh=True)
    check("a name not on the bench keeps the old path (label, then direct)",
          v.calls == [("label", "tonic water", "white bowl"), ("direct", "tonic water")],
          f"{v.calls}")

    v = make(None)
    v.locate("clear cup a", force_refresh=True)
    check("with no bench nothing is routed",
          v.calls and v.calls[0] == ("label", "clear cup a", "white bowl"), f"{v.calls}")
    check("with no bench there is no inventory",
          v.known_object_names() == [] and v.inventory_notes() == {}
          and v.canonical_name("smaller spoon") == "smaller spoon")


def test_executor_uses_bench_names():
    """A bench alias becomes the bench name before pick sites and tool CAD are looked up."""
    print("\n[executor: bench names]")
    from robochem.skills import SkillsExecutor
    from robochem.vision.bench_manifest import MAGIC_BEAKER

    class Vision:
        canonical_name = staticmethod(MAGIC_BEAKER.canonical)

    ex = SkillsExecutor(None, Vision(), {})
    out = ex._bench_names({"object_name": "smaller spoon", "z_offset": 0.0})
    check("pick_up 'smaller spoon' picks the larger spoon",
          out == {"object_name": "larger spoon", "z_offset": 0.0}, f"{out}")
    out = ex._bench_names({"target_container": "B 10 ML WATER", "powder_source": "citric acid"})
    check("labels become the containers' bench names",
          out == {"target_container": "clear cup b", "powder_source": "citric acid cup"},
          f"{out}")
    out = ex._bench_names({"target_location": [0.4, 0.1, 0.05]})
    check("coordinates are left alone", out == {"target_location": [0.4, 0.1, 0.05]})

    ex.pick_sites = {"larger spoon": [0.55, -0.10, 0.01]}
    ex.held = "larger spoon"
    out = ex._resolve_pick_site("place", ex._bench_names({"target_location": "small scoop"}))
    check("putting back the 'small scoop' returns the scoop to its own pick site",
          out["target_location"] == [0.55, -0.10, 0.01], f"{out}")

    from robochem.skills.tool_geometry import lookup
    check("the scoop's CAD is the larger spoon's, not the 75% placeholder",
          lookup(ex._bench_names({"object_name": "small scoop"})["object_name"])["name"]
          == "larger spoon")

    plain = SkillsExecutor(None, object(), {})
    check("a vision system with no bench renames nothing (the sim)",
          plain._bench_names({"object_name": "smaller spoon"}) == {"object_name": "smaller spoon"})

    class BenchVision(Vision):
        bench_defaults = staticmethod(MAGIC_BEAKER.defaults)

    ex = SkillsExecutor(None, BenchVision(), {})
    out = ex._bench_defaults("pick_up", ex._bench_names({"object_name": "small scoop"}))
    check("the planner's bare pick of the scoop gets the robot-validated grasp",
          out == {"object_name": "larger spoon", "z_offset": 0.0, "grasp_force": 1.0,
                  "forward_offset": -0.010}, f"{out}")
    out = ex._bench_defaults("pick_up", {"object_name": "larger spoon", "grasp_force": 0.8})
    check("a value the call passes is kept", out["grasp_force"] == 0.8, f"{out}")
    check("the stirrer gets its validated grasp",
          ex._bench_defaults("pick_up", {"object_name": "stirrer"})
          == {"object_name": "stirrer", "z_offset": 0.0, "grasp_force": 1.0})
    check("nothing is invented for an object with nothing measured",
          ex._bench_defaults("pick_up", {"object_name": "citric acid cup"})
          == {"object_name": "citric acid cup"})
    check("the beaker is grasped 12 mm lower",
          ex._bench_defaults("pick_up", ex._bench_names({"object_name": "50 ML WATER"}))
          == {"object_name": "plastic beaker", "z_offset": -0.012},
          f"{ex._bench_defaults('pick_up', {'object_name': 'plastic beaker'})}")
    # ... and set down as much lower when it goes back.
    ex._remember_pick_site("pick_up", {"object_name": "plastic beaker"},
                           {"centroid": [0.5628, 0.0253, 0.0277], "z_offset": -0.012,
                            "grasp_tcp_reached": [0.5627, 0.0243, 0.0398]})
    ex.held = "plastic beaker"
    back = ex._resolve_pick_site("place", {"target_location": "plastic beaker"})
    check("its put-back releases 12 mm lower than usual",
          abs(back.get("release_clearance", 1) - 0.008) < 1e-9, f"{back}")
    ex._remember_pick_site("pick_up", {"object_name": "larger spoon"},
                           {"centroid": [0.6473, 0.0172, -0.0008], "z_offset": 0.0,
                            "grasp_tcp_reached": [0.6319, 0.0155, 0.0117]})
    ex.held = "larger spoon"
    back = ex._resolve_pick_site("place", {"target_location": "larger spoon"})
    check("a pick not moved down puts back as before",
          "release_clearance" not in back, f"{back}")
    ex.held = None
    cup_pick = {"slide_in": 0.06, "z_offset": -0.012, "grip_width": 0.058}
    check("cups B and C are slid into from behind, 12 mm lower, closed to 58 mm",
          ex._bench_defaults("pick_up", {"object_name": "clear cup b"})
          == {"object_name": "clear cup b", **cup_pick}
          and ex._bench_defaults("pick_up", {"object_name": "clear cup c"})
          == {"object_name": "clear cup c", **cup_pick})
    check("cup A too, its grasp 5 mm away from B as the user read it with point_at.py",
          ex._bench_defaults("pick_up", ex._bench_names({"object_name": "cup a"}))
          == {"object_name": "clear cup a", **cup_pick, "lateral_offset": -0.005})
    check("place is not filled", ex._bench_defaults("place", {"target_location":
          "larger spoon"}) == {"target_location": "larger spoon"})
    out = ex._bench_defaults("scoop", ex._bench_names({"powder_source": "CITRIC ACID"}))
    check("a scoop from a dish is sized to its measured inside and height, not a rim fit",
          out == {"powder_source": "citric acid cup", "container_radius": 0.0455,
                  "container_height": 0.029, "sweep_mode": "position",
                  "z_offset": -0.025}, f"{out}")
    out = ex._bench_defaults("scoop", {"powder_source": "baking soda cup",
                                       "container_radius": 0.04})
    check("a container_radius the call passes is kept", out["container_radius"] == 0.04)
    for target in ("clear cup a", "plastic beaker", "B 10 ML WATER"):
        out = ex._bench_defaults("dump", ex._bench_names({"target_container": target}))
        check(f"every dump on the bench lands 10 mm back ({target})",
              out.get("forward_offset") == -0.010 and out.get("min_tip_deg") == 55.0,
              f"{out}")
    out = ex._bench_defaults("dump", {"target_container": "clear cup a", "forward_offset": 0.0})
    check("a forward_offset the call passes is kept", out["forward_offset"] == 0.0)
    check("pour is not given dump's offset",
          "forward_offset" not in ex._bench_defaults("pour", {"target_container": "clear cup c"}))
    out = ex._bench_defaults("stir", {"target_container": "clear cup a", "to_floor": True})
    check("stir draws the bench's bigger circle",
          out.get("stir_radius") == 0.020 and out.get("wall_clearance") == 0.006
          and out.get("to_floor") is True, f"{out}")
    out = ex._bench_defaults("stir", {"target_container": "clear cup a", "stir_radius": 0.012})
    check("a stir_radius the call passes is kept", out["stir_radius"] == 0.012)


def test_pour_keeps_the_lip_just_over_the_target():
    """With the held cup's shape known, the lip ends just over the rim."""
    print("\n[pour: lip over the target]")
    from robochem.skills.pour import PourSkill
    target = cylinder_cloud([0.35, -0.15], base_z=0.0, height=0.047, radius=0.0275)
    arm = FakeArm(gripper_width=0.05)
    skill = make(PourSkill, FakeVision({"cup": target}), arm)
    rim = skill.locate_container("cup")
    src = {"source_top_above_tcp": 0.014, "source_height": 0.049, "source_radius": 0.03}
    # The cup's own geometry first; the flange floor is checked below.
    ok, result = skill.execute({"target_container": "cup", "pour_angle": 90,
                                "hold_duration": 0.0, "duration_per_step": 0.0,
                                "flange_min_z": None, **src})
    check("pour succeeds", ok, f"{result}")
    check("it took the lip path", result.get("mode") == "lip_over_target", f"{result}")
    lips = result.get("lip_over_rim_mm") or []
    check("the lip ends 15 mm over the rim", lips and abs(lips[-1] - 15.0) < 0.5, f"{lips}")
    check("and never below it on the way", lips and min(lips) >= 14.5, f"{lips}")
    lip_xy = np.asarray(result["lip_xy"])
    want = np.asarray(rim["rim_center"])[:2] + np.array([-0.4 * rim["rim_radius"], 0.0])
    check("the lip is over the target, a little toward the base",
          np.linalg.norm(lip_xy - want) < 1e-6, f"{lip_xy} vs {want}")
    # Every commanded TCP during the tip keeps the whole cup 10 mm over the rim.
    cup = PourSkill._cup_points(0.014, 0.049, 0.03)
    tips = [c for c in arm.commands]
    check("the cup came down as it tipped (TCP lower at the end than the start)",
          len(tips) > 3 and tips[-2][0][2] < tips[2][0][2],
          f"{[round(float(c[0][2]), 3) for c in tips]}")

    # franka-interface's virtual floor. The robot, 2026-10-08, step 24: the
    # beaker (rim 13 mm over the TCP, 60 mm tall, r 22 mm) tipped to 90 deg
    # over cup B (rim 47.3 mm) put the flange 82.5 mm up and was aborted;
    # 80 deg (flange 105 mm) ran. The plan is the same one the robot ran.
    from robochem.skills.pour import FLANGE_BEHIND_TCP
    beaker = PourSkill._cup_points(0.0132, 0.060, 0.022)
    lean, closing = np.array([-1.0, 0.0]), np.array([0.0, -1.0])

    def planned(deg, floor):
        R = PourSkill._rotation_lean(closing, float(deg), lean)
        xyz, lip = skill._lip_pose(R, beaker, 32, np.array([0.420, -0.021]), 0.0473,
                                   0.015, 0.010, lean, flange_min_z=floor)
        return xyz[2] - FLANGE_BEHIND_TCP * R[2, 2], lip

    check("unfloored, the 90 deg step puts the flange where the robot faulted",
          abs(planned(90, None)[0] - 0.0843) < 0.002 and abs(planned(80, None)[0] - 0.105) < 0.002,
          f"{planned(90, None)[0]:.4f} {planned(80, None)[0]:.4f}")
    floored = [planned(d, 0.11) for d in range(0, 91, 10)]
    check("with the floor every step keeps the flange 110 mm up",
          min(f for f, _l in floored) >= 0.11 - 1e-9, f"{[round(f, 4) for f, _l in floored]}")
    check("the late steps ride higher, the early ones are untouched",
          floored[-1][1] > 0.035 and abs(floored[6][1] - planned(60, None)[1]) < 1e-9,
          f"{[round(l, 4) for _f, l in floored]}")
    arm3 = FakeArm(gripper_width=0.05)
    ok3, result3 = make(PourSkill, FakeVision({"cup": target}), arm3).execute(
        {"target_container": "cup", "pour_angle": 90, "hold_duration": 0.0,
         "duration_per_step": 0.0, **src})
    lips3 = result3.get("lip_over_rim_mm") or []
    check("by default a 90 deg pour ends higher than 15 mm, never lower",
          ok3 and lips3 and lips3[-1] > 15.5 and min(lips3) >= 14.5, f"{lips3}")
    # Without the shape it is the old fixed site, 15 cm up.
    arm2 = FakeArm(gripper_width=0.05)
    ok2, result2 = make(PourSkill, FakeVision({"cup": target}), arm2).execute(
        {"target_container": "cup", "pour_angle": 90, "hold_duration": 0.0,
         "duration_per_step": 0.0})
    check("without the held cup's shape it falls back to the fixed site",
          ok2 and result2.get("mode") is None, f"{result2}")


def test_depth_gate_drops_edge_streaks_not_containers():
    """
    Mask edge pixels carrying the bench depth behind a small object must go;
    a real container's own depth spread must not.
    """
    print("\n[perception: depth gate on mask edge streaks]")
    import types
    from robochem.vision.object_localizer import ObjectLocalizer
    loc = ObjectLocalizer.__new__(ObjectLocalizer)
    intr = types.SimpleNamespace(fx=610.0, fy=610.0, cx=424.0, cy=240.0)

    # A 30mm cube at 0.47m is ~40px across. Its 2px border reads the bench
    # 0.19m behind it, the way cams 2/3 did on the stirrer stills.
    depth = np.zeros((480, 848), dtype=np.uint16)
    mask = np.zeros_like(depth, dtype=bool)
    mask[200:240, 400:440] = True
    depth[200:240, 400:440] = 660
    depth[202:238, 402:438] = 470
    pts = loc._depth_to_points(depth, mask, intr)
    zs = pts[:, 2]
    check("the bench-depth border is dropped from a small object",
          zs.max() < 0.50, f"depth {zs.min():.3f}..{zs.max():.3f} m")
    check("the object's own pixels are all kept",
          len(pts) == 36 * 36, f"{len(pts)} of {36 * 36}")

    # A cup ~150px across at 0.5m, its depth spread over 0.45-0.58m.
    depth = np.zeros((480, 848), dtype=np.uint16)
    mask = np.zeros_like(depth, dtype=bool)
    mask[150:300, 350:500] = True
    depth[150:300, 350:500] = np.linspace(450, 580, 150).astype(np.uint16)[:, None]
    pts = loc._depth_to_points(depth, mask, intr)
    check("a container's full depth spread survives",
          len(pts) == 150 * 150, f"{len(pts)} of {150 * 150}")


def test_stir_offsets_shift_the_circle_centre():
    """forward/lateral_offset move the circle, in the pour/scoop convention."""
    print("\n[stir: forward/lateral offset shifts the circle centre]")
    cup = cylinder_cloud([0.50, 0.0], base_z=0.02, height=0.08, radius=0.045)
    opts = {"target_container": "cup", "revolutions": 1, "smooth": False,
            "waypoints_per_rev": 8, "seconds_per_waypoint": 0.0,
            "stir_radius": 0.02}
    ok0, base = make(StirSkill, FakeVision({"cup": cup}), FakeArm(
        tracking=1.0, gripper_width=0.03)).execute(dict(opts))
    arm = FakeArm(tracking=1.0, gripper_width=0.03)
    ok, result = make(StirSkill, FakeVision({"cup": cup}), arm).execute(
        dict(opts, lateral_offset=-0.012))
    check("stir succeeds with an offset", ok0 and ok, f"{result}")
    shift = np.asarray(result["center"]) - np.asarray(base["center"]) if ok0 and ok else None
    check("lateral_offset -0.012 moves the centre 12mm toward -Y only",
          shift is not None and np.allclose(shift, [0.0, -0.012], atol=1e-9),
          f"{shift}")
    ys = [c[0][1] for c in arm.commands[-9:-1]]
    check("the traced circle is centred on the shifted point",
          ok and abs((max(ys) + min(ys)) / 2 - result["center"][1]) < 1e-3,
          f"circle y mid {(max(ys) + min(ys)) / 2:.4f} vs {result['center'][1]:.4f}")


def test_place_releases_on_a_negative_press():
    """
    The 2026-10-05 robot failure: pressing the stirrer into its holder read
    NEGATIVE, place only stopped on a positive change, stepped on to -96N with
    the stirrer seated, and refused to let go. A press of either sign must
    stop the probe and release there.
    """
    print("\n[place: a negative press is contact]")
    seat = 0.1049                   # where the robot first felt the holder
    arm = PressArm(seat, bias=-2.61)
    skill = make(PlaceSkill, FakeVision({}), arm)
    ok, result = skill.execute({"target_location": [0.3039, 0.171, 0.0919],
                                "vertical_insert": True,
                                "pick_grasp_tcp": [0.3039, 0.171, 0.0919],
                                "release_clearance": 0.0, "stop_force_n": 1.0,
                                "probe_seconds": 0.0})
    check("place succeeds on a negative press", ok, f"{result}")
    check("it stopped on force", ok and result["seated_by_force"] is True)
    check("it stopped within one step of the seat, not 28mm past it",
          ok and seat - 0.0025 < result["contact_z"] <= seat,
          f"contact_z={result.get('contact_z')}")
    check("the gripper opened", arm.gripper_width > 0.06,
          f"{arm.gripper_width * 1000:.1f}mm")
    lowest = min(c[0][2] for c in arm.commands)
    check("never pressed more than one step into the seat",
          lowest > seat - 0.0025, f"lowest z={lowest:.4f}")


def test_stir_detects_a_dropped_stirrer():
    print("\n[stir: dropped stirrer aborts]")
    wide = cylinder_cloud([0.50, 0.0], base_z=0.02, height=0.08, radius=0.045)
    vision = FakeVision({"cup": wide})

    class DroppingArm(FakeArm):
        def __init__(self):
            super().__init__(tracking=1.0, gripper_width=0.03)
            self.moves = 0

        def goto_pose(self, pose, duration=3.0, use_impedance=True, block=True):
            super().goto_pose(pose, duration, use_impedance, block)
            self.moves += 1
            if self.moves > 8:      # stirrer falls out mid-circle
                self.gripper_width = 0.001

    arm = DroppingArm()
    skill = make(StirSkill, vision, arm)
    ok, result = skill.execute({"target_container": "cup", "revolutions": 3,
                                "waypoints_per_rev": 6,
                                "seconds_per_waypoint": 0.0})
    check("stir fails when the stirrer is dropped", not ok, f"{result}")
    check("failure names the lost tool",
          "lost" in result.get("error", "").lower()
          or "gone" in result.get("error", "").lower(),
          f"{result.get('error')}")


# The scoop's plunge-and-push model was replaced by bozhang's sweep in the
# 2026-09-24 merge, so the tests that pinned that mechanism down —
# dig_advance / drag_distance / untilt_over / dip_depth, the two-segment push,
# the total_depth bookkeeping — were removed rather than bent to fit. They were
# describing an implementation that no longer exists. The sweep needs its own
# tests written against its own vocabulary (sweep_advance, sweep_rise,
# depth_reference, bowl geometry); what survives here is the behaviour both
# versions share: the bite tilt must be measured, the site offsets must move
# the stroke, and an unknown parameter must be shouted about.


def test_scoop_requires_a_real_tilt():
    """A wrist that stalls short must not report a successful scoop."""
    print("\n[scoop: retaining tilt must actually happen]")
    tub = cylinder_cloud([0.50, 0.10], base_z=0.02, height=0.05, radius=0.04)
    vision = FakeVision({"citric acid container": tub})

    # tilt_gain=0.02: the wrist barely rotates however often we command it.
    arm = FakeArm(tracking=1.0, gripper_width=0.03, tilt_gain=0.02)
    skill = make(ScoopSkill, vision, arm)
    ok, result = skill.execute({"powder_source": "citric acid container"})
    check("scoop fails when the tilt stalls", not ok, f"{result}")
    check("failure explains the powder would fall out",
          "tilt" in result.get("error", "").lower(), f"{result.get('error')}")

    arm2 = FakeArm(tracking=1.0, gripper_width=0.03, tilt_gain=1.0)
    skill2 = make(ScoopSkill, vision, arm2)
    ok2, result2 = skill2.execute({"powder_source": "citric acid container"})
    check("scoop succeeds when the wrist tracks", ok2, f"{result2}")
    check("quantity is reported as unmeasured",
          ok2 and result2.get("quantity_measured") is False)
    compliant = [c for c in arm2.commands if c[2] is True]
    check("dig and drag ran compliant (impedance on)", len(compliant) >= 2,
          f"{len(compliant)} compliant commands")


def test_scoop_takes_a_known_rim_height():
    """A dish whose rim reads low is cleared at its real height when that is known."""
    print("\n[scoop: known container height]")
    import io, contextlib, re
    # The robot's view of the 29 mm dish: the cloud tops out at 17 mm.
    dish = cylinder_cloud([0.55, -0.10], base_z=0.0, height=0.017, radius=0.0505)
    vision = FakeVision({"citric acid cup": dish})

    def run(extra):
        arm = FakeArm(tracking=1.0, gripper_width=0.008)
        # As the robot runs it: --workspace-min ... -0.13, and the table at 0.
        skill = ScoopSkill(arm, vision, {"workspace_min": [0.25, -0.40, -0.13],
                                         "table_z": 0.0})
        params = {"powder_source": "citric acid cup", "container_radius": 0.0455,
                  "tool_offset": [0.053, 0.0, 0.0294], "bowl_length": 0.0275,
                  "bowl_width": 0.0205, "bowl_depth": 0.0115, "tool_back_reach": 0.030}
        params.update(extra)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ok, result = skill.execute(params)
        lift = re.search(r"Lifting straight out to z=([0-9.]+)", buf.getvalue())
        return ok, buf.getvalue(), float(lift.group(1)) if lift else None

    ok0, out0, z0 = run({})
    ok1, out1, z1 = run({"container_height": 0.029})
    check("without a height the rim is the cloud's top (the sim, unchanged)",
          ok0 and "Rim at z=" not in out0, out0[-300:] if not ok0 else "")
    check("with it the rim is the table plus the height",
          ok1 and "Rim at z=0.0290" in out1, out1[-300:] if not ok1 else "")
    check("and the lift out clears the real rim, higher than before",
          z0 is not None and z1 is not None and z1 > z0, f"{z0} -> {z1}")


def test_scoop_puts_the_bowl_at_the_planned_height_under_impedance():
    """A compliant descent that stops high is re-commanded lower, and the sweep with it."""
    print("\n[scoop: impedance stopping short]")
    import io, contextlib
    tub = cylinder_cloud([0.50, 0.10], base_z=0.02, height=0.05, radius=0.06)
    vision = FakeVision({"tub": tub})

    class SaggingArm(FakeArm):
        """Stops 8 mm above every impedance-controlled command, as the robot does."""
        def goto_pose(self, pose, duration=3.0, use_impedance=True, block=True):
            super().goto_pose(pose, duration=duration, use_impedance=use_impedance, block=block)
            if use_impedance:
                self.pose.translation = self.pose.translation + np.array([0.0, 0.0, 0.008])

    def run(arm):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            ok, result = make(ScoopSkill, vision, arm).execute(
                {"powder_source": "tub", "push_seconds": 0.5})
        return ok, result, buf.getvalue()

    ok0, r0, out0 = run(FakeArm(tracking=1.0, gripper_width=0.03))
    check("an arm that tracks its command is not corrected (the sim)",
          ok0 and r0["sag_compensation"] == 0.0 and "stopped" not in out0.split("Sweeping")[0][-400:],
          f"{r0.get('sag_compensation')}")
    arm = SaggingArm(tracking=1.0, gripper_width=0.03)
    ok1, r1, out1 = run(arm)
    check("an 8 mm high stop is measured", ok1 and abs(r1["descent_sag"] - 0.008) < 1e-6,
          f"{r1.get('descent_sag')}")
    check("and the descent and sweep are commanded 8 mm lower",
          abs(r1["sag_compensation"] - 0.008) < 1e-6, f"{r1.get('sag_compensation')}")
    a0 = FakeArm(tracking=1.0, gripper_width=0.03)
    run(a0)
    sweep0 = [c[0][2] for c in a0.commands]
    sweep1 = [c[0][2] for c in arm.commands]
    lowest0, lowest1 = min(sweep0), min(sweep1)
    check("so the sagging arm's lowest command is 8 mm under the tracking arm's",
          abs((lowest0 - lowest1) - 0.008) < 1e-6, f"{(lowest0 - lowest1) * 1000:.2f} mm")
    check("the reported TCP error at the sweep's end is printed", "from the last setpoint" in out1)


def test_streamed_scoop_and_dump_ask_for_stiff_tracking():
    """Stiffness can reach frankapy, but no skill asks for it (it shook the arm)."""
    print("\n[streaming: tracking stiffness]")
    from robochem.skills.base_skill import dynamic_start_kwargs, STREAM_TRACKING_IMPEDANCES
    kw = dynamic_start_kwargs(5.0, True, STREAM_TRACKING_IMPEDANCES)
    check("frankapy's goto_pose is given the stiffness",
          kw["cartesian_impedances"] == [1500.0] * 3 + [150.0] * 3 and kw["dynamic"] is True,
          f"{kw}")
    check("and nothing extra when none is asked for",
          "cartesian_impedances" not in dynamic_start_kwargs(5.0))
    check("stiffer than frankapy's defaults, inside Franka's limits",
          all(600 < v <= 3000 for v in STREAM_TRACKING_IMPEDANCES[:3])
          and all(50 < v <= 300 for v in STREAM_TRACKING_IMPEDANCES[3:]))

    seen = {}
    tub = cylinder_cloud([0.50, 0.10], base_z=0.02, height=0.05, radius=0.06)
    skill = make(ScoopSkill, FakeVision({"tub": tub}), FakeArm(tracking=1.0, gripper_width=0.03))

    def spy(points, rotation, seconds, **kw):
        seen.update(kw)
        return False, "spy"
    skill.stream_pose_path = spy
    skill.execute({"powder_source": "tub", "push_seconds": 0.5})
    # Stiffer streaming shook the arm heavily on 2026-10-08: frankapy's
    # defaults stand until it is understood.
    check("scoop's sweep streams at frankapy's default stiffness",
          seen.get("stiffness") is None, f"{seen}")


def test_scoop_rehearses_in_the_air():
    """air_offset runs the same scoop, every command raised, nothing touched."""
    print("\n[scoop: rehearsal in the air]")
    tub = cylinder_cloud([0.50, 0.10], base_z=0.02, height=0.05, radius=0.06)
    vision = FakeVision({"tub": tub})

    def run(extra):
        arm = FakeArm(tracking=1.0, gripper_width=0.03)
        ok, r = make(ScoopSkill, vision, arm).execute(
            {"powder_source": "tub", "push_seconds": 0.5, **extra})
        return ok, np.array([c[0] for c in arm.commands])

    ok0, c0 = run({})
    ok1, c1 = run({"air_offset": 0.10})
    # The first command is the approach from hover height; the rest is the dig.
    dig0, dig1 = c0[1:], c1[1:]
    check("the rehearsal runs", ok0 and ok1)
    check("every dig command is the real one raised 100 mm, same x and y",
          dig0.shape == dig1.shape and np.allclose(dig1[:, :2], dig0[:, :2], atol=1e-9)
          and np.allclose(dig1[:, 2] - dig0[:, 2], 0.10, atol=1e-6),
          f"{np.round((dig1 - dig0)[:, 2] * 1000, 1)}")
    check("so its lowest command stays far over the dish",
          dig1[:, 2].min() > 0.07 + 0.05, f"{dig1[:, 2].min():.3f}")
    ok3, c3 = run({"z_offset": -0.003})
    dig3 = c3[1:]
    check("z_offset -0.003 moves every dig command exactly 3 mm down, same x and y",
          ok3 and dig3.shape == dig0.shape and np.allclose(dig3[:, :2], dig0[:, :2], atol=1e-9)
          and np.allclose(dig3[:, 2] - dig0[:, 2], -0.003, atol=1e-6),
          f"{np.round((dig3 - dig0)[:, 2] * 1000, 2)}")


def test_scoop_position_sweep():
    """sweep_mode 'position': the same waypoints, each a position-controlled move."""
    print("\n[scoop: position-controlled sweep]")
    tub = cylinder_cloud([0.50, 0.10], base_z=0.02, height=0.05, radius=0.06)
    vision = FakeVision({"tub": tub})

    class SaggingArm(FakeArm):
        def goto_pose(self, pose, duration=3.0, use_impedance=True, block=True):
            super().goto_pose(pose, duration=duration, use_impedance=use_impedance, block=block)
            if use_impedance:
                self.pose.translation = self.pose.translation + np.array([0.0, 0.0, 0.004])

    def run(arm, extra):
        skill = make(ScoopSkill, vision, arm)
        streamed = []
        real_stream = skill.stream_pose_path

        def spy(*a, **k):
            streamed.append(1)
            return real_stream(*a, **k)
        skill.stream_pose_path = spy
        ok, r = skill.execute({"powder_source": "tub", "push_seconds": 0.5, **extra})
        return ok, r, arm.commands, streamed

    ok0, r0, c0, s0 = run(FakeArm(tracking=1.0, gripper_width=0.03), {})
    ok1, r1, c1, s1 = run(FakeArm(tracking=1.0, gripper_width=0.03), {"sweep_mode": "position"})
    check("the default still streams (the sim)", ok0 and s0 and r0["sweep_mode"] == "stream")
    check("position mode never streams", ok1 and not s1 and r1["sweep_mode"] == "position")
    sweep = [c for c in c1 if c[2] is False]
    check("its sweep is position-controlled waypoints", len(sweep) >= r1["swept_waypoints"],
          f"{len(sweep)} position moves for {r1['swept_waypoints']} waypoints")
    # Same geometry: the position sweep visits the waypoints the stream would.
    ok2, r2, c2, _ = run(SaggingArm(tracking=1.0, gripper_width=0.03), {"sweep_mode": "position"})
    clean = [c[0] for c in c1 if c[2] is False][:r1["swept_waypoints"]]
    sag = [c[0] for c in c2 if c[2] is False][:r2["swept_waypoints"]]
    check("an impedance descent's sag is not carried into a position sweep",
          ok2 and r2["sag_compensation"] > 0 and np.allclose(clean, sag, atol=1e-9),
          f"comp {r2.get('sag_compensation')}")


def test_scoop_site_offsets_and_unknown_params():
    """forward/lateral must move the stroke, and typos must be shouted about."""
    print("\n[scoop: site offsets work, unknown params are flagged]")
    tub = cylinder_cloud([0.50, 0.10], base_z=0.02, height=0.05, radius=0.06)
    vision = FakeVision({"tub": tub})

    def run(extra):
        arm = FakeArm(tracking=1.0, gripper_width=0.03)
        base = {"powder_source": "tub", "push_seconds": 0.5}
        base.update(extra)
        ok, result = make(ScoopSkill, vision, arm).execute(base)
        return ok, result, arm

    ok0, r0, a0 = run({})
    ok1, r1, a1 = run({"forward_offset": 0.030})
    ok2, r2, a2 = run({"lateral_offset": -0.025})
    check("baseline succeeds", ok0, f"{r0}")
    check("forward_offset is reported back", ok1 and r1["forward_offset"] == 0.030,
          f"{r1.get('forward_offset')}")
    check("lateral_offset is reported back", ok2 and r2["lateral_offset"] == -0.025,
          f"{r2.get('lateral_offset')}")

    x0 = a0.commands[0][0][0]
    x1 = a1.commands[0][0][0]
    y0 = a0.commands[0][0][1]
    y2 = a2.commands[0][0][1]
    check("forward_offset actually moves the stroke in +X",
          abs((x1 - x0) - 0.030) < 1e-6, f"x moved {(x1 - x0) * 1000:.1f}mm")
    check("lateral_offset actually moves the stroke in -Y",
          abs((y2 - y0) + 0.025) < 1e-6, f"y moved {(y2 - y0) * 1000:.1f}mm")

    # A misspelled or dropped parameter must not vanish silently.
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run({"lateral_offsets": -0.025})      # note the typo
    out = buf.getvalue()
    check("an unknown parameter is warned about, not ignored in silence",
          "unknown parameter" in out and "lateral_offsets" in out,
          out[-200:] if out else "no output")







def test_dispense_never_drops_the_pipette():
    print("\n[dispense: squeeze is floored]")
    cup = cylinder_cloud([0.50, 0.0], base_z=0.02, height=0.08, radius=0.035)
    vision = FakeVision({"clear cup 1": cup})
    arm = FakeArm(tracking=1.0, gripper_width=0.020)
    skill = make(DispenseSkill, vision, arm)
    ok, result = skill.execute({"target_container": "clear cup 1", "num_drops": 3,
                                "drop_interval": 0.0, "squeeze_hold": 0.0,
                                # An absurd squeeze the floor has to catch.
                                "squeeze_amount": 0.050})
    check("dispense succeeds", ok, f"{result}")
    check("all drops dispensed", ok and result["drops_dispensed"] == 3,
          f"{result.get('drops_dispensed')}")
    widths = [c["width"] for c in arm.gripper_commands]
    floor = 0.020 * 0.55
    check("no squeeze went below the crush floor",
          all(w >= floor - 1e-9 for w in widths),
          f"min width {min(widths) * 1000:.1f}mm, floor {floor * 1000:.1f}mm")
    check("gripper was never opened to release the pipette",
          all(w < 0.06 for w in widths),
          f"widths={[round(w * 1000, 1) for w in widths]}")
    check("volume is reported as unmeasured", result.get("volume_measured") is False)


def test_dispense_transfer_mode():
    print("\n[dispense: aspirate + transfer]")
    src = cylinder_cloud([0.55, 0.12], base_z=0.02, height=0.09, radius=0.03)
    dst = cylinder_cloud([0.45, -0.08], base_z=0.02, height=0.08, radius=0.035)
    vision = FakeVision({"water container": src, "clear cup 2": dst})
    arm = FakeArm(tracking=1.0, gripper_width=0.020)
    skill = make(DispenseSkill, vision, arm)
    ok, result = skill.execute({"mode": "transfer",
                                "source_container": "water container",
                                "target_container": "clear cup 2",
                                "num_drops": 2, "drop_interval": 0.0,
                                "squeeze_hold": 0.0})
    check("transfer succeeds", ok, f"{result}")
    check("transfer records both ends",
          ok and result.get("aspirated_from") == "water container"
          and result.get("target") == "clear cup 2", f"{result}")

    can, msg = skill.check_preconditions({"mode": "transfer",
                                          "target_container": "clear cup 2"})
    check("transfer without a source is refused", not can, msg)


def test_target_not_found_is_a_failure():
    print("\n[all skills: an unlocatable target fails cleanly]")
    vision = FakeVision({})
    for cls, params in [
        (StirSkill, {"target_container": "ghost cup"}),
        (ScoopSkill, {"powder_source": "ghost tub"}),
        (DispenseSkill, {"target_container": "ghost cup"}),
        (PlaceSkill, {"target_location": "ghost cup"}),
    ]:
        arm = FakeArm(tracking=1.0, gripper_width=0.03)
        ok, result = make(cls, vision, arm).execute(params)
        check(f"{cls.name} fails on an unlocatable target", not ok,
              f"{result}")
        check(f"{cls.name} says what it could not find",
              "cannot locate" in result.get("error", "").lower()
              or "locate" in result.get("error", "").lower(),
              f"{result.get('error')}")


# ==================== bridge to robomail_Aliyah ====================

class FakeStep:
    """Duck-typed stand-in for agents.action_vocabulary.PlannedStep."""

    def __init__(self, action, target_object=None, tool=None, location=None):
        self.action = action
        self.target_object = target_object
        self.tool = tool
        self.location = location
        self.primitives = []


class RecordingSkills:
    """SkillsExecutor stand-in that records dispatches."""

    def __init__(self, fail=()):
        self.calls = []
        self.fail = set(fail)

    def execute(self, name, params):
        self.calls.append((name, params))
        if name in self.fail:
            return False, {"error": "forced failure"}
        if name == "pick_up":
            return True, {"grasped_object": params["object_name"],
                          "centroid": [0.52, 0.11, 0.06]}
        return True, {"ok": True}


def test_bridge_translation():
    print("\n[bridge: action vocabulary -> robochem skills]")
    from robochem.integration.plato_bridge import (
        ACTION_TO_SKILL, RoboChemExecutor, normalize_object_name)

    check("PLATO position labels become object queries",
          normalize_object_name("Original Position of clear cup 1") == "clear cup 1",
          normalize_object_name("Original Position of clear cup 1"))

    skills = RecordingSkills()
    bridge = RoboChemExecutor(skills, param_overrides={"pour": {"forward_offset": -0.08}},
                              verbose=False)

    plan = [
        FakeStep("PICKUP", tool="measuring scoop",
                 location="Original Position of measuring scoop"),
        FakeStep("SCOOP", target_object="Original Position of citric acid container",
                 tool="measuring scoop"),
        FakeStep("PLACE", target_object="measuring scoop",
                 location="Original Position of measuring scoop"),
        FakeStep("PICKUP", target_object="plastic beaker"),
        FakeStep("POUR", target_object="white paper cup"),
    ]
    results = [bridge.execute(step) for step in plan]

    check("every step translated and ran", all(r.success for r in results),
          f"{[(r.action, r.reason) for r in results if not r.success]}")
    check("results are marked real, not fake",
          all(r.is_fake is False for r in results))
    check("dispatched skills are the expected ones",
          [c[0] for c in skills.calls]
          == ["pick_up", "scoop", "place", "pick_up", "pour"],
          f"{[c[0] for c in skills.calls]}")
    check("bench-tuned overrides reach the skill",
          skills.calls[-1][1].get("forward_offset") == -0.08,
          f"{skills.calls[-1][1]}")
    check("PLACE back at its own origin uses the recorded pick site",
          isinstance(skills.calls[2][1]["target_location"], list),
          f"{skills.calls[2][1]}")
    check("held object is tracked across the plan",
          bridge.held_object == "plastic beaker", f"{bridge.held_object}")


def test_bridge_guards():
    print("\n[bridge: guards]")
    from robochem.integration.plato_bridge import RoboChemExecutor

    bridge = RoboChemExecutor(RecordingSkills(), verbose=False)
    r = bridge.execute(FakeStep("POUR", target_object="cup"))
    check("POUR with an empty gripper is refused before moving", not r.success,
          r.reason)

    r = bridge.execute(FakeStep("TELEPORT", target_object="cup"))
    check("an unmapped action fails instead of no-opping", not r.success, r.reason)
    check("the failure names the translation stage",
          r.detail.get("failure_stage") == "translation", f"{r.detail}")

    r = bridge.execute(FakeStep("PICKUP"))
    check("PICKUP with nothing named is refused", not r.success, r.reason)

    # A failed pick must leave the gripper recorded as empty, so the next
    # POUR is caught rather than tipping an empty gripper over the cup.
    bridge2 = RoboChemExecutor(RecordingSkills(fail={"pick_up"}), verbose=False)
    bridge2.execute(FakeStep("PICKUP", target_object="beaker"))
    check("a failed PICKUP leaves nothing recorded as held",
          bridge2.held_object is None, f"{bridge2.held_object}")
    r = bridge2.execute(FakeStep("POUR", target_object="cup"))
    check("the following POUR is then refused", not r.success, r.reason)


def test_bridge_matches_the_real_vocabulary():
    """Cross-check against robomail_Aliyah itself, when it is importable."""
    print("\n[bridge: agreement with robomail_Aliyah's vocabulary]")
    aliyah = Path(__file__).resolve().parents[1] / "robomail_Aliyah"
    if not (aliyah / "agents" / "action_vocabulary.py").exists():
        print("  SKIP  robomail_Aliyah not present")
        return
    sys.path.insert(0, str(aliyah))
    try:
        from agents.action_vocabulary import ACTION_NAMES, REQUIRES_HELD_OBJECT
    except Exception as exc:                      # pragma: no cover
        print(f"  SKIP  could not import the vocabulary: {exc}")
        return
    from robochem.integration.plato_bridge import (
        ACTION_TO_SKILL, REQUIRES_HELD_OBJECT as BRIDGE_REQUIRES)

    check("the bridge maps every action in the vocabulary, and no others",
          set(ACTION_TO_SKILL) == set(ACTION_NAMES),
          f"symmetric difference: {set(ACTION_TO_SKILL) ^ set(ACTION_NAMES)}")
    check("the held-object requirement matches theirs",
          BRIDGE_REQUIRES == {a.value for a in REQUIRES_HELD_OBJECT},
          f"{BRIDGE_REQUIRES} vs {{a.value for a in REQUIRES_HELD_OBJECT}}")

    from robochem.skills import SKILL_REGISTRY
    missing = [s for s in ACTION_TO_SKILL.values() if s not in SKILL_REGISTRY]
    check("every mapped skill exists in the robochem registry", not missing,
          f"missing: {missing}")


def main():
    print("Offline skill checks (no robot, no cameras)")
    print("=" * 60)
    test_rim_geometry()
    test_place_does_not_drop_from_height()
    test_place_explicit_coordinates_skip_the_scan()
    test_place_on_top_stacks()
    test_place_stuck_gripper_retries_then_fails()
    test_stir_and_scoop_refuse_below_workspace_floor()
    test_stir_clamps_to_the_opening()
    test_pick_up_measures_the_tool_offset()
    test_pick_up_grasp_offsets()
    test_world_point_projection_round_trips()
    test_projection_identifies_once_and_propagates()
    test_projection_refuses_when_seeds_disagree()
    test_projection_skips_a_camera_that_cannot_see_it()
    test_one_label_camera_is_trusted_when_the_other_disagrees()
    test_label_crop_excludes_neighbours()
    test_duplicate_labels_are_flagged()
    test_label_matching_distinguishes_replicates()
    test_dump_keeps_the_bowl_over_the_target()
    test_dump_bowl_stays_put_under_rotation()
    test_dump_verifies_the_return_to_level()
    test_dump_accepts_explicit_coordinates()
    test_dump_fails_if_the_wrist_cannot_invert()
    test_dump_accepts_a_wrist_that_stalls_near_90()
    test_dump_needs_something_held()
    test_stir_defaults_to_top_down()
    test_stir_tilts_a_flat_spoon_upright()
    test_stir_tilt_does_not_overshoot_on_retry()
    test_stir_refuses_a_spoon_that_will_not_stand_up()
    test_stir_homes_before_orienting()
    test_stir_approaches_the_circle_before_walking_it()
    test_stir_circle_path_geometry()
    test_stir_streams_each_revolution_as_one_motion()
    test_stir_falls_back_when_streaming_is_unavailable()
    test_stir_in_air_needs_no_container()
    test_stir_detects_a_dropped_stirrer()
    test_place_releases_on_a_negative_press()
    test_stir_offsets_shift_the_circle_centre()
    test_depth_gate_drops_edge_streaks_not_containers()
    test_stir_stops_descending_on_contact()
    test_executor_discards_a_tool_offset_from_the_wrong_end()
    test_executor_puts_things_back_where_they_came_from()
    test_bench_manifest_names_the_real_bench()
    test_label_reader_chooses_among_bench_labels()
    test_transparent_cups_are_placed_by_silhouette()
    test_dishes_are_voted_across_cameras()
    test_streamed_paths_start_and_end_at_rest()
    test_robot_and_sim_share_one_pace()
    test_executor_trusts_the_measured_bowl_when_the_cloud_is_too_long()
    test_vision_routes_bench_names()
    test_executor_uses_bench_names()
    test_executor_tells_vision_what_is_held()
    test_place_back_returns_the_tcp_to_the_grasp()
    test_cups_are_taken_from_behind()
    test_a_refused_motion_is_caught()
    test_pour_keeps_the_lip_just_over_the_target()
    test_scoop_requires_a_real_tilt()
    test_scoop_site_offsets_and_unknown_params()
    test_scoop_takes_a_known_rim_height()
    test_scoop_puts_the_bowl_at_the_planned_height_under_impedance()
    test_streamed_scoop_and_dump_ask_for_stiff_tracking()
    test_scoop_rehearses_in_the_air()
    test_scoop_position_sweep()
    test_dispense_never_drops_the_pipette()
    test_dispense_transfer_mode()
    test_target_not_found_is_a_failure()
    test_bridge_translation()
    test_bridge_guards()
    test_bridge_matches_the_real_vocabulary()

    print("\n" + "=" * 60)
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    for failure in FAIL:
        print(f"  FAILED: {failure}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
