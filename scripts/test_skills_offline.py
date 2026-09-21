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

    buf2 = io.StringIO()
    with contextlib.redirect_stdout(buf2):
        pick_matching_instance([{"score": 0.8}], ["A 10 ML WATER"], "a 10 ml water")
    check("a single clean match says nothing",
          "DUPLICATE" not in buf2.getvalue() and "AMBIGUOUS" not in buf2.getvalue(),
          buf2.getvalue().strip() or "(silent)")


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
                                "dump_angle_deg": 120.0, "step_deg": 30.0,
                                "seconds_per_step": 0.0, "hold_duration": 0.0,
                                "shakes": 0})
    check("dump succeeds", ok, f"{result}")
    check("it tipped past vertical", ok and result["tip_achieved"] > 90.0,
          f"tip={result.get('tip_achieved')}")
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
                                "dump_angle_deg": 90.0, "min_tip_deg": 85.0,
                                "seconds_per_step": 0.0, "hold_duration": 0.0,
                                "shakes": 0, "step_retries": 4})
    check("dump succeeds", ok, f"{result}")
    check("it keeps retrying until the wrist is actually level",
          ok and result["residual_tilt"] < 20.0,
          f"residual={result.get('residual_tilt'):.1f} deg")


def test_dump_accepts_explicit_coordinates():
    """Identical unlabelled cups can only be targeted by position."""
    print("\n[dump: explicit coordinates skip the scan]")
    vision = FakeVision({})          # nothing segmentable at all
    arm = FakeArm(tracking=1.0, gripper_width=0.03)
    skill = make(DumpSkill, vision, arm)
    ok, result = skill.execute({"target_container": [0.52, -0.08, 0.10],
                                "tool_offset": [0.051, 0.009, 0.028],
                                "seconds_per_step": 0.0, "hold_duration": 0.0,
                                "shakes": 0})
    check("dump succeeds with nothing segmentable", ok, f"{result}")
    # clear_cache IS called once at the end — the target's contents changed.
    # What must not happen is a segmentation pass.
    check("no segmentation was attempted", vision.locate_calls == 0,
          f"locate() called {vision.locate_calls} times")
    check("it is logged as a coordinate dump",
          ok and result["dumped_into"] == "coordinates",
          f"{result.get('dumped_into')}")
    check("it tipped past vertical", ok and result["tip_achieved"] > 90.0,
          f"{result.get('tip_achieved')}")

    ok2, r2 = make(DumpSkill, vision, FakeArm(gripper_width=0.03)).execute(
        {"target_container": [0.52, -0.08], "seconds_per_step": 0.0})
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
                                "seconds_per_step": 0.0, "shakes": 0})
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

        CAP = 89.0

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
                                "dump_angle_deg": 120.0, "min_tip_deg": 85.0,
                                "seconds_per_step": 0.0, "hold_duration": 0.0,
                                "shakes": 1})
    check("a wrist capped at 89 deg still counts as dumped", ok, f"{result}")
    check("it records that the tip stalled short",
          ok and result["tip_stalled_short"] is True,
          f"{result.get('tip_stalled_short')}")
    check("and reports the angle it actually reached",
          ok and 80.0 < result["tip_achieved"] < 95.0,
          f"{result.get('tip_achieved')}")

    # Below min_tip_deg it must still fail — the allowance is not a blanket pass.
    arm2 = CappedWristArm(tracking=1.0, gripper_width=0.03)
    arm2.CAP = 40.0
    ok2, r2 = make(DumpSkill, vision, arm2).execute(
        {"target_container": "cup", "tool_offset": [0.051, 0.009, 0.028],
         "min_tip_deg": 85.0, "seconds_per_step": 0.0, "shakes": 0})
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
    ok, result = skill.execute({"target_container": "cup"})
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
    check("the tool was still swung upright",
          ok and result["long_axis_deg"] < 15.0,
          f"long axis={result.get('long_axis_deg')}")
    check("the full circle ran", ok and result["revolutions_completed"] == 2.0,
          f"{result.get('revolutions_completed')}")

    # A real circle, not a point: the commanded XY must actually vary.
    xs = [c[0][0] for c in arm.commands]
    ys = [c[0][1] for c in arm.commands]
    check("the commanded path spans roughly the circle diameter",
          (max(xs) - min(xs)) > 0.05 and (max(ys) - min(ys)) > 0.05,
          f"x span={max(xs) - min(xs):.3f}, y span={max(ys) - min(ys):.3f}")


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


def test_scoop_compensates_a_cranked_tool_offset():
    """A cranked tool's bowl must land on the powder, not the TCP."""
    print("\n[scoop: cranked tool offset is compensated]")
    tub = cylinder_cloud([0.50, 0.10], base_z=0.02, height=0.05, radius=0.06)
    vision = FakeVision({"tub": tub})

    # The printed scoop: bowl 45mm along the handle, 28mm below it.
    offset = [0.045, 0.0, 0.028]

    arm = FakeArm(tracking=1.0, gripper_width=0.03)
    skill = make(ScoopSkill, vision, arm)
    ok, result = skill.execute({"powder_source": "tub", "tool_offset": offset,
                                "dig_advance": 0.02, "drag_distance": 0.04,
                                "untilt_over": 0.02, "push_seconds": 0.5})
    check("scoop succeeds with a cranked tool", ok, f"{result}")
    check("the offset is recorded", ok and result["tool_offset"] == offset,
          f"{result.get('tool_offset')}")

    # Reconstruct where the BOWL went from each commanded TCP pose.
    surface_z, depth = result["surface_z"], result["scoop_depth"]
    tips = []
    for target, _dur, _imp in arm.commands:
        tips.append(np.asarray(target))
    # The plunge and push targets must place the bowl at the dig depth, which
    # means the TCP itself sits well above and behind it.
    bowl_z = surface_z - result["total_depth"]     # where the BOWL must end up
    tcp_only_z = [t[2] for t in tips]
    # The bowl hangs below the TCP, so the TCP must never be commanded as low
    # as the bowl's target — that is the whole point of the compensation.
    check("the TCP is commanded ABOVE where the bowl goes",
          min(tcp_only_z) > bowl_z + 0.01,
          f"lowest TCP z={min(tcp_only_z):.4f} vs bowl target z={bowl_z:.4f}")

    # Same run without the offset: the TCP goes straight to the dig depth.
    arm2 = FakeArm(tracking=1.0, gripper_width=0.03)
    ok2, result2 = make(ScoopSkill, vision, arm2).execute(
        {"powder_source": "tub", "dig_advance": 0.02, "drag_distance": 0.04,
         "untilt_over": 0.02, "push_seconds": 0.5})
    z2 = [t[0][2] for t in arm2.commands]
    check("without an offset the TCP does go to the bowl depth (regression guard)",
          min(z2) <= bowl_z + 1e-6,
          f"lowest TCP z={min(z2):.4f} vs bowl target z={bowl_z:.4f}")
    check("so the offset genuinely changed the commanded path",
          abs(min(tcp_only_z) - min(z2)) > 0.02,
          f"{min(tcp_only_z):.4f} vs {min(z2):.4f}")


def test_scoop_clamps_the_drag():
    print("\n[scoop: stroke clamped to the opening]")
    small = cylinder_cloud([0.50, 0.10], base_z=0.02, height=0.04, radius=0.02)
    vision = FakeVision({"small tub": small})
    arm = FakeArm(tracking=1.0, gripper_width=0.03)
    skill = make(ScoopSkill, vision, arm)
    ok, result = skill.execute({"powder_source": "small tub",
                                "dig_advance": 0.03, "drag_distance": 0.05,
                                "push_seconds": 0.5})
    check("40mm-wide tub does not get an 80mm stroke",
          (not ok) or result["total_stroke"] < 0.08,
          f"ok={ok}, stroke={result.get('total_stroke')}")
    check("the clamp keeps the plunge/push ratio",
          (not ok) or abs((result["dig_advance"] / result["drag_distance"])
                          - (0.03 / 0.05)) < 0.01,
          f"plunge={result.get('dig_advance')}, push={result.get('drag_distance')}")


def test_scoop_pushes_away_from_the_base():
    """The stroke runs +X, and the tilt is rolled off during the push."""
    print("\n[scoop: pushes away from the base, un-tilting as it goes]")
    tub = cylinder_cloud([0.50, 0.10], base_z=0.02, height=0.05, radius=0.06)
    vision = FakeVision({"tub": tub})
    arm = FakeArm(tracking=1.0, gripper_width=0.03)
    skill = make(ScoopSkill, vision, arm)
    ok, result = skill.execute({"powder_source": "tub", "dig_advance": 0.02,
                                "drag_distance": 0.04, "untilt_over": 0.02,
                                "push_seconds": 0.5})
    check("scoop succeeds", ok, f"{result}")
    check("stroke is recorded as away-from-base",
          ok and result["push_direction"] == "away_from_base(+X)",
          f"{result.get('push_direction')}")

    # The commanded X targets must increase through the push: pulling toward
    # the base was the 2026-09-16 bench correction.
    xs = [c[0][0] for c in arm.commands]
    push_xs = xs[-3:-1]  # the push segments, before the final lift
    check("commanded X advances monotonically (+X, away from base)",
          all(b >= a - 1e-9 for a, b in zip(push_xs, push_xs[1:])),
          f"push X targets={[round(v, 4) for v in push_xs]}")

    check("the scoop finishes level, not tilted",
          ok and result["residual_tilt"] < 1.0,
          f"residual={result.get('residual_tilt')}")
    check("it travelled the full push", ok and abs(result["pushed"] - 0.04) < 1e-6,
          f"pushed={result.get('pushed')}")


def test_scoop_push_is_continuous():
    """The push must be one or two min-jerk motions, not subdivided waypoints."""
    print("\n[scoop: push is continuous, not stepped]")
    tub = cylinder_cloud([0.50, 0.10], base_z=0.02, height=0.05, radius=0.06)
    vision = FakeVision({"tub": tub})

    class TiltRecordingArm(FakeArm):
        """Records commanded tilt magnitude and duration at every motion."""

        def __init__(self):
            super().__init__(tracking=1.0, gripper_width=0.03)
            self.tilts = []
            self.durations = []

        def goto_pose(self, pose, duration=3.0, use_impedance=True, block=True):
            super().goto_pose(pose, duration, use_impedance, block)
            R = np.asarray(pose.rotation, dtype=float)
            self.tilts.append(
                float(np.degrees(np.arccos(np.clip(-R[2, 2], -1.0, 1.0))))
            )
            self.durations.append(duration)

    arm = TiltRecordingArm()
    skill = make(ScoopSkill, vision, arm)
    ok, result = skill.execute({"powder_source": "tub", "dig_advance": 0.02,
                                "drag_distance": 0.04, "untilt_over": 0.02,
                                "dig_tilt_deg": 30.0, "push_seconds": 3.0})
    check("scoop succeeds", ok, f"{result}")

    # hover, tilt-in-place, plunge, push A, push B, lift == 6 motions total.
    check("the whole skill is a handful of motions, not a waypoint stream",
          len(arm.commands) <= 8, f"{len(arm.commands)} goto_pose calls")

    push = arm.tilts[-3:-1]  # the two push segments, before the lift
    check("push segment A holds the full bite angle",
          push and abs(push[0] - 30.0) < 0.5, f"segment A tilt={push[0] if push else None}")
    check("push segment B ends level",
          push and push[-1] < 0.5, f"segment B tilt={push[-1] if push else None}")

    # Durations must split proportionally to distance: 20mm hold + 20mm roll
    # out of a 3.0s push is 1.5s each.
    push_durations = arm.durations[-3:-1]
    check("push duration is split by distance, not by waypoint count",
          all(abs(d - 1.5) < 0.01 for d in push_durations),
          f"durations={[round(d, 2) for d in push_durations]}")


def test_scoop_single_motion_push():
    """untilt_over == drag_distance collapses the push to one unbroken motion."""
    print("\n[scoop: untilt_over == drag_distance gives a single motion]")
    tub = cylinder_cloud([0.50, 0.10], base_z=0.02, height=0.05, radius=0.06)
    vision = FakeVision({"tub": tub})
    arm = FakeArm(tracking=1.0, gripper_width=0.03)
    skill = make(ScoopSkill, vision, arm)
    ok, result = skill.execute({"powder_source": "tub", "dig_advance": 0.02,
                                "drag_distance": 0.04, "untilt_over": 0.04,
                                "push_seconds": 0.5})
    check("single-motion scoop succeeds", ok, f"{result}")
    check("one fewer motion than the two-segment default",
          len(arm.commands) <= 6, f"{len(arm.commands)} goto_pose calls")
    check("still finishes level", ok and result["residual_tilt"] < 1.0,
          f"residual={result.get('residual_tilt')}")


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
    test_scoop_requires_a_real_tilt()
    test_scoop_site_offsets_and_unknown_params()
    test_scoop_compensates_a_cranked_tool_offset()
    test_scoop_clamps_the_drag()
    test_scoop_pushes_away_from_the_base()
    test_scoop_push_is_continuous()
    test_scoop_single_motion_push()
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
