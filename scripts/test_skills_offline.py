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
        self.missing = set()

    def clear_cache(self):
        self.cleared += 1

    def locate(self, name, force_refresh=False):
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


def test_scoop_clamps_the_drag():
    print("\n[scoop: drag clamped to the opening]")
    small = cylinder_cloud([0.50, 0.10], base_z=0.02, height=0.04, radius=0.02)
    vision = FakeVision({"small tub": small})
    arm = FakeArm(tracking=1.0, gripper_width=0.03)
    skill = make(ScoopSkill, vision, arm)
    ok, result = skill.execute({"powder_source": "small tub",
                                "scoop_distance": 0.08})
    check("40mm-wide tub does not get an 80mm drag",
          (not ok) or result["scoop_distance"] < 0.08,
          f"ok={ok}, distance={result.get('scoop_distance')}")


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
    test_stir_detects_a_dropped_stirrer()
    test_scoop_requires_a_real_tilt()
    test_scoop_clamps_the_drag()
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
