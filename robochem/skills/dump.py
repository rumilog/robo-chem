"""
Dump Skill

Empties a loaded scoop into a target container: put the BOWL over the target,
tip it past level until the powder falls out, shake it loose, come back level.

This is the delivery half of a scoop. `pour` cannot do it: pour tips a held
*container* about its own grip, where the spout is a few centimetres from the
wrist and the payload is liquid that runs out at ~90 degrees. A scoop's bowl
hangs on the end of a cranked handle, 50mm or more from the TCP, and powder
does not run — it sits in the bowl until the bowl is past vertical, and then
often still needs shaking.

The consequence that drives the whole implementation: **the bowl swings as the
wrist rotates**. Tipping in place moves the bowl on an arc of radius
|tool_offset|, so by the time it is inverted it is nowhere near the container
it was lined up with. Every tip step therefore recomputes the TCP so the BOWL
stays put — see ``BaseSkill.tcp_for_tip``. Without that a 50mm offset throws
the powder about 70mm wide of a cup at 120 degrees of tip.

Tip steps follow pour's proven shape: step the angle, measure what was actually
achieved, retry the shortfall, and fail rather than report success on a wrist
that stalled halfway.
"""

from typing import Dict, Any, Tuple
import numpy as np

from .base_skill import BaseSkill, _orthonormalize, tool_y_delta


class DumpSkill(BaseSkill):
    """
    Tip a loaded scoop out into a target container.

    Pipeline:
    1. Record held width; the scoop must survive the whole motion
    2. Home, still holding, so the wrist has travel for a big tip
    3. Scan the target, measure its rim
    4. Put the BOWL over the rim, level, with clearance
    5. Tip past level in steps, holding the bowl over the rim throughout
    6. Hold, then shake to dislodge what static holds back
    7. Come back level and lift clear
    """

    name = "dump"
    required_params = ["target_container"]

    #: ``target_container`` may be a name to segment, or explicit coordinates.
    #: Coordinates exist because identical unlabelled containers cannot be told
    #: apart by description: with four clear cups on the bench, each camera
    #: picks its own "best" one, they disagree in 3D, and consensus rejects the
    #: lot. A name is right when the object is distinctive or labelled; give
    #: coordinates when it is one of several lookalikes.

    @property
    def optional_params(self) -> Dict[str, Any]:
        return {
            # How far past level to tip. Powder needs the bowl actually
            # inverted — at 90 degrees it sits in the corner and stays there.
            "dump_angle_deg": 120.0,
            "step_deg": 30.0,
            "seconds_per_step": 1.5,
            "hold_duration": 1.0,
            # Static and damp powder cling; a couple of small oscillations at
            # the dump angle shift far more than a longer hold does.
            "shakes": 2,
            "shake_deg": 12.0,
            "shake_seconds": 0.4,
            # Height of the BOWL above the target rim while dumping. Generous:
            # the bowl sweeps an arc as it tips and must clear the rim.
            "clearance": 0.06,
            "approach_height": 0.12,
            "lift_height": 0.12,
            "home_first": True,
            "reset_before_scan": True,
            # TCP -> bowl offset in the TOOL frame. Same value scoop used; it
            # is what keeps the bowl over the target through the whole tip.
            "tool_length": 0.0,
            "tool_offset": None,
            # Site trim in the base frame, same convention as pour and scoop:
            # +X is forward from the base, so negative pulls it back.
            "forward_offset": 0.0,
            "lateral_offset": 0.0,
            "hover_tol": 0.05,
            "tip_tol_deg": 20.0,
            "step_retries": 3,
        }

    def check_preconditions(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        valid, msg = self.validate_params(params)
        if not valid:
            return False, msg

        if not self.is_gripper_holding():
            return False, ("Gripper is not holding anything. Pick up a scoop "
                           "and scoop something first.")

        if float(params.get("dump_angle_deg", 120.0)) <= 90.0:
            # Not fatal, but it will not empty, so say so up front.
            print("[Dump] WARNING: dump_angle_deg <= 90 leaves the bowl "
                  "upright-ish; powder will sit in the corner rather than fall.")

        return True, "Preconditions met"

    def execute(self, params: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        params = self.get_params_with_defaults(params)
        target = params["target_container"]
        dump_angle = abs(float(params["dump_angle_deg"]))
        step_deg = max(5.0, abs(float(params["step_deg"])))
        step_seconds = float(params["seconds_per_step"])
        hold_duration = float(params["hold_duration"])
        clearance = float(params["clearance"])
        tool_offset = self.resolve_tool_offset(params)
        tip_tol = float(params["tip_tol_deg"])
        retries = max(1, int(params["step_retries"]))

        print(f"[Dump] Emptying the scoop into '{target}'")

        start_width = self.held_width()
        print(f"[Dump] Holding scoop at {start_width * 1000:.1f}mm")

        # 1. Home first. A 120 degree tip needs wrist travel that the pose the
        # scoop finished in — out over the source container — does not have.
        if bool(params["home_first"]):
            print("[Dump] reset_joints (home) before tipping — keeping hold "
                  "of the scoop...")
            if not self.go_home():
                return False, {"error": "Failed to reset_joints before tipping"}
            self.wait(0.4)
            ok, msg = self.check_still_holding(start_width, tag="Dump")
            if not ok:
                return False, {"error": f"Lost the scoop while homing: {msg}"}

        # 2. Work out where the target is: segment it, or take it as given.
        if isinstance(target, (list, tuple, np.ndarray)):
            coords = np.asarray(target, dtype=float)
            if coords.shape[0] != 3:
                return False, {
                    "error": (f"target_container coordinates need [x, y, z], "
                              f"got {coords.shape[0]} values"),
                }
            rim_center = coords[:2]
            rim_z = float(coords[2])
            rim_radius = None
            print(f"[Dump] Target given as coordinates {np.round(coords, 4)} "
                  f"— no scan. Rim assumed at z={rim_z:.4f}.")
        else:
            if not self.clear_cameras(params, tag="Dump"):
                return False, {"error": "Failed to clear the cameras before scanning"}

            self.vision.clear_cache()
            located = self.locate_container(target, force_refresh=True)
            if located is None:
                return False, {"error": f"Cannot locate '{target}'"}

            rim_center = located["rim_center"]
            rim_z = located["top_z"]
            rim_radius = located["rim_radius"]

        ok, msg = self.check_still_holding(start_width, tag="Dump")
        if not ok:
            return False, {"error": f"Lost the scoop before dumping: {msg}"}
        site = np.asarray(rim_center, dtype=float) + np.array([
            float(params["forward_offset"]), float(params["lateral_offset"])])
        if rim_radius is not None:
            print(f"[Dump] '{target}' rim centre {np.round(rim_center, 4)}, "
                  f"z={rim_z:.4f}, radius {rim_radius * 1000:.0f}mm")
        if not np.allclose(site, rim_center):
            print(f"[Dump] Dump site shifted to {np.round(site, 4)}")

        # 3. The bowl is what has to sit over the container, not the wrist.
        level_rotation = self.tool_down_rotation()

        def rotation_at(angle_deg: float) -> np.ndarray:
            return _orthonormalize(level_rotation @ tool_y_delta(-float(angle_deg)))

        bowl_target = np.array([site[0], site[1], rim_z + clearance])

        approach = self.tcp_for_tip(
            np.array([site[0], site[1], rim_z + float(params["approach_height"])]),
            level_rotation, tool_offset)
        print(f"[Dump] Approaching above '{target}'...")
        if not self.goto_pose_rigid(approach, level_rotation, duration=3.0):
            return False, {"error": "Failed to command the approach"}
        arrived, err = self.reached(approach, float(params["hover_tol"]))
        if not arrived:
            return False, {
                "error": (f"Approach above '{target}' unreachable "
                          f"(off by {err * 1000:.0f}mm)"),
            }

        hover = self.tcp_for_tip(bowl_target, level_rotation, tool_offset)
        print(f"[Dump] Bowl over the rim at {np.round(bowl_target, 4)} "
              f"(TCP {np.round(hover, 4)})...")
        if not self.goto_pose_rigid(hover, level_rotation, duration=2.5):
            return False, {"error": "Failed to command the dump pose"}
        arrived, err = self.reached(hover, float(params["hover_tol"]))
        if not arrived:
            return False, {
                "error": (f"Could not place the bowl over '{target}' "
                          f"(off by {err * 1000:.0f}mm)"),
            }

        # 4. Tip in steps, keeping the BOWL over the rim at every angle.
        # Tipping in place would swing the bowl out on an arc of
        # |tool_offset| and throw the powder wide of the container.
        waypoints = list(np.arange(step_deg, dump_angle + 1e-6, step_deg))
        if not waypoints or abs(waypoints[-1] - dump_angle) > 0.5:
            waypoints.append(dump_angle)

        achieved = self.tool_tip_deg()
        for goal in waypoints:
            goal = float(goal)
            reached_step = False
            for attempt in range(1, retries + 1):
                R = rotation_at(goal)
                tcp = self.tcp_for_tip(bowl_target, R, tool_offset)
                duration = step_seconds * (1.0 + 0.4 * (attempt - 1))
                print(f"[Dump] → tip {goal:.0f}° "
                      f"(attempt {attempt}/{retries}, {duration:.1f}s), "
                      f"TCP {np.round(tcp, 4)}")
                if not self.goto_pose_rigid(tcp, R, duration=duration):
                    print("[Dump]   command failed")
                    continue
                achieved = self.tool_tip_deg()
                print(f"[Dump]   measured tip {achieved:.1f}°")
                if achieved >= goal - tip_tol:
                    reached_step = True
                    break
            if not reached_step:
                print(f"[Dump] Giving up at {goal:.0f}° (measured "
                      f"{achieved:.1f}°) — returning level")
                self.goto_pose_rigid(
                    self.tcp_for_tip(bowl_target, level_rotation, tool_offset),
                    level_rotation, duration=3.0)
                return False, {
                    "error": (f"Tip stalled at {achieved:.1f}° of "
                              f"{dump_angle:.0f}° — the scoop never inverted, "
                              f"so the powder did not come out"),
                    "tip_achieved": achieved,
                    "tip_commanded": dump_angle,
                }

        print(f"[Dump] Holding {hold_duration:.1f}s at {achieved:.1f}°...")
        self.wait(hold_duration)

        # 5. Shake. Powder clings; oscillation shifts what a hold will not.
        shakes = max(0, int(params["shakes"]))
        shake_deg = abs(float(params["shake_deg"]))
        if shakes and shake_deg > 0.5:
            print(f"[Dump] Shaking {shakes}x +/-{shake_deg:.0f}° to dislodge...")
            for i in range(shakes):
                for angle in (dump_angle - shake_deg, dump_angle):
                    R = rotation_at(angle)
                    self.goto_pose_rigid(
                        self.tcp_for_tip(bowl_target, R, tool_offset), R,
                        duration=float(params["shake_seconds"]))

        # 6. Back to level, then clear.
        print("[Dump] Returning level...")
        self.goto_pose_rigid(
            self.tcp_for_tip(bowl_target, level_rotation, tool_offset),
            level_rotation, duration=3.0)
        residual = self.tool_tip_deg()

        lift_tip = np.array([site[0], site[1],
                             rim_z + float(params["lift_height"])])
        lift = self.tcp_for_tip(lift_tip, level_rotation, tool_offset)
        print(f"[Dump] Lifting clear...")
        if not self.goto_pose_rigid(lift, level_rotation, duration=3.0):
            return False, {"error": "Dumped, but failed to lift clear",
                           "tip_achieved": achieved}

        ok, msg = self.check_still_holding(start_width, tag="Dump")
        if not ok:
            return False, {"error": f"Scoop lost during the dump: {msg}"}

        # The target's contents changed, so the cached cloud is stale.
        self.vision.clear_cache()

        print(f"[Dump] Emptied the scoop into '{target}' "
              f"(tipped {achieved:.1f}°)")
        return True, {
            "dumped_into": (target if isinstance(target, str) else "coordinates"),
            "tip_commanded": dump_angle,
            "tip_achieved": achieved,
            "residual_tilt": residual,
            "shakes": shakes,
            "rim_radius": rim_radius,
            "rim_z": rim_z,
            "tool_offset": [float(v) for v in tool_offset],
            "forward_offset": float(params["forward_offset"]),
            "lateral_offset": float(params["lateral_offset"]),
            # Nothing weighs what left the bowl, or what stayed stuck in it.
            "quantity_measured": False,
        }
