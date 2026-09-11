"""
Scoop Skill

Transfers a measured quantity of powder out of a source container with a held
scoop. This is how dry reagents are metered — the profile in robomail_Aliyah's
`config/robot_profile.py` explicitly rules out metering a pour by weight.

``powder_source`` should be the **reagent label** on the paper under the cup
(e.g. ``"citric acid"``, ``"baking soda"``), not a generic ``"white paper cup"``.
Vision segments every white paper cup, reads the labels with a VLM, then scoops
from the matching instance.

Bench-corrected 2026-09-11: the scoop must tilt forward and dig/drag through
the powder *tilted*, the way pour tips toward its site before the liquid moves
— tilting only after the drag just plows the powder flat instead of collecting
it. Order is now: tilt forward -> dig+drag (tilted, compliant) -> tilt back to
normal -> lift (level).

Hardened along the same lines as pick_up / pour:
  - reset_joints before scanning, world-frame rim geometry from the pointcloud
  - the drag is clamped to the measured opening, so the scoop cannot ram the wall
  - the dig and drag run *compliant* (impedance on) because they are contact
    motions; the free-space hover and the lift run stiff and are verified
  - the forward dig tilt uses tool-axis rotation and is confirmed by
    measurement, the way pour confirms its tip instead of assuming it
"""

from typing import Dict, Any, Tuple
import numpy as np

from .base_skill import BaseSkill, _orthonormalize


class ScoopSkill(BaseSkill):
    """
    Scoop powder from a source container with a held scoop or spoon.

    Pipeline:
    1. Record held width; the scoop must survive the whole motion
    2. Clear the cameras, scan the source, measure rim height / radius
    3. Hover level over the powder (free space, verified strictly)
    4. Tilt forward about the tool axis — bites into the powder the way pour
       tips toward its site — and verify the tilt actually happened
    5. Descend into the powder tilted AND advance forward in the same motion
       (the entry stroke itself carries travel), then drag any remaining
       distance across the bed — both compliant
    6. Tilt back to normal (level) before leaving the powder
    7. Lift straight out, level
    """

    name = "scoop"
    required_params = ["powder_source"]

    @property
    def optional_params(self) -> Dict[str, Any]:
        return {
            # How far below the powder surface to dig. The surface is taken as
            # the top of the fused cloud inside the container, which for a
            # part-full tub is the powder itself, not the rim.
            "scoop_depth": 0.015,
            # Length of the drag through the powder. Clamped to the opening.
            "scoop_distance": 0.04,
            # How far to advance (toward the base, same direction as the drag)
            # WHILE tilting into the powder — the entry stroke itself carries
            # forward travel, not just a straight vertical plunge. Clamped to
            # scoop_distance. Bench feedback 2026-09-11: with this at 0 (the
            # original behaviour) the dig was a straight-down plunge with no
            # bite; 3cm is the corrected starting point.
            "dig_advance": 0.03,
            # Degrees to tilt forward, about the tool's own closing axis (NOT
            # the base frame — see rotate_about_tool_axis), BEFORE dragging.
            # This is the "bite" angle — the same role pour's tip angle plays —
            # and is what lets the drag actually gather powder onto the scoop
            # rather than push it flat.
            "dig_tilt_deg": 30.0,
            "lift_height": 0.12,
            "approach_height": 0.10,
            "reset_before_scan": True,
            # Distance kept from the container wall during the drag.
            "wall_clearance": 0.015,
            # Offset from the gripper TCP to the scoop bowl, in metres.
            # MEASURE THIS for your scoop. Left at 0 the arm digs with the
            # gripper itself, which is wrong for any tool of real length.
            "tool_length": 0.0,
            "hover_tol": 0.05,
            # The dig and drag are contact motions: the powder resists, so the
            # arm legitimately stops short and a tight tolerance would fail
            # every successful scoop. Only a gross miss is a failure.
            "contact_tol": 0.05,
            "tilt_tol_deg": 12.0,
            # frankapy often needs a second attempt before the wrist tracks a
            # commanded orientation; pour retries the same way.
            "tilt_retries": 3,
        }

    def check_preconditions(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        valid, msg = self.validate_params(params)
        if not valid:
            return False, msg

        if not self.is_gripper_holding():
            return False, "Gripper is not holding anything. Pick up a scoop/spoon first."

        return True, "Preconditions met"

    def execute(self, params: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        params = self.get_params_with_defaults(params)
        source = params["powder_source"]
        scoop_depth = float(params["scoop_depth"])
        requested_distance = float(params["scoop_distance"])
        requested_dig_advance = float(params["dig_advance"])
        dig_tilt_deg = float(params["dig_tilt_deg"])
        lift_height = float(params["lift_height"])
        tool_length = float(params["tool_length"])
        wall_clearance = float(params["wall_clearance"])
        contact_tol = float(params["contact_tol"])
        tilt_tol = float(params["tilt_tol_deg"])
        retries = max(1, int(params["tilt_retries"]))

        print(f"[Scoop] Starting scoop from '{source}'")

        start_width = self.held_width()
        print(f"[Scoop] Holding scoop at {start_width * 1000:.1f}mm")

        # 1. Clear the cameras and scan the source.
        if not self.clear_cameras(params, tag="Scoop"):
            return False, {"error": "Failed to clear the cameras before scanning"}

        self.vision.clear_cache()
        located = self.locate_container(source, force_refresh=True)
        if located is None:
            return False, {"error": f"Cannot locate '{source}'"}

        ok, msg = self.check_still_holding(start_width, tag="Scoop")
        if not ok:
            return False, {"error": f"Lost the scoop before scooping: {msg}"}

        rim_center = located["rim_center"]
        surface_z = located["top_z"]
        rim_radius = located["rim_radius"]
        print(f"[Scoop] '{source}' surface z={surface_z:.4f}, "
              f"centre {np.round(rim_center, 4)}, "
              f"opening radius {rim_radius * 1000:.0f}mm")

        # 2. Clamp the drag so it stays inside the container. A 50mm drag in a
        # 35mm-wide reagent tub just rams the far wall and stalls the arm.
        usable = max(0.0, rim_radius - wall_clearance)
        distance = min(requested_distance, 2.0 * usable)
        if distance < requested_distance:
            print(f"[Scoop] Clamping drag {requested_distance * 1000:.0f}mm -> "
                  f"{distance * 1000:.0f}mm (opening radius "
                  f"{rim_radius * 1000:.0f}mm minus "
                  f"{wall_clearance * 1000:.0f}mm clearance)")
        if distance < 0.008:
            return False, {
                "error": (f"'{source}' opening is only "
                          f"{rim_radius * 2000:.0f}mm across — no room to drag "
                          f"the scoop through the powder"),
                "rim_radius": rim_radius,
            }

        # 3. Drag toward the robot base (-X), the same direction pour tips.
        # Pulling toward the base keeps the elbow inside its comfortable range;
        # pushing away runs into the reach limit that pour already documented.
        drag = np.array([-1.0, 0.0])
        entry_xy = rim_center - drag * (distance / 2.0)
        exit_xy = rim_center + drag * (distance / 2.0)

        # The entry stroke itself carries forward travel (see dig_advance
        # above) — clamp it to the overall drag so the dig point can never
        # overshoot exit_xy.
        dig_advance = min(requested_dig_advance, distance)
        dig_xy = entry_xy + drag * dig_advance
        remaining_drag = distance - dig_advance

        level_rotation = self.tool_down_rotation()
        dig_z = surface_z - scoop_depth + tool_length
        if dig_z < self.workspace_min[2]:
            return False, {
                "error": (f"Dig depth would put the wrist at z={dig_z:.3f}, below "
                          f"the workspace floor {self.workspace_min[2]:.3f}"),
            }

        # 4. Hover above the entry point, level (free space, verified strictly).
        hover = np.array([entry_xy[0], entry_xy[1],
                          surface_z + float(params["approach_height"]) + tool_length])
        print(f"[Scoop] Hovering above the powder at {np.round(hover, 4)}...")
        if not self.goto_pose_rigid(hover, level_rotation, duration=3.0):
            return False, {"error": "Failed to command hover pose"}
        arrived, err = self.reached(hover, float(params["hover_tol"]))
        if not arrived:
            return False, {
                "error": (f"Hover above '{source}' unreachable "
                          f"(off by {err * 1000:.0f}mm)"),
            }

        # 5. Tilt FORWARD before touching the powder — the same role pour's
        # tip angle plays. Tilting only after the drag (the old order) just
        # plows the powder flat instead of collecting it onto the scoop; the
        # forward tilt is what lets the leading edge bite in as it drags.
        tilt_before = self.tool_tip_deg()
        print(f"[Scoop] Tilting forward {dig_tilt_deg:.0f}° before entering "
              f"the powder (tip now {tilt_before:.1f}° from vertical)...")

        # Retry the *remaining* angle, not the whole command: frankapy often
        # needs a second, longer goto_pose before the wrist tracks orientation
        # (the same lag pour retries through). Re-commanding the full angle
        # would stack rotations and overshoot.
        #
        # NEGATED on purpose: tool_tip_deg() is an arccos-based magnitude
        # (always >= 0), so it cannot tell "tilted forward" from "tilted the
        # wrong way" — only how far from vertical. Bench feedback 2026-09-11
        # was that the un-negated command tilted 30 deg in the wrong physical
        # direction while still reading as "achieved 30 deg" here, since the
        # magnitude was right even though the sign was not.
        achieved = 0.0
        for attempt in range(1, retries + 1):
            remaining = dig_tilt_deg - achieved
            if remaining <= 0.5:
                break
            if not self.rotate_about_tool_axis(-remaining, axis="y"):
                return False, {"error": "Failed to command the forward dig tilt"}
            self.wait(0.3)
            achieved = self.tool_tip_deg() - tilt_before
            print(f"[Scoop]   attempt {attempt}/{retries}: tilt "
                  f"{achieved:+.1f}° of {dig_tilt_deg:.0f}°")
            if abs(achieved) >= dig_tilt_deg - tilt_tol:
                break

        print(f"[Scoop] Measured forward tilt {self.tool_tip_deg():.1f}° "
              f"(Δ{achieved:+.1f}°, commanded {dig_tilt_deg:.0f}°)")
        if abs(achieved) < dig_tilt_deg - tilt_tol:
            # A scoop that never got the bite angle just pushes powder around
            # level; that is not a successful scoop, so fail rather than
            # continue into a drag that will not collect anything. Back off
            # to level first — we are still above the powder, not in it.
            self.goto_pose_rigid(hover, level_rotation, duration=2.0)
            return False, {
                "error": (f"Forward dig tilt stalled: commanded {dig_tilt_deg:.0f}° "
                          f"but only achieved {achieved:.1f}° after {retries} "
                          f"attempts (tol {tilt_tol:.0f}°). Dragging level "
                          f"through the powder without this tilt would not "
                          f"collect any onto the scoop."),
                "tilt_commanded": dig_tilt_deg,
                "tilt_achieved": achieved,
            }

        tilted_rotation = _orthonormalize(
            np.asarray(self.get_current_pose().rotation, dtype=float)
        )

        # 6. Dig in, tilted — AND advance forward by dig_advance in the same
        # motion. This is the entry stroke: a straight-down plunge with no
        # forward component doesn't bite into anything, the way a shovel has
        # to already be moving into the ground as it goes in. Compliant: the
        # scoop is entering a granular bed, so the controller should yield
        # rather than push through at full stiffness.
        entry = np.array([dig_xy[0], dig_xy[1], dig_z])
        print(f"[Scoop] Digging in to z={dig_z:.4f} and advancing "
              f"{dig_advance * 1000:.0f}mm toward the base "
              f"({scoop_depth * 1000:.0f}mm deep, tilted, compliant)...")
        if not self.goto_pose_rigid(entry, tilted_rotation, duration=3.0,
                                    use_impedance=True):
            return False, {"error": "Failed to command dig pose"}
        arrived, err = self.reached(entry, contact_tol)
        if not arrived:
            print(f"[Scoop] Dig stopped {err * 1000:.0f}mm short — backing out")
            self.goto_pose_rigid(hover, level_rotation, duration=3.0)
            return False, {
                "error": (f"Could not enter the powder in '{source}' "
                          f"(off by {err * 1000:.0f}mm); the scoop is probably "
                          f"fouling the rim"),
            }

        # 7. Drag the remaining distance through the powder, still tilted,
        # still compliant. This is the rest of the motion that gathers powder
        # onto the scoop, continuing from where the dig's forward advance left
        # off; a no-op if dig_advance already covered the whole scoop_distance.
        exit_point = np.array([exit_xy[0], exit_xy[1], dig_z])
        print(f"[Scoop] Dragging the remaining {remaining_drag * 1000:.0f}mm "
              f"toward the base to {np.round(exit_point, 4)} (tilted)...")
        if not self.goto_pose_rigid(exit_point, tilted_rotation, duration=3.0,
                                    use_impedance=True):
            return False, {"error": "Failed to command the drag"}
        arrived, err = self.reached(exit_point, contact_tol)
        if not arrived:
            # Worth reporting but not fatal — a partial drag still lifts powder.
            print(f"[Scoop] Drag ended {err * 1000:.0f}mm short of target "
                  f"(powder resistance); continuing with a partial scoop")

        ok, msg = self.check_still_holding(start_width, tag="Scoop")
        if not ok:
            return False, {"error": f"Scoop lost while digging: {msg}"}

        # 8. Tilt back to normal before leaving the powder — the pour-return
        # equivalent of coming back upright after tipping. Recomputing
        # tool_down_rotation() rather than rotating back by -achieved is
        # self-correcting: rotate_about_tool_axis(..., axis="y") leaves the
        # tool's own y-axis fixed, so flattening from wherever we ended up
        # recovers the pre-tilt level orientation exactly rather than
        # compounding drift. Not gated on success, same as pour's return to
        # upright — the scoop already has whatever it collected; a residual
        # few degrees here costs some spillage, not the whole attempt.
        print("[Scoop] Tilting back to normal before lifting...")
        residual = None
        for attempt in range(1, retries + 1):
            normal_rotation = self.tool_down_rotation()
            if not self.goto_pose_rigid(exit_point, normal_rotation, duration=2.0,
                                        use_impedance=True):
                print("[Scoop] Warning: return-to-normal command failed")
                break
            self.wait(0.2)
            residual = self.tool_tip_deg()
            print(f"[Scoop]   attempt {attempt}/{retries}: residual tilt "
                  f"{residual:.1f}° from vertical")
            if residual <= tilt_tol:
                break
        if residual is not None and residual > tilt_tol:
            print(f"[Scoop] Did not fully return to level (residual "
                  f"{residual:.1f}°) — lifting anyway rather than holding "
                  f"the scoop tilted in the powder")

        level_rotation = _orthonormalize(
            np.asarray(self.get_current_pose().rotation, dtype=float)
        )

        # 9. Lift straight out, level.
        lift = np.asarray(self.get_current_pose().translation, dtype=float).copy()
        lift[2] = surface_z + lift_height + tool_length
        print(f"[Scoop] Lifting clear to z={lift[2]:.4f}...")
        if not self.goto_pose_rigid(lift, level_rotation, duration=3.0):
            return False, {"error": "Failed to lift the scoop clear"}
        arrived, err = self.reached(lift, float(params["hover_tol"]))
        if not arrived:
            return False, {
                "error": f"Scoop did not lift clear (off by {err * 1000:.0f}mm)",
            }

        ok, msg = self.check_still_holding(start_width, tag="Scoop")
        if not ok:
            return False, {"error": f"Scoop lost during the lift: {msg}"}

        # The powder bed changed shape, so the cached source cloud is stale.
        self.vision.clear_cache()

        print(f"[Scoop] Successfully scooped from '{source}'")
        return True, {
            "scooped_from": source,
            "scoop_depth": scoop_depth,
            "scoop_distance": distance,
            "requested_distance": requested_distance,
            "dig_advance": dig_advance,
            "remaining_drag": remaining_drag,
            "rim_radius": rim_radius,
            "surface_z": surface_z,
            "dig_tilt_commanded": dig_tilt_deg,
            "dig_tilt_achieved": achieved,
            "residual_tilt_after_normalize": residual,
            # There is no measurement of how much powder is on the scoop. The
            # quantity is nominal, set by depth and drag length.
            "quantity_measured": False,
        }
