"""
Scoop Skill

Transfers a measured quantity of powder out of a source container with a held
scoop. This is how dry reagents are metered — the profile in robomail_Aliyah's
`config/robot_profile.py` explicitly rules out metering a pour by weight.

``powder_source`` should be the **reagent label** on the paper under the cup
(e.g. ``"citric acid"``, ``"baking soda"``), not a generic ``"white paper cup"``.
Vision segments every white paper cup, reads the labels with a VLM, then scoops
from the matching instance.

The stroke, as corrected against the bench over 2026-09-11..16:

    hover level -> tilt forward -> plunge in while advancing (tilted)
                -> push AWAY from the base, rolling back to level over the
                   last stretch of that push -> lift

Three things that each took a hardware run to find:

  1. **The tilt has to happen before the powder is touched.** Tilting after the
     stroke just plows the powder flat instead of collecting it.
  2. **The tilt direction is the opposite of the obvious sign.** See the comment
     on the negation in ``execute``: ``tool_tip_deg()`` is an arccos magnitude,
     always >= 0, so it cannot tell "tilted forward" from "tilted backward" and
     reports a wrong-direction tilt as a clean success. Nothing catches this
     except a person watching the arm.
  3. **The un-tilt belongs inside the push, not after it.** Rolling the bowl
     level *while still moving forward* is what keeps the powder on the scoop —
     the way a person finishes a scooping stroke. Snapping upright after the
     motion has already stopped just drops it back in the cup.

Hardened along the same lines as pick_up / pour:
  - reset_joints before scanning, world-frame rim geometry from the pointcloud
  - the stroke is clamped to the measured opening, so it cannot ram the wall
  - the plunge and push run *compliant* (impedance on) because they are contact
    motions; the free-space hover and the lift run stiff and are verified
  - the forward tilt is confirmed by measurement before the scoop commits to
    the powder, the way pour confirms its tip instead of assuming it
  - the push is discrete blocking waypoints, not a streamed trajectory, for the
    same reason stir walks its circle that way (see P11 in SKILLS_ROADMAP.md)
"""

from typing import Dict, Any, Tuple
import numpy as np

from .base_skill import BaseSkill, _orthonormalize, tool_y_delta


class ScoopSkill(BaseSkill):
    """
    Scoop powder from a source container with a held scoop or spoon.

    Pipeline:
    1. Record held width; the scoop must survive the whole motion
    2. Clear the cameras, scan the source, measure rim height / radius
    3. Hover level over the powder (free space, verified strictly)
    4. Tilt forward about the tool axis and verify it actually happened —
       failing here, before the powder is touched, is cheap
    5. Plunge to the dig depth while advancing forward, tilted, compliant
    6. Push away from the base in discrete waypoints, rolling the tilt back to
       level over the last ``untilt_over`` metres of that push
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
            # Forward travel during the tilted plunge — the entry stroke itself
            # carries travel rather than being a straight vertical drop, the
            # way a shovel is already moving into the ground as it enters.
            "dig_advance": 0.02,
            # The push after the plunge, away from the base. In this direction
            # it is really more of a push than a drag.
            "drag_distance": 0.04,
            # Over the last this-many metres of the push, blend the tilt back
            # to level. Finishing the roll *while still moving* is what makes
            # the stroke read as a scoop instead of a plow followed by a flick.
            "untilt_over": 0.02,
            # Duration of the whole push. frankapy's goto_pose is a min-jerk
            # interpolation, so the push is one or two continuous motions —
            # NOT subdivided waypoints. See the comment in execute().
            "push_seconds": 3.0,
            # Degrees to tilt forward, about the tool's own closing axis (NOT
            # the base frame). This is the "bite" angle — the same role pour's
            # tip angle plays. Raised from 30 to 45 on the bench 2026-09-18:
            # the shallower entry skated over the powder instead of cutting in.
            "dig_tilt_deg": 45.0,
            # Extra descent AFTER the tilted plunge has landed, before the push
            # starts — the scoop drops its nose in, then digs down a little
            # further into the bed. Separate from scoop_depth so the entry
            # angle and the final depth can be tuned independently.
            "dip_depth": 0.010,
            "lift_height": 0.12,
            "approach_height": 0.10,
            "reset_before_scan": True,
            # SAM category to search when the target is identified by a
            # written label. The label path only considers instances of this
            # category, so a labelled CLEAR cup is invisible under the default
            # "white paper cup" and the query quietly finds nothing.
            "container_category": None,
            # Shift the whole stroke in the robot base frame, same sign
            # convention as pour and pick_up: +X is forward from the base
            # toward the workspace, +Y is the robot's left. The stroke is
            # otherwise centred on the measured rim, which is only right when
            # the segmentation found the container itself and the tool_offset
            # is current for the grasp actually being held.
            "forward_offset": 0.0,
            "lateral_offset": 0.0,
            # Distance kept from the container wall across the whole stroke.
            # 10mm, not 15mm: a 60mm stroke centred in an 86mm cup leaves 13mm
            # at each end, and the old 15mm was clamping strokes that fit fine.
            "wall_clearance": 0.010,
            # Offset from the gripper TCP to the scoop bowl, in metres.
            # MEASURE THIS for your scoop. Left at 0 the arm digs with the
            # gripper itself, which is wrong for any tool of real length.
            #
            # A CRANKED tool needs the full vector, not just a depth: pass
            # tool_offset [x, y, z] in the TOOL frame (x along the handle,
            # z down). The printed scoop is [0.045, 0, 0.028] for a mid-grip
            # grasp. tool_length alone puts the bowl 25mm off the target once
            # the bite tilt is applied. See BaseSkill.resolve_tool_offset.
            "tool_length": 0.0,
            "tool_offset": None,
            "hover_tol": 0.05,
            # The plunge and push are contact motions: the powder resists, so
            # the arm legitimately stops short and a tight tolerance would fail
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
        requested_advance = float(params["dig_advance"])
        requested_drag = float(params["drag_distance"])
        requested_untilt = float(params["untilt_over"])
        dig_tilt_deg = float(params["dig_tilt_deg"])
        lift_height = float(params["lift_height"])
        tool_offset = self.resolve_tool_offset(params)
        tool_length = float(tool_offset[2])   # nominal drop, for logging/limits
        wall_clearance = float(params["wall_clearance"])
        contact_tol = float(params["contact_tol"])
        dip_depth = max(0.0, float(params["dip_depth"]))
        tilt_tol = float(params["tilt_tol_deg"])
        retries = max(1, int(params["tilt_retries"]))
        push_seconds = max(0.5, float(params["push_seconds"]))

        # scoop_distance used to mean "total travel", and the plunge advance was
        # subtracted out of it — which silently produced a zero-length push when
        # the two happened to be equal. Say so rather than ignoring the key.
        for gone, replacement in (("drag_waypoints", "push_seconds"),
                                  ("seconds_per_waypoint", "push_seconds")):
            if params.get(gone) is not None:
                print(f"[Scoop] NOTE: {gone} is no longer used — the push is one "
                      f"continuous min-jerk motion now, timed by {replacement}.")

        if params.get("scoop_distance") is not None:
            print(f"[Scoop] NOTE: scoop_distance="
                  f"{float(params['scoop_distance']):.3f} is no longer used. The "
                  f"stroke is now dig_advance ({requested_advance:.3f}) for the "
                  f"plunge plus drag_distance ({requested_drag:.3f}) for the "
                  f"push. Pass those instead.")

        print(f"[Scoop] Starting scoop from '{source}'")

        start_width = self.held_width()
        print(f"[Scoop] Holding scoop at {start_width * 1000:.1f}mm")

        # 1. Clear the cameras and scan the source.
        if not self.clear_cameras(params, tag="Scoop"):
            return False, {"error": "Failed to clear the cameras before scanning"}

        self.vision.clear_cache()
        located = self.locate_container(source, force_refresh=True,
                                        category=params.get("container_category"))
        if located is None:
            return False, {"error": f"Cannot locate '{source}'"}

        ok, msg = self.check_still_holding(start_width, tag="Scoop")
        if not ok:
            return False, {"error": f"Lost the scoop before scooping: {msg}"}

        rim_center = located["rim_center"]
        surface_z = located["top_z"]
        rim_radius = located["rim_radius"]
        # Print base_z and height too: a container that segments as a few
        # millimetres tall is almost always the PAPER LABEL under it rather
        # than the container itself, and surface_z alone hides that.
        print(f"[Scoop] '{source}' surface z={surface_z:.4f}, "
              f"base z={located['base_z']:.4f}, "
              f"height {located['height'] * 1000:.0f}mm, "
              f"centre {np.round(rim_center, 4)}, "
              f"opening radius {rim_radius * 1000:.0f}mm")
        if located["height"] < 0.02:
            print(f"[Scoop] WARNING: that is only "
                  f"{located['height'] * 1000:.0f}mm tall. A real cup is "
                  f"~90mm. This is probably the paper label, not the "
                  f"container — check the mask before trusting the depth.")

        # 2. Clamp the whole stroke to the measured opening, scaling the plunge
        # and the push together so their ratio survives the clamp.
        requested_total = requested_advance + requested_drag
        usable = max(0.0, rim_radius - wall_clearance)
        max_total = 2.0 * usable
        if requested_total > max_total and requested_total > 0:
            scale = max_total / requested_total
            dig_advance = requested_advance * scale
            drag_distance = requested_drag * scale
            untilt_over = requested_untilt * scale
            print(f"[Scoop] Clamping stroke {requested_total * 1000:.0f}mm -> "
                  f"{max_total * 1000:.0f}mm (opening radius "
                  f"{rim_radius * 1000:.0f}mm minus {wall_clearance * 1000:.0f}mm "
                  f"clearance): plunge {dig_advance * 1000:.0f}mm, "
                  f"push {drag_distance * 1000:.0f}mm")
        else:
            dig_advance = requested_advance
            drag_distance = requested_drag
            untilt_over = requested_untilt

        total = dig_advance + drag_distance
        if total < 0.008:
            return False, {
                "error": (f"'{source}' opening is only "
                          f"{rim_radius * 2000:.0f}mm across — no room to run a "
                          f"scoop stroke through the powder"),
                "rim_radius": rim_radius,
            }
        # Blending over more than the push itself is meaningless.
        untilt_over = min(untilt_over, drag_distance)

        # 3. Push AWAY from the robot base (+X). Corrected on the bench
        # 2026-09-16: pulling toward the base scrapes powder toward the near
        # wall and the bowl comes out the far side of the stroke empty.
        push = np.array([1.0, 0.0])
        site = np.asarray(rim_center, dtype=float) + np.array([
            float(params["forward_offset"]), float(params["lateral_offset"])])
        if not np.allclose(site, rim_center):
            print(f"[Scoop] Stroke site shifted by "
                  f"forward(X)={float(params['forward_offset']):+.3f}m "
                  f"lateral(Y)={float(params['lateral_offset']):+.3f}m: "
                  f"rim {np.round(rim_center, 4)} -> {np.round(site, 4)}")
        start_xy = site - push * (total / 2.0)
        dig_xy = start_xy + push * dig_advance
        end_xy = start_xy + push * total

        level_rotation = self.tool_down_rotation()
        dig_z = surface_z - scoop_depth
        push_z = dig_z - dip_depth          # where the stroke actually runs
        if push_z < self.workspace_min[2]:
            return False, {
                "error": (f"Dig depth plus dip would put the bowl at "
                          f"z={push_z:.3f}, below the workspace floor "
                          f"{self.workspace_min[2]:.3f}"),
            }

        def rotation_at(angle_deg: float) -> np.ndarray:
            """Wrist rotation for a given forward tilt, 0 = level."""
            return _orthonormalize(level_rotation @ tool_y_delta(-float(angle_deg)))

        # 4. Hover above the start of the stroke, level (free space, strict).
        hover_tip = np.array([start_xy[0], start_xy[1],
                              surface_z + float(params["approach_height"])])
        hover = self.tcp_for_tip(hover_tip, level_rotation, tool_offset)
        print(f"[Scoop] Hovering with the bowl above the powder at "
              f"{np.round(hover_tip, 4)} (TCP {np.round(hover, 4)})...")
        if not self.goto_pose_rigid(hover, level_rotation, duration=3.0):
            return False, {"error": "Failed to command hover pose"}
        arrived, err = self.reached(hover, float(params["hover_tol"]))
        if not arrived:
            return False, {
                "error": (f"Hover above '{source}' unreachable "
                          f"(off by {err * 1000:.0f}mm)"),
            }

        # 5. Tilt FORWARD before touching the powder, and confirm it happened.
        #
        # NEGATED on purpose: tool_tip_deg() is an arccos magnitude (always
        # >= 0), so it cannot tell "tilted forward" from "tilted the wrong way"
        # — only how far from vertical. Bench feedback 2026-09-11 was that the
        # un-negated command tilted 30 deg the wrong way while still reading as
        # "achieved 30 deg" here, because the magnitude was right.
        tilt_before = self.tool_tip_deg()
        print(f"[Scoop] Tilting forward {dig_tilt_deg:.0f}° before entering "
              f"the powder (tip now {tilt_before:.1f}° from vertical)...")

        # Retry the *remaining* angle, not the whole command: frankapy often
        # needs a second, longer goto_pose before the wrist tracks orientation
        # (the same lag pour retries through). Re-commanding the full angle
        # would stack rotations and overshoot.
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
            # level. Fail before entering the powder — we are still in free
            # space, so backing off is cheap.
            self.goto_pose_rigid(hover, level_rotation, duration=2.0)
            return False, {
                "error": (f"Forward dig tilt stalled: commanded {dig_tilt_deg:.0f}° "
                          f"but only achieved {achieved:.1f}° after {retries} "
                          f"attempts (tol {tilt_tol:.0f}°). Pushing level through "
                          f"the powder without this tilt would not collect any "
                          f"onto the scoop."),
                "tilt_commanded": dig_tilt_deg,
                "tilt_achieved": achieved,
            }

        tilted_rotation = rotation_at(dig_tilt_deg)

        # 6. Plunge in, tilted, advancing forward in the same motion. A
        # straight-down drop bites into nothing.
        entry = self.tcp_for_tip(np.array([dig_xy[0], dig_xy[1], dig_z]),
                                 tilted_rotation, tool_offset)
        print(f"[Scoop] Plunging the bowl to z={dig_z:.4f} while advancing "
              f"{dig_advance * 1000:.0f}mm away from the base "
              f"({scoop_depth * 1000:.0f}mm deep, tilted, compliant)...")
        if not self.goto_pose_rigid(entry, tilted_rotation, duration=3.0,
                                    use_impedance=True):
            return False, {"error": "Failed to command dig pose"}
        arrived, err = self.reached(entry, contact_tol)
        if not arrived:
            print(f"[Scoop] Plunge stopped {err * 1000:.0f}mm short — backing out")
            self.goto_pose_rigid(hover, level_rotation, duration=3.0)
            return False, {
                "error": (f"Could not enter the powder in '{source}' "
                          f"(off by {err * 1000:.0f}mm); the scoop is probably "
                          f"fouling the rim"),
            }

        # 6b. Dip: having cut in at the entry angle, drive the bowl a little
        # deeper before the stroke starts. The plunge sets the angle, this sets
        # the depth — keeping them separate means the entry can be made sharper
        # without also making the scoop deeper, and vice versa.
        if dip_depth > 1e-6:
            dip = self.tcp_for_tip(np.array([dig_xy[0], dig_xy[1], push_z]),
                                   tilted_rotation, tool_offset)
            print(f"[Scoop] Dipping a further {dip_depth * 1000:.0f}mm to "
                  f"z={push_z:.4f} before pushing...")
            if not self.goto_pose_rigid(dip, tilted_rotation, duration=2.0,
                                        use_impedance=True):
                return False, {"error": "Failed to command the dip"}
            arrived, err = self.reached(dip, contact_tol)
            if not arrived:
                # The bed resisting is expected; only a gross miss matters.
                print(f"[Scoop] Dip ended {err * 1000:.0f}mm short "
                      f"(powder resistance); continuing")

        # 7. Push away from the base, rolling back to level as it goes.
        #
        # ONE smooth motion, or two at most. frankapy's goto_pose is already a
        # min-jerk interpolation of both translation and orientation, so a
        # single call is a continuous motion — an earlier version subdivided
        # this into eight waypoints and it visibly stepped, because each
        # waypoint decelerated to a stop, not because the controller needed
        # the subdivision. (stir's P11 lesson is about a 50Hz *non-blocking*
        # stream, which is a different failure and does not apply here.)
        #
        # Two segments, because a single min-jerk segment interpolates
        # orientation monotonically end-to-end and so cannot hold the bite
        # angle and then roll off:
        #   A: translate at the full bite angle (no rotation)
        #   B: translate the last untilt_over while rotating to level
        # Set untilt_over == drag_distance to collapse this to one unbroken
        # motion that rolls level across the whole push.
        hold_distance = max(0.0, drag_distance - untilt_over)
        segments = []
        if hold_distance > 1e-4:
            segments.append((hold_distance, dig_tilt_deg, "holding the bite angle"))
        segments.append((drag_distance, 0.0, "rolling back to level"))

        print(f"[Scoop] Pushing {drag_distance * 1000:.0f}mm away from the base "
              f"over {push_seconds:.1f}s in {len(segments)} smooth "
              f"{'motion' if len(segments) == 1 else 'motions'}, un-tilting over "
              f"the last {untilt_over * 1000:.0f}mm...")

        pushed = 0.0
        angle_now = dig_tilt_deg
        for target_s, end_angle, label in segments:
            segment_length = target_s - pushed
            if segment_length <= 1e-6:
                continue
            xy = dig_xy + push * target_s
            # The offset rotates with the wrist, so the TCP target has to be
            # recomputed for THIS segment's end orientation, not fixed once.
            point = self.tcp_for_tip(np.array([xy[0], xy[1], push_z]),
                                     rotation_at(end_angle), tool_offset)
            seconds = push_seconds * (segment_length / drag_distance)
            print(f"[Scoop]   -> {target_s * 1000:.0f}mm, tilt "
                  f"{angle_now:.0f}° -> {end_angle:.0f}° "
                  f"({label}, {seconds:.1f}s)")
            angle_now = end_angle
            if not self.goto_pose_rigid(point, rotation_at(end_angle),
                                        duration=max(0.5, seconds),
                                        use_impedance=True):
                print(f"[Scoop]   push segment failed — stopping here")
                break
            pushed = target_s

        exit_point = self.tcp_for_tip(np.array([end_xy[0], end_xy[1], push_z]),
                                      rotation_at(0.0), tool_offset)
        arrived, err = self.reached(exit_point, contact_tol)
        if not arrived:
            # Not fatal — a partial push still lifts powder.
            print(f"[Scoop] Push ended {err * 1000:.0f}mm short of target "
                  f"(powder resistance); continuing with a partial scoop")

        ok, msg = self.check_still_holding(start_width, tag="Scoop")
        if not ok:
            return False, {"error": f"Scoop lost during the push: {msg}"}

        residual = self.tool_tip_deg()
        print(f"[Scoop] Push done ({pushed * 1000:.0f}mm travelled), "
              f"residual tilt {residual:.1f}° from vertical")
        if residual > tilt_tol:
            # Worth knowing but not worth aborting: the powder is already on
            # the scoop, and a few degrees costs spillage, not the attempt.
            print(f"[Scoop] Did not fully return to level — lifting anyway")

        # 8. Lift straight out, level.
        level_now = _orthonormalize(
            np.asarray(self.get_current_pose().rotation, dtype=float)
        )
        lift = np.asarray(self.get_current_pose().translation, dtype=float).copy()
        lift[2] = surface_z + lift_height + float(tool_offset[2])
        print(f"[Scoop] Lifting clear to z={lift[2]:.4f}...")
        if not self.goto_pose_rigid(lift, level_now, duration=3.0):
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
            "dip_depth": dip_depth,
            "total_depth": scoop_depth + dip_depth,
            "dig_advance": dig_advance,
            "drag_distance": drag_distance,
            "pushed": pushed,
            "untilt_over": untilt_over,
            "total_stroke": total,
            "requested_stroke": requested_total,
            "push_direction": "away_from_base(+X)",
            "rim_radius": rim_radius,
            "surface_z": surface_z,
            "forward_offset": float(params["forward_offset"]),
            "lateral_offset": float(params["lateral_offset"]),
            "tool_offset": [float(v) for v in tool_offset],
            "dig_tilt_commanded": dig_tilt_deg,
            "dig_tilt_achieved": achieved,
            "residual_tilt": residual,
            # There is no measurement of how much powder is on the scoop. The
            # quantity is nominal, set by depth and stroke length.
            "quantity_measured": False,
        }
