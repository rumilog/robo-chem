"""
Arc Scoop Skill

A second scooping stroke, kept apart from ``scoop`` so that one keeps working.
``scoop`` drives a straight push through the bed and rolls the bowl level at the
end of it. This one follows the shape a person's hand actually makes: down into
the powder, then a curved sweep that leaves the bed climbing, then straight up.

    hover -> tilt to the bite angle -> plunge straight down into the bed
          -> sweep an ARC: forward and down to the lowest point, then forward
             and up out of the powder, the wrist rolling from nose-down to
             nose-up across the sweep
          -> lift, rolling back to level on the way

The difference that matters is the exit. A straight push leaves the bowl moving
horizontally, so the powder has to be held on by the bowl alone; an arc leaves
it moving up and forward, which is what tips the load to the back of the bowl
and keeps it there. Hence the two knobs worth tuning first, ``exit_angle_deg``
(how steeply the sweep is climbing when it leaves the bed) and ``cup_tilt_deg``
(how far past level the wrist has rolled by then).

The path and the wrist are deliberately **decoupled**. Tying the bowl rigidly to
the arc's radius looks tidy in the maths and wrong in the viewer: a 110 degree
sweep then rotates the wrist 110 degrees, which no hand does. The path comes
from ``arc_radius`` / ``entry_angle_deg`` / ``exit_angle_deg``; the wrist is
interpolated separately from ``dig_tilt_deg`` to ``-cup_tilt_deg``.

Everything around the stroke -- reset before scanning, world-frame rim geometry,
the tilt confirmed by measurement before the powder is touched, compliance for
the contact motions and stiffness for free space, the held-width checks -- is
the same as ``scoop``, because those were each paid for with a hardware run.

**On hardware, read the note on ``sweep_waypoints``.** The arc is subdivided
into blocking waypoints, and frankapy stops at every one of them.
"""

from typing import Dict, Any, List, Tuple

import numpy as np

from .base_skill import BaseSkill, _orthonormalize, tool_y_delta


class ArcScoopSkill(BaseSkill):
    """
    Scoop powder along a curved stroke that exits the bed climbing.

    Pipeline:
    1. Record held width; the scoop must survive the whole motion
    2. Clear the cameras, scan the source, measure rim height / radius
    3. Fit the arc to the measured opening
    4. Hover level above the arc's entry point (free space, verified strictly)
    5. Tilt to the bite angle and confirm it, before any powder is touched
    6. Plunge straight down to the entry point, tilted, compliant
    7. Sweep the arc as a chain of short compliant waypoints, the wrist
       rolling from nose-down to nose-up as it goes
    8. Lift clear, rolling back to level
    """

    name = "arc_scoop"
    required_params = ["powder_source"]

    @property
    def optional_params(self) -> Dict[str, Any]:
        return {
            # --- the arc -------------------------------------------------
            # Radius of the swept path, metres. Clamped down if the opening
            # cannot fit the resulting chord, so this is a request, not a
            # promise. Bigger = flatter, shallower stroke; smaller = tighter
            # and more of a flick.
            "arc_radius": 0.035,
            # Where the sweep starts and ends, as angles around the arc centre
            # measured from straight-down. Negative is behind the low point
            # (the near side), positive is past it (away from the base).
            # entry < 0 < exit. The exit angle IS the climb angle as the bowl
            # leaves the bed: 60 degrees leaves it moving mostly upward.
            "entry_angle_deg": -60.0,
            "exit_angle_deg": 75.0,
            # Depth of the LOWEST point of the arc below the powder surface.
            # The entry and exit sit higher than this by the arc's curvature,
            # so the bowl is only at full depth in the middle of the sweep --
            # which is the point of the shape.
            "scoop_depth": 0.015,
            # Where the powder surface actually is, in world z.
            #
            # Leave it None and the surface is taken as the measured top of the
            # container, which is what `scoop` does -- but that top is the 97th
            # percentile of the CONTAINER's cloud, i.e. the RIM. For a cup that
            # is nearly full those are the same thing; for a part-full one they
            # are not, and the whole stroke then runs in the air above the
            # powder. Watching it in simulation is what turned that up: with a
            # 27mm bed in a 90mm cup the bowl bottomed out 50mm above the
            # powder and the skill still reported success.
            #
            # So: pass the surface when anything knows it (a fill-level
            # estimate, a taught value for a standard cup, a probe), and
            # scoop_depth then means what it says.
            "powder_surface_z": None,
            # How many poses the sweep is cut into. In simulation more is
            # smoother and free. ON HARDWARE each one is a separate frankapy
            # skill that decelerates to a stop, so a high count turns a smooth
            # curve into a visible staircase -- the lesson recorded in
            # ScoopSkill's push comment. Start at 6-8 on the robot.
            "sweep_waypoints": 16,
            "sweep_seconds": 1.6,
            # Carry on along the exit tangent for this far before lifting. A
            # hand does not stop dead at the top of the sweep and then go
            # vertically up; it keeps going the way it was going and the lift
            # grows out of that. Without it the path has a visible corner.
            "extract_distance": 0.035,

            # --- the wrist -----------------------------------------------
            # Nose-down angle at entry: the bite. Same role and same sign
            # convention as ScoopSkill's dig_tilt_deg.
            "dig_tilt_deg": 50.0,
            # Nose-UP angle by the end of the sweep. This is the cupping that
            # carries the powder out; 0 makes the wrist finish level and the
            # stroke read as a plow with a curve in it.
            "cup_tilt_deg": 20.0,
            # How late in the sweep the roll happens. 1.0 rolls at a constant
            # rate from the first millimetre, which starts spilling the load
            # before the bowl has finished digging; higher holds the bite angle
            # through the bed and does most of the roll as the bowl climbs out,
            # which is what a wrist does.
            "roll_late": 2.0,
            # Roll back to level during the lift, the way a hand levels a
            # loaded spoon on the way up. False keeps the cup angle, which is
            # safer for a deep bowl and looks wrong for a shallow one.
            "level_on_lift": True,

            # --- approach and exit ---------------------------------------
            "approach_height": 0.10,
            "lift_height": 0.12,
            "plunge_seconds": 2.0,
            "reset_before_scan": True,

            # --- siting, same convention as scoop / pour / pick_up --------
            "forward_offset": 0.0,
            "lateral_offset": 0.0,
            "wall_clearance": 0.010,
            # The tool's own length along the stroke -- for the printed scoop,
            # the 27.5mm bowl. Reserved at BOTH ends of the sweep, because the
            # path describes the bowl's centre and the rest of it overhangs.
            # Left at 0 the clamp protects a point, and the bowl clips the wall.
            "tool_span": 0.0,

            # --- the tool ------------------------------------------------
            # See BaseSkill.resolve_tool_offset. A cranked scoop needs the
            # full [x, y, z] tool_offset, not a bare length.
            "tool_length": 0.0,
            "tool_offset": None,

            # --- tolerances ----------------------------------------------
            "hover_tol": 0.05,
            "contact_tol": 0.05,
            "tilt_tol_deg": 12.0,
            "tilt_retries": 3,
        }

    def check_preconditions(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        valid, msg = self.validate_params(params)
        if not valid:
            return False, msg
        if not self.is_gripper_holding():
            return False, "Gripper is not holding anything. Pick up a scoop/spoon first."
        return True, "Preconditions met"

    # ------------------------------------------------------------------ arc

    @staticmethod
    def plan_arc(centre_xy: np.ndarray, centre_z: float, radius: float,
                 push: np.ndarray, entry_deg: float, exit_deg: float,
                 count: int, ease: bool = True) -> List[Tuple[float, np.ndarray]]:
        """
        Sample the swept path.

        Returns ``(fraction, tip_xyz)`` pairs from entry to exit, where the
        fraction is how far along the sweep *in time* that point is -- which is
        what the wrist angle is interpolated against, so the roll is timed
        against the motion rather than against the geometry.

        ``ease`` spaces the points by a smoothstep instead of uniformly. Every
        point gets the same slice of the sweep's duration, so clustering them
        near the ends is what gives the stroke a speed profile: it enters the
        bed gently, drives through the middle, and slows as it climbs out.
        Uniform spacing is a constant-speed scrape, which is the single thing
        that most makes a robot scoop look like a robot.
        """
        out = []
        for k in range(count + 1):
            frac = k / count
            shaped = frac * frac * (3.0 - 2.0 * frac) if ease else frac
            phi = np.radians(entry_deg + (exit_deg - entry_deg) * shaped)
            xy = centre_xy + push * (radius * np.sin(phi))
            out.append((frac, np.array([xy[0], xy[1], centre_z - radius * np.cos(phi)])))
        return out

    # -------------------------------------------------------------- execute

    def execute(self, params: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        params = self.get_params_with_defaults(params)
        source = params["powder_source"]

        requested_radius = float(params["arc_radius"])
        entry_deg = float(params["entry_angle_deg"])
        exit_deg = float(params["exit_angle_deg"])
        scoop_depth = float(params["scoop_depth"])
        waypoints = max(2, int(params["sweep_waypoints"]))
        sweep_seconds = max(0.5, float(params["sweep_seconds"]))
        dig_tilt_deg = float(params["dig_tilt_deg"])
        cup_tilt_deg = float(params["cup_tilt_deg"])
        wall_clearance = float(params["wall_clearance"])
        contact_tol = float(params["contact_tol"])
        tilt_tol = float(params["tilt_tol_deg"])
        retries = max(1, int(params["tilt_retries"]))
        lift_height = float(params["lift_height"])
        tool_offset = self.resolve_tool_offset(params)

        if not entry_deg < exit_deg:
            return False, {"error": (f"entry_angle_deg ({entry_deg:.0f}) must be "
                                     f"less than exit_angle_deg ({exit_deg:.0f}); "
                                     f"the sweep runs from the near side to the far "
                                     f"side of the low point")}

        print(f"[ArcScoop] Starting arc scoop from '{source}'")
        start_width = self.held_width()
        print(f"[ArcScoop] Holding scoop at {start_width * 1000:.1f}mm")

        # 1. Clear the cameras and scan the source.
        if not self.clear_cameras(params, tag="ArcScoop"):
            return False, {"error": "Failed to clear the cameras before scanning"}

        self.vision.clear_cache()
        located = self.locate_container(source, force_refresh=True)
        if located is None:
            return False, {"error": f"Cannot locate '{source}'"}

        ok, msg = self.check_still_holding(start_width, tag="ArcScoop")
        if not ok:
            return False, {"error": f"Lost the scoop before scooping: {msg}"}

        rim_center = located["rim_center"]
        surface_z = located["top_z"]
        rim_radius = located["rim_radius"]
        print(f"[ArcScoop] '{source}' surface z={surface_z:.4f}, "
              f"base z={located['base_z']:.4f}, "
              f"height {located['height'] * 1000:.0f}mm, "
              f"centre {np.round(rim_center, 4)}, "
              f"opening radius {rim_radius * 1000:.0f}mm")
        if located["height"] < 0.02:
            print(f"[ArcScoop] WARNING: that is only "
                  f"{located['height'] * 1000:.0f}mm tall — probably the paper "
                  f"label rather than the container. Check the mask.")

        # 2. Fit the arc to the opening. The chord the sweep needs is set by
        # the two angles; shrink the radius until it fits rather than clipping
        # the sweep, so the shape of the stroke survives a small cup.
        span = np.sin(np.radians(exit_deg)) - np.sin(np.radians(entry_deg))
        # Reserve room for the BOWL, not just for a point. The path is the
        # bowl's centre, so half the tool's length hangs off each end of it;
        # clamping without that lets the leading edge ram the far wall while
        # the commanded path still reads as comfortably inside the opening.
        # Measured in a 65mm dish: 4158 contacts against the far wall, all in
        # the last 0.7s of the sweep, with the path itself 3mm clear.
        tool_span = max(0.0, float(params["tool_span"]))
        usable = max(0.0, 2.0 * (rim_radius - wall_clearance) - tool_span)
        chord = requested_radius * span
        if chord > usable and span > 1e-6:
            radius = usable / span
            print(f"[ArcScoop] Arc radius {requested_radius * 1000:.0f}mm -> "
                  f"{radius * 1000:.0f}mm so the {chord * 1000:.0f}mm chord fits "
                  f"the {usable * 1000:.0f}mm of clear opening")
        else:
            radius = requested_radius
        if radius * span < 0.008:
            return False, {
                "error": (f"'{source}' opening is only {rim_radius * 2000:.0f}mm "
                          f"across — no room to sweep an arc through the powder"),
                "rim_radius": rim_radius,
            }

        # 3. Site the arc. The LOW POINT of the arc sits at the stroke centre,
        # so the deepest part of the sweep is in the middle of the cup rather
        # than against a wall.
        push = np.array([1.0, 0.0])            # away from the base, as scoop
        # Centre the SWEPT CHORD on the cup, not the arc's lowest point.
        #
        # The sweep is not symmetric about its low point -- -60 to +75 degrees
        # reaches sin(75)R forward and only sin(60)R back -- so siting the low
        # point at the centre pushes the whole stroke away from the base and
        # into the far wall. Subtracting the chord's own midpoint puts equal
        # room at both ends, which is where a hand would put it.
        chord_mid = radius * (np.sin(np.radians(exit_deg))
                              + np.sin(np.radians(entry_deg))) / 2.0
        site = (np.asarray(rim_center, dtype=float)
                - push * chord_mid
                + np.array([float(params["forward_offset"]),
                            float(params["lateral_offset"])]))
        if abs(chord_mid) > 1e-4:
            print(f"[ArcScoop] Centring the chord moves the stroke "
                  f"{-chord_mid * 1000:+.1f}mm in X (toward the base)")
        if not np.allclose(site, rim_center):
            print(f"[ArcScoop] Stroke site shifted by "
                  f"forward(X)={float(params['forward_offset']):+.3f}m "
                  f"lateral(Y)={float(params['lateral_offset']):+.3f}m")

        # The depth reference: the powder, when it is known, and otherwise the
        # measured container top with a warning, because those are only the
        # same thing in a full cup.
        rim_z = surface_z
        override = params.get("powder_surface_z")
        if override is not None:
            surface_z = float(override)
            print(f"[ArcScoop] Powder surface given as z={surface_z:.4f}, "
                  f"{(rim_z - surface_z) * 1000:.0f}mm below the measured rim "
                  f"({rim_z:.4f})")
        elif rim_z - located["base_z"] > 0.04:
            print(f"[ArcScoop] NOTE: digging {scoop_depth * 1000:.0f}mm below the "
                  f"measured TOP of '{source}' ({rim_z:.4f}), which is its rim, "
                  f"not its contents. In a part-full container the stroke will "
                  f"run above the powder — pass powder_surface_z if you know it.")

        floor_z = surface_z - scoop_depth
        centre_z = floor_z + radius              # arc centre, above the bed
        if floor_z < self.workspace_min[2]:
            return False, {
                "error": (f"Arc bottom at z={floor_z:.3f} is below the workspace "
                          f"floor {self.workspace_min[2]:.3f}"),
            }
        if floor_z < located["base_z"]:
            print(f"[ArcScoop] NOTE: the arc bottom ({floor_z:.4f}) is below the "
                  f"measured container base ({located['base_z']:.4f}); the bowl "
                  f"will be riding on the bottom of the cup.")

        path = self.plan_arc(site, centre_z, radius, push,
                             entry_deg, exit_deg, waypoints)
        entry_tip = path[0][1]
        exit_tip = path[-1][1]
        print(f"[ArcScoop] Arc r={radius * 1000:.0f}mm from {entry_deg:+.0f}° to "
              f"{exit_deg:+.0f}°: enters at z={entry_tip[2]:.4f} "
              f"({(surface_z - entry_tip[2]) * 1000:+.0f}mm below the surface), "
              f"bottoms at z={floor_z:.4f}, leaves at z={exit_tip[2]:.4f} "
              f"climbing {exit_deg:.0f}° over {(exit_tip[0] - entry_tip[0]) * 1000:.0f}mm")

        level_rotation = self.tool_down_rotation()

        def rotation_at(angle_deg: float) -> np.ndarray:
            """Wrist rotation for a forward tilt: positive is nose-down."""
            return _orthonormalize(level_rotation @ tool_y_delta(-float(angle_deg)))

        roll_late = max(0.2, float(params["roll_late"]))

        def tilt_at(frac: float) -> float:
            """Nose-down angle at this point of the sweep, rolling to nose-up."""
            return dig_tilt_deg + (-cup_tilt_deg - dig_tilt_deg) * frac ** roll_late

        # 4. Hover above the entry point, level and strict.
        hover_tip = np.array([entry_tip[0], entry_tip[1],
                              rim_z + float(params["approach_height"])])
        hover = self.tcp_for_tip(hover_tip, level_rotation, tool_offset)
        print(f"[ArcScoop] Hovering with the bowl above the entry point at "
              f"{np.round(hover_tip, 4)}...")
        if not self.goto_pose_rigid(hover, level_rotation, duration=3.0):
            return False, {"error": "Failed to command hover pose"}
        arrived, err = self.reached(hover, float(params["hover_tol"]))
        if not arrived:
            return False, {"error": (f"Hover above '{source}' unreachable "
                                     f"(off by {err * 1000:.0f}mm)")}

        # 5. Take the bite angle in free space and confirm it by measurement.
        #
        # Negated for the reason ScoopSkill documents at length: tool_tip_deg()
        # is an arccos magnitude, so it reports a tilt in the WRONG direction as
        # a clean success. Only the sign of the command distinguishes them.
        tilt_before = self.tool_tip_deg()
        print(f"[ArcScoop] Tilting to the {dig_tilt_deg:.0f}° bite angle before "
              f"entering the powder (tip now {tilt_before:.1f}° from vertical)...")
        achieved = 0.0
        for attempt in range(1, retries + 1):
            remaining = dig_tilt_deg - achieved
            if remaining <= 0.5:
                break
            if not self.rotate_about_tool_axis(-remaining, axis="y"):
                return False, {"error": "Failed to command the bite tilt"}
            self.wait(0.3)
            achieved = self.tool_tip_deg() - tilt_before
            print(f"[ArcScoop]   attempt {attempt}/{retries}: tilt "
                  f"{achieved:+.1f}° of {dig_tilt_deg:.0f}°")
            if abs(achieved) >= dig_tilt_deg - tilt_tol:
                break
        if abs(achieved) < dig_tilt_deg - tilt_tol:
            self.goto_pose_rigid(hover, level_rotation, duration=2.0)
            return False, {
                "error": (f"Bite tilt stalled: commanded {dig_tilt_deg:.0f}° but "
                          f"achieved {achieved:.1f}° after {retries} attempts "
                          f"(tol {tilt_tol:.0f}°). Sweeping level collects nothing."),
                "tilt_commanded": dig_tilt_deg,
                "tilt_achieved": achieved,
            }
        print(f"[ArcScoop] Measured bite tilt {self.tool_tip_deg():.1f}°")

        # 5b. Re-hover with the bite angle applied, so the bowl -- not the TCP
        # -- is directly above the entry point.
        #
        # Taking the hover level and then tilting swings the bowl out on the end
        # of the tool, so the descent that follows is a diagonal that crosses
        # the rim on its way in. Filming it made that obvious: the bowl tracked
        # in from outside the cup rather than dropping into it. Re-aiming here
        # costs one short free-space move and makes the plunge vertical.
        entry_rotation = rotation_at(tilt_at(0.0))
        aimed_tip = np.array([entry_tip[0], entry_tip[1], hover_tip[2]])
        aimed = self.tcp_for_tip(aimed_tip, entry_rotation, tool_offset)
        print(f"[ArcScoop] Re-aiming the tilted bowl over the entry point...")
        if not self.goto_pose_rigid(aimed, entry_rotation, duration=1.5):
            return False, {"error": "Failed to aim the tilted bowl over the entry"}

        # 6. Plunge straight down onto the entry point. Straight down, not
        # along the arc: the descent is what the user sees as "going in", and
        # curving it as well makes the entry read as a skid.
        entry = self.tcp_for_tip(entry_tip, entry_rotation, tool_offset)
        print(f"[ArcScoop] Plunging to the entry point z={entry_tip[2]:.4f} "
              f"(tilted {tilt_at(0.0):.0f}°, compliant)...")
        if not self.goto_pose_rigid(entry, entry_rotation,
                                    duration=max(0.5, float(params["plunge_seconds"])),
                                    use_impedance=True):
            return False, {"error": "Failed to command the plunge"}
        arrived, err = self.reached(entry, contact_tol)
        if not arrived:
            print(f"[ArcScoop] Plunge stopped {err * 1000:.0f}mm short — backing out")
            self.goto_pose_rigid(hover, level_rotation, duration=3.0)
            return False, {
                "error": (f"Could not enter the powder in '{source}' (off by "
                          f"{err * 1000:.0f}mm); the scoop is probably fouling "
                          f"the rim"),
            }

        # 7. Sweep the arc. Compliant, because every one of these is a contact
        # motion, and short: the segment time is the sweep time shared out, so
        # raising sweep_waypoints makes the curve finer without making it slower.
        print(f"[ArcScoop] Sweeping the arc in {waypoints} steps over "
              f"{sweep_seconds:.1f}s, wrist {dig_tilt_deg:.0f}° nose-down -> "
              f"{cup_tilt_deg:.0f}° nose-up...")

        rotations = [rotation_at(tilt_at(frac)) for frac, _ in path[1:]]
        tcps = [self.tcp_for_tip(tip, R, tool_offset)
                for (_, tip), R in zip(path[1:], rotations)]

        # ONE continuous motion if the arm can do it. A scoop is a single
        # gesture; a chain of blocking waypoints decelerates to a standstill at
        # each one, and filming that is what showed the stroke reading as a
        # staircase rather than a scoop -- 2.5s of commanded sweep took 26s and
        # stopped 16 times. Streaming keeps the wrist rolling through the bed.
        swept = 0
        streamed, why = self.stream_pose_path(
            tcps, rotations, seconds=sweep_seconds, tag="ArcScoop",
            cartesian_impedance=True)
        if streamed:
            swept = waypoints
            print(f"[ArcScoop]   swept as one continuous motion ({why})")
        else:
            # Fallback: discrete waypoints. Correct, just not smooth.
            print(f"[ArcScoop]   continuous sweep unavailable ({why}); "
                  f"falling back to {waypoints} blocking waypoints")
            seg_seconds = sweep_seconds / waypoints
            for (frac, _), rotation, point in zip(path[1:], rotations, tcps):
                if not self.goto_pose_rigid(point, rotation,
                                            duration=max(0.05, seg_seconds),
                                            use_impedance=True):
                    print(f"[ArcScoop]   sweep stopped at {frac * 100:.0f}% of the arc")
                    break
                swept += 1

        exit_rotation = rotation_at(tilt_at(1.0))
        exit_pose = self.tcp_for_tip(exit_tip, exit_rotation, tool_offset)
        arrived, err = self.reached(exit_pose, contact_tol)
        if not arrived:
            print(f"[ArcScoop] Sweep ended {err * 1000:.0f}mm short of the exit "
                  f"(powder resistance); continuing with a partial scoop")

        ok, msg = self.check_still_holding(start_width, tag="ArcScoop")
        if not ok:
            return False, {"error": f"Scoop lost during the sweep: {msg}"}

        residual = self.tool_tip_deg()
        print(f"[ArcScoop] Sweep done ({swept}/{waypoints} steps), bowl "
              f"{residual:.1f}° from vertical")

        # 7b. Carry on the way the sweep was going before turning the motion
        # into a lift. The exit tangent is perpendicular to the arc's radius,
        # so at exit_angle_deg the bowl is already travelling up and forward;
        # continuing along it for a few centimetres is what removes the corner
        # between "scooping" and "lifting" and is the difference between a
        # gesture and two moves played back to back.
        extract = max(0.0, float(params["extract_distance"]))
        exit_rad = np.radians(exit_deg)
        tangent = np.array([push[0] * np.cos(exit_rad), push[1] * np.cos(exit_rad),
                            np.sin(exit_rad)])
        if extract > 1e-4:
            extract_tip = exit_tip + tangent * extract
            extract_pose = self.tcp_for_tip(extract_tip, exit_rotation, tool_offset)
            print(f"[ArcScoop] Carrying {extract * 1000:.0f}mm along the exit "
                  f"tangent ({exit_deg:.0f}° up) to z={extract_tip[2]:.4f}...")
            if not self.goto_pose_rigid(extract_pose, exit_rotation, duration=1.0,
                                        use_impedance=True):
                print("[ArcScoop]   extract move failed; lifting from here")
            else:
                exit_tip = extract_tip

        # 8. Lift clear, levelling on the way up if asked.
        lift_rotation = level_rotation if params["level_on_lift"] else exit_rotation
        lift_tip = np.array([exit_tip[0], exit_tip[1], rim_z + lift_height])
        lift = self.tcp_for_tip(lift_tip, lift_rotation, tool_offset)
        print(f"[ArcScoop] Lifting clear to z={lift_tip[2]:.4f}"
              + (", rolling back to level..." if params["level_on_lift"] else "..."))
        if not self.goto_pose_rigid(lift, lift_rotation, duration=3.0):
            return False, {"error": "Failed to lift the scoop clear"}
        arrived, err = self.reached(lift, float(params["hover_tol"]))
        if not arrived:
            return False, {"error": f"Scoop did not lift clear (off by {err * 1000:.0f}mm)"}

        ok, msg = self.check_still_holding(start_width, tag="ArcScoop")
        if not ok:
            return False, {"error": f"Scoop lost during the lift: {msg}"}

        # The bed changed shape, so the cached source cloud is stale.
        self.vision.clear_cache()

        print(f"[ArcScoop] Successfully arc-scooped from '{source}'")
        return True, {
            "scooped_from": source,
            "arc_radius": radius,
            "requested_radius": requested_radius,
            "entry_angle_deg": entry_deg,
            "exit_angle_deg": exit_deg,
            "chord": float(radius * span),
            "scoop_depth": scoop_depth,
            "entry_z": float(entry_tip[2]),
            "bottom_z": float(floor_z),
            "exit_z": float(exit_tip[2]),
            "sweep_waypoints": waypoints,
            "swept_waypoints": swept,
            "dig_tilt_commanded": dig_tilt_deg,
            "dig_tilt_achieved": achieved,
            "cup_tilt_deg": cup_tilt_deg,
            "residual_tilt": residual,
            "surface_z": surface_z,
            "rim_radius": rim_radius,
            "tool_offset": [float(v) for v in tool_offset],
            # Nothing here measures how much powder came out.
            "quantity_measured": False,
        }
