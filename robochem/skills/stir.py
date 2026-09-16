"""
Stir Skill

Agitates the contents of a container with a held stirrer, without transferring
anything between containers.

Hardened along the same lines as pick_up / pour:
  - homes first, still holding the tool, so the uprighting tilt has full wrist
    travel — from the pose pick_up leaves out over the bench it does not
  - a 90 deg tilt toward the base stands a flat-grasped spoon up, verified
    against straight down rather than by an angle magnitude
  - reset_joints before scanning, then measure the rim in the world frame
  - circle radius is clamped to the measured opening, so the stirrer cannot
    scrape the wall or knock the cup over
  - each revolution is traced as ONE streamed frankapy skill, so the circle is
    continuous; pauses between revolutions are deliberate. Blocking waypoints
    remain as the fallback wherever streaming is unavailable
  - the tool is confirmed still held before, during and after stirring
"""

from typing import Dict, Any, Tuple, List
import numpy as np

from .base_skill import BaseSkill, _orthonormalize


class StirSkill(BaseSkill):
    """
    Stir the contents of a container with a held stirring tool.

    Pipeline:
    1. Record held width; the stirrer must survive the whole motion
    2. Clear the cameras, scan the container, measure rim height and radius
    3. Tilt toward the base so a flat-grasped spoon stands vertical
    4. Hover over the rim centre, then descend to the immersion depth
    5. Approach the rim of the circle, then trace N revolutions, each one a
       single streamed motion
    6. Lift clear, re-centre, and verify the tool is still held
    """

    name = "stir"
    required_params = ["target_container"]

    @property
    def optional_params(self) -> Dict[str, Any]:
        return {
            "revolutions": 3,          # How many full circles to walk
            "stir_radius": 0.015,      # Requested circle radius (metres)
            "stir_depth": 0.03,        # Immersion below the rim (metres)
            # A circle cannot be one min-jerk move (those interpolate straight
            # lines), so unlike scoop's push this genuinely needs subdividing.
            # More, shorter waypoints read as smoother.
            "waypoints_per_rev": 12,
            "seconds_per_waypoint": 0.4,
            # Time for the one long move that gets the tool from wherever it is
            # to the start of the circle. This is NOT a circle step: in-air runs
            # start at the home pose, which can be 20cm from the circle, and
            # giving that the per-waypoint dwell asks for ~0.5 m/s and the move
            # simply does not happen.
            "approach_seconds": 3.0,
            # Trace each revolution as ONE streamed motion (frankapy dynamic
            # mode) so the circle is continuous instead of stepping between
            # waypoints. Pauses BETWEEN revolutions are fine — each revolution
            # is its own skill. Falls back to blocking waypoints wherever
            # streaming is unavailable (no ROS, dry runs, tests).
            "smooth": True,
            "seconds_per_revolution": 4.0,
            "stream_rate_hz": 50.0,
            # Swing the tool's long axis from horizontal to straight down
            # before stirring. pick_up grasps a spoon lying FLAT — handle along
            # +X, away from the base — so without this the "stirrer" is held
            # horizontally and cannot enter a cup at all.
            # reset_joints to home BEFORE the orientation swing. pick_up
            # leaves the arm out over the object it just grasped, which is the
            # reach where the wrist runs out of range mid-swing; from home it
            # has its full travel. Safe while holding — pour homes with a full
            # beaker in the jaws. Does NOT touch the gripper, unlike
            # run_experiment's --reset flag.
            "home_first": True,
            # Tilt the tool toward the base to bring the handle vertical.
            # pick_up grasps a spoon lying FLAT with the handle pointing away
            # from the base, so a 90deg tilt toward the base drops it to
            # straight down. Same primitive and sign convention as scoop's
            # bite tilt, just a quarter turn instead of 30deg.
            "verticalize": True,
            "verticalize_deg": 90.0,
            "orient_tol_deg": 15.0,
            "orient_retries": 3,
            # Demo / dry-run: skip the scan and stir in free space, so the
            # motion can be watched without a container under the tool.
            "in_air": False,
            "air_height": None,        # Z for the in-air circle; None = current
            "approach_height": 0.10,
            "lift_height": 0.10,
            "reset_before_scan": True,
            # Distance kept between the stirrer and the container wall. The
            # stirrer's own width is unknown to us, so this is deliberately
            # generous — a stirrer that scrapes the wall tips soft cups over.
            "wall_clearance": 0.012,
            # How far the tool tip sits below the gripper TCP, in metres.
            # MEASURE THIS for your stirrer: the arm commands the TCP, not the
            # tip, so an unmeasured tool either dredges the bottom or never
            # touches the liquid.
            "tool_length": 0.0,
            "hover_tol": 0.05,
            "descend_tol": 0.03,
            # Circle waypoints are allowed to run loose: the stirrer is in
            # fluid and mild lag is harmless. Only a gross departure counts.
            "circle_tol": 0.05,
        }

    def check_preconditions(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        # An in-air demo has nothing to aim at, so target_container is not
        # required there — everywhere else it still is.
        if not params.get("in_air", False):
            valid, msg = self.validate_params(params)
            if not valid:
                return False, msg

        if not self.is_gripper_holding():
            return False, "Gripper is not holding anything. Pick up a stirrer first."

        if int(params.get("revolutions", 3)) < 1:
            return False, "revolutions must be at least 1"

        return True, "Preconditions met"

    def execute(self, params: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        params = self.get_params_with_defaults(params)
        target = params.get("target_container")
        revolutions = int(params["revolutions"])
        requested_radius = float(params["stir_radius"])
        stir_depth = float(params["stir_depth"])
        per_rev = max(4, int(params["waypoints_per_rev"]))
        dwell = float(params["seconds_per_waypoint"])
        tool_length = float(params["tool_length"])
        wall_clearance = float(params["wall_clearance"])
        in_air = bool(params["in_air"])
        verticalize = bool(params["verticalize"])
        verticalize_deg = float(params["verticalize_deg"])
        orient_tol = float(params["orient_tol_deg"])
        orient_retries = max(1, int(params["orient_retries"]))

        where = "mid air" if in_air else f"'{target}'"
        print(f"[Stir] Starting stir in {where}: {revolutions} revolutions")

        start_width = self.held_width()
        print(f"[Stir] Holding stirrer at {start_width * 1000:.1f}mm")

        # 0. Home first, still holding the tool. The swing below needs wrist
        # travel that the post-pick pose out over the bench does not have.
        if bool(params["home_first"]):
            print("[Stir] reset_joints (home) before orienting — keeping hold "
                  "of the tool...")
            if not self.go_home():
                return False, {"error": "Failed to reset_joints before orienting"}
            self.wait(0.4)
            ok, msg = self.check_still_holding(start_width, tag="Stir")
            if not ok:
                return False, {"error": f"Lost the stirrer while homing: {msg}"}

        # 1. Stand the spoon up before anything else.
        #
        # pick_up takes a spoon lying flat with a top grasp, which leaves the
        # handle horizontal along +X, pointing away from the base, with tool Z
        # down. A horizontal spoon cannot go into a cup.
        if verticalize:
            # Tilt toward the base about the tool's own closing axis. From the
            # flat grasp (handle out along +X) a quarter turn drops the handle
            # to straight down. Negative for the same reason scoop's bite tilt
            # is negative — the un-negated direction goes the other way.
            #
            # Retry the REMAINING angle, never the original command: the wrist
            # lags, and rotate_about_tool_axis is relative, so re-issuing the
            # full 90deg would stack rotations past vertical.
            #
            # Verified with tool_long_axis_deg, which measures the handle
            # against straight down and separates 0deg (down) from 180deg (up).
            # tool_tip_deg would not — it is an arccos magnitude, the blind
            # spot that let the scoop tilt run backwards while reporting
            # success.
            print(f"[Stir] Tilting {verticalize_deg:.0f}° toward the base to "
                  f"stand the spoon up (handle {self.tool_long_axis_deg():.1f}° "
                  f"from down)...")
            for attempt in range(1, orient_retries + 1):
                remaining = self.tool_long_axis_deg()
                if remaining <= orient_tol:
                    break
                if not self.rotate_about_tool_axis(-remaining, axis="y"):
                    return False, {"error": "Failed to command the tilt"}
                self.wait(0.3)
                print(f"[Stir]   attempt {attempt}/{orient_retries}: handle "
                      f"{self.tool_long_axis_deg():.1f}° from down")

            handle = self.tool_long_axis_deg()
            if handle > orient_tol:
                return False, {
                    "error": (f"Could not stand the spoon up: handle is "
                              f"{handle:.1f}° from straight down after "
                              f"{orient_retries} attempts (tol {orient_tol:.0f}°). "
                              f"{'It is pointing UP, not down. ' if handle > 120 else ''}"
                              f"A horizontal spoon cannot enter a container."),
                    "long_axis_deg": handle,
                }
            print(f"[Stir] Spoon vertical ({handle:.1f}° from down)")

        rotation = _orthonormalize(
            np.asarray(self.get_current_pose().rotation, dtype=float)
        )

        ok, msg = self.check_still_holding(start_width, tag="Stir")
        if not ok:
            return False, {"error": f"Lost the stirrer while orienting: {msg}"}

        # 2. Work out where the circle goes.
        if in_air:
            # Demo mode: no container, no scan. Stir in free space so the
            # motion can be watched. Nothing is clamped because nothing is
            # there to hit — which is exactly why this is not a real stir.
            here = np.asarray(self.get_current_pose().translation, dtype=float)
            air_height = params.get("air_height")
            stir_z = float(air_height) if air_height is not None else float(here[2])
            center = np.array([here[0], here[1]])
            radius = requested_radius
            rim_radius = None
            print(f"[Stir] IN-AIR DEMO — no container, no scan. Circling "
                  f"{radius * 1000:.0f}mm at {np.round(center, 4)}, z={stir_z:.4f}")
        else:
            if not self.clear_cameras(params, tag="Stir"):
                return False, {"error": "Failed to clear the cameras before scanning"}

            self.vision.clear_cache()
            located = self.locate_container(target, force_refresh=True)
            if located is None:
                return False, {"error": f"Cannot locate '{target}'"}

            ok, msg = self.check_still_holding(start_width, tag="Stir")
            if not ok:
                return False, {"error": f"Lost the stirrer before stirring: {msg}"}

            center = located["rim_center"]
            rim_z = located["top_z"]
            rim_radius = located["rim_radius"]
            print(f"[Stir] '{target}' rim centre {np.round(center, 4)}, "
                  f"z={rim_z:.4f}, radius {rim_radius * 1000:.0f}mm")

            max_radius = max(0.0, rim_radius - wall_clearance)
            radius = min(requested_radius, max_radius)
            if radius < requested_radius:
                print(f"[Stir] Clamping stir radius "
                      f"{requested_radius * 1000:.0f}mm -> {radius * 1000:.0f}mm "
                      f"(opening {rim_radius * 1000:.0f}mm minus "
                      f"{wall_clearance * 1000:.0f}mm wall clearance)")
            if radius < 0.003:
                return False, {
                    "error": (f"'{target}' opening is only "
                              f"{rim_radius * 1000:.0f}mm across — no room to "
                              f"stir without hitting the wall"),
                    "rim_radius": rim_radius,
                }

            stir_z = rim_z - stir_depth + tool_length
            if stir_z < self.workspace_min[2]:
                return False, {
                    "error": (f"Stir depth would put the wrist at z={stir_z:.3f}, "
                              f"below the workspace floor "
                              f"{self.workspace_min[2]:.3f}"),
                }

        # 3. Get over the circle, then down into it. Skipped in air: the tool
        # is already at the demo height and there is nothing to descend into.
        hover = np.array([center[0], center[1], stir_z])
        if not in_air:
            hover = np.array([center[0], center[1],
                              rim_z + float(params["approach_height"]) + tool_length])
            print(f"[Stir] Hovering above the opening at {np.round(hover, 4)}...")
            if not self.goto_pose_rigid(hover, rotation, duration=3.0):
                return False, {"error": "Failed to command hover pose"}
            arrived, err = self.reached(hover, float(params["hover_tol"]))
            if not arrived:
                return False, {
                    "error": (f"Hover above '{target}' unreachable "
                              f"(off by {err * 1000:.0f}mm)"),
                }

            entry = np.array([center[0], center[1], stir_z])
            print(f"[Stir] Lowering to z={stir_z:.4f} "
                  f"({stir_depth * 1000:.0f}mm below the rim)...")
            if not self.goto_pose_rigid(entry, rotation, duration=4.0):
                return False, {"error": "Failed to command entry pose"}
            arrived, err = self.reached(entry, float(params["descend_tol"]))
            if not arrived:
                print(f"[Stir] Entry stopped {err * 1000:.0f}mm short — lifting out")
                self.goto_pose_rigid(hover, rotation, duration=3.0)
                return False, {
                    "error": (f"Could not reach the stir depth inside '{target}' "
                              f"(off by {err * 1000:.0f}mm) — the stirrer may be "
                              f"fouling the rim"),
                }

        # 5. Trace the circle.
        #
        # Each revolution is ONE streamed skill, so the trace itself is
        # continuous — a chain of goto_pose calls decelerates to a stop at every
        # waypoint and visibly steps. Stopping between revolutions is fine and
        # is what lets each one be its own skill.
        smooth = bool(params["smooth"])
        approach_seconds = max(0.5, float(params["approach_seconds"]))
        rev_seconds = max(0.5, float(params["seconds_per_revolution"]))
        rate_hz = max(5.0, float(params["stream_rate_hz"]))
        circle_tol = float(params["circle_tol"])
        completed = 0

        # Get to the rim of the circle first, at a duration that suits the
        # distance. In air that trip starts at home and is ~17cm; giving it a
        # circle-step duration asks for ~0.5 m/s and the move does not happen.
        start_point = np.array([center[0] + radius, center[1], stir_z])
        gap = float(np.linalg.norm(
            np.asarray(self.get_current_pose().translation, dtype=float)
            - start_point))
        print(f"[Stir] Approaching the circle start ({gap * 1000:.0f}mm away) "
              f"over {approach_seconds:.1f}s...")
        if not self.goto_pose_rigid(start_point, rotation,
                                    duration=approach_seconds):
            return False, {"error": "Failed to command the circle approach"}
        arrived, err = self.reached(start_point, circle_tol)
        if not arrived:
            return False, {
                "error": (f"Could not reach the start of the circle "
                          f"(off by {err * 1000:.0f}mm)"),
            }

        streamed = False
        if smooth:
            print(f"[Stir] Tracing {revolutions} revolution(s), "
                  f"{rev_seconds:.1f}s each, streamed at {rate_hz:.0f}Hz...")
            streamed = True
            for rev in range(1, revolutions + 1):
                path = self._circle_path(center, radius, stir_z,
                                         rev_seconds, rate_hz)
                ok, msg = self.stream_pose_path(path, rotation,
                                                seconds=rev_seconds,
                                                rate_hz=rate_hz, tag="Stir")
                if not ok:
                    print(f"[Stir] Smooth trace unavailable ({msg}); "
                          f"falling back to waypoints")
                    streamed = False
                    break

                ok, held = self.check_still_holding(start_width, tag="Stir")
                if not ok:
                    print(f"[Stir] {held} — aborting mid-stir")
                    return False, {
                        "error": f"Stirrer lost during stirring: {held}",
                        "revolutions_completed": float(rev - 1),
                    }
                completed = rev * per_rev
                print(f"[Stir]   revolution {rev}/{revolutions} done ({msg})")

        if not streamed:
            waypoints = self._circle_waypoints(center, radius, stir_z,
                                               revolutions, per_rev)
            print(f"[Stir] Stirring: {len(waypoints)} waypoints, "
                  f"radius {radius * 1000:.0f}mm, ~{dwell:.1f}s each")
            completed = 0
            for i, point in enumerate(waypoints):
                if not self.goto_pose_rigid(point, rotation, duration=dwell):
                    print(f"[Stir] Waypoint {i + 1}/{len(waypoints)} command failed")
                    break
                arrived, err = self.reached(point, circle_tol)
                if not arrived:
                    print(f"[Stir] Waypoint {i + 1} off by {err * 1000:.0f}mm "
                          f"(tol {circle_tol * 1000:.0f}mm) — stopping the circle")
                    break
                completed += 1

                # A mid-stir drop is the failure this skill most needs to catch:
                # the remaining waypoints would stir with an empty gripper and
                # still report success.
                if (i + 1) % per_rev == 0:
                    ok, msg = self.check_still_holding(start_width, tag="Stir")
                    if not ok:
                        print(f"[Stir] {msg} — aborting mid-stir")
                        self.goto_pose_rigid(hover, rotation, duration=3.0)
                        return False, {
                            "error": f"Stirrer lost during stirring: {msg}",
                            "waypoints_completed": completed,
                        }

        total_steps = revolutions * per_rev
        revolutions_done = completed / float(per_rev)

        # 6. Lift out along the axis we came in on. In air there is no rim to
        # measure from, so rise from the circle plane itself.
        print(f"[Stir] Lifting out...")
        lift_from = stir_z if in_air else rim_z + tool_length
        lift = np.array([center[0], center[1],
                         lift_from + float(params["lift_height"])])
        if not self.goto_pose_rigid(lift, rotation, duration=3.0):
            return False, {
                "error": "Stirred, but failed to lift the stirrer clear",
                "revolutions_completed": revolutions_done,
            }

        ok, msg = self.check_still_holding(start_width, tag="Stir")
        if not ok:
            return False, {"error": f"Stirrer lost during the lift: {msg}",
                           "revolutions_completed": revolutions_done}

        if completed < total_steps:
            return False, {
                "error": (f"Stir stopped early: {completed}/{total_steps} "
                          f"waypoints ({revolutions_done:.1f} of {revolutions} "
                          f"revolutions)"),
                "revolutions_completed": revolutions_done,
                "waypoints_completed": completed,
                "stir_radius": radius,
            }

        print(f"[Stir] Successfully stirred {where} "
              f"({revolutions} revolutions, {radius * 1000:.0f}mm radius)")
        return True, {
            "stirred_container": target,
            "in_air": in_air,
            "center": [float(center[0]), float(center[1])],
            "verticalized": verticalize,
            "long_axis_deg": self.tool_long_axis_deg(),
            "revolutions": revolutions,
            "revolutions_completed": revolutions_done,
            "stir_radius": radius,
            "requested_radius": requested_radius,
            "rim_radius": rim_radius,
            "stir_depth": stir_depth,
            "smooth": streamed,
            "seconds_per_revolution": rev_seconds if streamed else None,
        }

    def _circle_path(self, center_xy: np.ndarray, radius: float, z: float,
                     seconds: float, rate_hz: float) -> List[np.ndarray]:
        """
        Dense points for one streamed revolution, swept with min-jerk timing.

        Constant angular velocity would jerk at the start and end of every
        revolution, because the tool is at rest either side. Easing the ANGLE
        through a min-jerk profile means the trace accelerates away from rest
        and settles back into it, which is what makes stopping between
        revolutions look deliberate rather than like a stall.
        """
        n = max(2, int(round(seconds * rate_hz)))
        path: List[np.ndarray] = []
        for i in range(n + 1):
            t = seconds * (i / float(n))
            angle = 2.0 * np.pi * self.min_jerk_fraction(t, seconds)
            path.append(np.array([
                center_xy[0] + radius * np.cos(angle),
                center_xy[1] + radius * np.sin(angle),
                z,
            ]))
        return path

    @staticmethod
    def _circle_waypoints(center_xy: np.ndarray, radius: float, z: float,
                          revolutions: int, per_rev: int) -> List[np.ndarray]:
        """
        Discrete points around a circle, as blocking goto_pose targets.

        The earlier implementation streamed non-blocking ``goto_pose`` calls in
        a 50Hz loop. frankapy treats each ``goto_pose`` as a new *skill*, so
        that pattern starts and cancels hundreds of skills a second: the arm
        judders, barely tracks the circle, and any reach check is meaningless.
        Walking a modest number of blocking waypoints is slower but is motion
        that actually happens and can be verified.
        """
        points: List[np.ndarray] = []
        total = revolutions * per_rev
        for i in range(total):
            angle = 2.0 * np.pi * (i / float(per_rev))
            points.append(np.array([
                center_xy[0] + radius * np.cos(angle),
                center_xy[1] + radius * np.sin(angle),
                z,
            ]))
        return points
