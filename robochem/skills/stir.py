"""
Stir Skill

Agitates the contents of a container with a held stirrer, without transferring
anything between containers.

Hardened along the same lines as pick_up / pour:
  - reset_joints before scanning, then measure the rim in the world frame
  - circle radius is clamped to the measured opening, so the stirrer cannot
    scrape the wall or knock the cup over
  - discrete blocking waypoints instead of a 50Hz stream of non-blocking
    goto_pose calls (see _circle_waypoints for why)
  - the tool is confirmed still held before, during and after stirring
"""

from typing import Dict, Any, Tuple, List
import numpy as np

from .base_skill import BaseSkill


class StirSkill(BaseSkill):
    """
    Stir the contents of a container with a held stirring tool.

    Pipeline:
    1. Record held width; the stirrer must survive the whole motion
    2. Clear the cameras, scan the container, measure rim height and radius
    3. Flatten the wrist so the stirrer hangs vertically
    4. Hover over the rim centre, then descend to the immersion depth
    5. Walk a circle as discrete blocking waypoints, for N revolutions
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
            "waypoints_per_rev": 8,    # Discrete stops around each circle
            "seconds_per_waypoint": 0.6,
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
        target = params["target_container"]
        revolutions = int(params["revolutions"])
        requested_radius = float(params["stir_radius"])
        stir_depth = float(params["stir_depth"])
        per_rev = max(4, int(params["waypoints_per_rev"]))
        dwell = float(params["seconds_per_waypoint"])
        tool_length = float(params["tool_length"])
        wall_clearance = float(params["wall_clearance"])

        print(f"[Stir] Starting stir in '{target}': {revolutions} revolutions")

        start_width = self.held_width()
        print(f"[Stir] Holding stirrer at {start_width * 1000:.1f}mm")

        # 1. Clear the cameras and scan.
        if not self.clear_cameras(params, tag="Stir"):
            return False, {"error": "Failed to clear the cameras before scanning"}

        self.vision.clear_cache()
        located = self.locate_container(target, force_refresh=True)
        if located is None:
            return False, {"error": f"Cannot locate '{target}'"}

        ok, msg = self.check_still_holding(start_width, tag="Stir")
        if not ok:
            return False, {"error": f"Lost the stirrer before stirring: {msg}"}

        rim_center = located["rim_center"]
        rim_z = located["top_z"]
        rim_radius = located["rim_radius"]
        print(f"[Stir] '{target}' rim centre {np.round(rim_center, 4)}, "
              f"z={rim_z:.4f}, radius {rim_radius * 1000:.0f}mm")

        # 2. Clamp the circle to what the opening can actually take.
        max_radius = max(0.0, rim_radius - wall_clearance)
        radius = min(requested_radius, max_radius)
        if radius < requested_radius:
            print(f"[Stir] Clamping stir radius {requested_radius * 1000:.0f}mm -> "
                  f"{radius * 1000:.0f}mm (opening {rim_radius * 1000:.0f}mm minus "
                  f"{wall_clearance * 1000:.0f}mm wall clearance)")
        if radius < 0.003:
            return False, {
                "error": (f"'{target}' opening is only "
                          f"{rim_radius * 1000:.0f}mm across — no room to stir "
                          f"without hitting the wall"),
                "rim_radius": rim_radius,
            }

        # 3. Vertical tool, TCP raised by the tool length so the *tip* ends up
        # at the requested depth rather than the gripper.
        rotation = self.tool_down_rotation()
        stir_z = rim_z - stir_depth + tool_length
        if stir_z < self.workspace_min[2]:
            return False, {
                "error": (f"Stir depth would put the wrist at z={stir_z:.3f}, "
                          f"below the workspace floor {self.workspace_min[2]:.3f}"),
            }

        hover = np.array([rim_center[0], rim_center[1],
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

        # 4. Descend into the container.
        entry = np.array([rim_center[0], rim_center[1], stir_z])
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

        # 5. Walk the circle.
        waypoints = self._circle_waypoints(rim_center, radius, stir_z,
                                           revolutions, per_rev)
        print(f"[Stir] Stirring: {len(waypoints)} waypoints, "
              f"radius {radius * 1000:.0f}mm, ~{dwell:.1f}s each")

        circle_tol = float(params["circle_tol"])
        completed = 0
        for i, point in enumerate(waypoints):
            if not self.goto_pose_rigid(point, rotation, duration=dwell):
                print(f"[Stir] Waypoint {i + 1}/{len(waypoints)} command failed")
                break
            arrived, err = self.reached(point, circle_tol)
            if not arrived:
                # Not a hard failure on its own — but a big miss mid-circle
                # means the stirrer is jammed against something.
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

        revolutions_done = completed / float(per_rev)

        # 6. Lift out along the axis we came in on.
        print(f"[Stir] Lifting out...")
        lift = np.array([rim_center[0], rim_center[1],
                         rim_z + float(params["lift_height"]) + tool_length])
        if not self.goto_pose_rigid(lift, rotation, duration=3.0):
            return False, {
                "error": "Stirred, but failed to lift the stirrer clear",
                "revolutions_completed": revolutions_done,
            }

        ok, msg = self.check_still_holding(start_width, tag="Stir")
        if not ok:
            return False, {"error": f"Stirrer lost during the lift: {msg}",
                           "revolutions_completed": revolutions_done}

        if completed < len(waypoints):
            return False, {
                "error": (f"Stir stopped early: {completed}/{len(waypoints)} "
                          f"waypoints ({revolutions_done:.1f} of {revolutions} "
                          f"revolutions)"),
                "revolutions_completed": revolutions_done,
                "waypoints_completed": completed,
                "stir_radius": radius,
            }

        print(f"[Stir] Successfully stirred '{target}' "
              f"({revolutions} revolutions, {radius * 1000:.0f}mm radius)")
        return True, {
            "stirred_container": target,
            "revolutions": revolutions,
            "revolutions_completed": revolutions_done,
            "stir_radius": radius,
            "requested_radius": requested_radius,
            "rim_radius": rim_radius,
            "stir_depth": stir_depth,
            "waypoints": len(waypoints),
        }

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
