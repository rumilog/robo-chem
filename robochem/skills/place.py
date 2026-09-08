"""
Place Skill

Sets a held object down and releases it — the counterpart to pick_up, and the
step that frees the single gripper before a different tool can be used.

Hardened along the same lines as pick_up / pour:
  - reset_joints before scanning, so the arm is not occluding the cage
  - world-frame rim height from the pointcloud, not PCA dimensions[2]
  - every motion's arrival is verified; the gripper is NEVER opened from a
    pose we did not actually reach, because that drops the object from height
  - release is confirmed by re-reading the gripper, not assumed
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np

from .base_skill import BaseSkill


class PlaceSkill(BaseSkill):
    """
    Place a held object at a target location.

    Pipeline:
    1. Record the held width, so slip can be detected later
    2. Clear the cameras (reset_joints) and scan the target
    3. Measure the support surface top from the pointcloud
    4. Hover above the target at safe height, keeping the held orientation
    5. Descend to release height (verified)
    6. Open gripper, confirm release
    7. Retract straight up and invalidate the vision cache
    """

    name = "place"
    required_params = ["target_location"]

    @property
    def optional_params(self) -> Dict[str, Any]:
        return {
            # Clearance between the bottom of the held object and the surface
            # at the moment of release. Small drops are fine and much safer
            # than driving the object into the table.
            "release_clearance": 0.02,
            "approach_height": 0.15,
            "offset": [0.0, 0.0, 0.0],  # XYZ nudge from the located target
            # Bare table height in the robot base frame, used when the target
            # is given as a name we cannot see or as an XY-only coordinate.
            "table_z": 0.02,
            "reset_before_scan": True,
            # Arrival tolerances (metres). The descend tolerance is tight: this
            # is the one motion where being 4cm high means dropping the object.
            "hover_tol": 0.05,
            "descend_tol": 0.025,
            # Place *beside* rather than *on top of* a located object. Placing
            # onto a container is almost never what a chemistry step wants; the
            # default drops the object next to it on the bench.
            "on_top": False,
            "beside_offset": 0.10,
        }

    def check_preconditions(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        valid, msg = self.validate_params(params)
        if not valid:
            return False, msg

        if not self.is_gripper_holding():
            return False, "Gripper is not holding anything. Pick up an object first."

        target = params["target_location"]
        if not isinstance(target, (str, list, tuple, np.ndarray)):
            return False, f"Invalid target_location type: {type(target).__name__}"

        return True, "Preconditions met"

    def execute(self, params: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        params = self.get_params_with_defaults(params)
        target = params["target_location"]
        release_clearance = float(params["release_clearance"])
        approach_height = float(params["approach_height"])
        offset = np.asarray(params["offset"], dtype=float)
        hover_tol = float(params["hover_tol"])
        descend_tol = float(params["descend_tol"])

        print(f"[Place] Starting place at '{target}'")

        start_width = self.held_width()
        print(f"[Place] Holding at {start_width * 1000:.1f}mm at start")

        # Keep the orientation the object was picked with. Re-flattening the
        # wrist here would twist a held beaker against the table on the way
        # down; pick_up already left us in a sane top-down pose.
        hold_rotation = np.asarray(self.get_current_pose().rotation, dtype=float)

        target_pos, surface_z, located = self._resolve_target(params, target)
        if target_pos is None:
            return False, {"error": f"Cannot locate target '{target}'"}

        # After the scan we are at home; confirm the object survived the trip.
        ok, msg = self.check_still_holding(start_width, tag="Place")
        if not ok:
            return False, {"error": f"Lost the object before placing: {msg}"}

        target_pos = target_pos + offset
        release_z = surface_z + release_clearance
        target_pos[2] = release_z
        print(f"[Place] Release site {np.round(target_pos, 4)} "
              f"(surface z={surface_z:.4f} + {release_clearance * 1000:.0f}mm)")

        # Step 1: hover well above the release site.
        hover = target_pos.copy()
        hover[2] = max(release_z + approach_height, self.safe_height)
        print(f"[Place] Hovering at {np.round(hover, 4)}...")
        if not self.goto_pose_rigid(hover, hold_rotation, duration=3.0):
            return False, {"error": "Failed to command hover pose"}
        arrived, err = self.reached(hover, hover_tol)
        if not arrived:
            return False, {
                "error": (f"Hover above '{target}' unreachable "
                          f"(off by {err * 1000:.0f}mm); still holding the object"),
                "still_holding": True,
            }

        # Step 2: descend. If this fails we go back up and keep hold of the
        # object rather than releasing it into thin air.
        print(f"[Place] Descending to release height...")
        if not self.goto_pose_rigid(target_pos, hold_rotation, duration=4.0):
            self.goto_pose_rigid(hover, hold_rotation, duration=3.0)
            return False, {"error": "Failed to command release pose",
                           "still_holding": True}
        arrived, err = self.reached(target_pos, descend_tol)
        if not arrived:
            print(f"[Place] Descent stopped {err * 1000:.0f}mm short "
                  f"(tol {descend_tol * 1000:.0f}mm) — NOT releasing")
            self.goto_pose_rigid(hover, hold_rotation, duration=3.0)
            return False, {
                "error": (f"Release pose not reached (off by {err * 1000:.0f}mm); "
                          f"refusing to open the gripper mid-air"),
                "still_holding": True,
            }

        # Step 3: release, then verify it actually let go.
        print(f"[Place] Opening gripper to release...")
        if not self.open_gripper():
            return False, {"error": "Failed to open gripper", "still_holding": True}
        self.wait(0.4)

        released_width = self.held_width()
        if released_width < 0.06:
            print(f"[Place] Gripper only reached {released_width * 1000:.1f}mm — "
                  f"retrying the open")
            self.open_gripper()
            self.wait(0.4)
            released_width = self.held_width()
            if released_width < 0.06:
                return False, {
                    "error": (f"Gripper did not open (stuck at "
                              f"{released_width * 1000:.1f}mm); object may still "
                              f"be held"),
                    "gripper_width": released_width,
                    "still_holding": True,
                }

        # Step 4: retract straight up, clear of whatever we just set down.
        print(f"[Place] Retracting...")
        retract = target_pos.copy()
        retract[2] = release_z + approach_height
        if not self.goto_pose_rigid(retract, hold_rotation, duration=3.0):
            print("[Place] Warning: retract command failed; object was released")

        # The bench changed, so every cached segmentation is now stale.
        self.vision.clear_cache()

        print(f"[Place] Successfully placed object at '{target}'")
        return True, {
            "placed_at": target if isinstance(target, str) else "coordinates",
            "place_position": target_pos.tolist(),
            "surface_z": surface_z,
            "release_width": released_width,
            "held_width_at_start": start_width,
            "located_target": located is not None,
        }

    def _resolve_target(
        self, params: Dict[str, Any], target
    ) -> Tuple[Optional[np.ndarray], float, Optional[Dict[str, Any]]]:
        """
        Turn ``target_location`` into an XYZ site plus the surface height.

        Returns ``(position, surface_z, located_or_None)``. Explicit coordinates
        skip the scan entirely — that is the escape hatch for "put it back on
        the bench at a spot nothing is occupying".
        """
        table_z = float(params["table_z"])

        if isinstance(target, (list, tuple, np.ndarray)):
            coords = np.asarray(target, dtype=float)
            if coords.shape[0] == 2:
                return np.array([coords[0], coords[1], table_z]), table_z, None
            if coords.shape[0] != 3:
                print(f"[Place] Expected 2 or 3 coordinates, got {coords.shape[0]}")
                return None, table_z, None
            # An explicit Z is the surface the caller wants to release above.
            return coords.copy(), float(coords[2]), None

        if not self.clear_cameras(params, tag="Place"):
            print("[Place] Failed to clear the cameras before scanning")
            return None, table_z, None

        self.vision.clear_cache()
        located = self.locate_container(target, force_refresh=True)
        if located is None:
            return None, table_z, None

        print(f"[Place] '{target}' rim at z={located['top_z']:.4f}, "
              f"base z={located['base_z']:.4f}, "
              f"radius {located['rim_radius'] * 1000:.0f}mm")

        if params.get("on_top", False):
            # Stack onto the located object: release above its top surface.
            site = np.array([located["rim_center"][0], located["rim_center"][1], 0.0])
            return site, located["top_z"], located

        # Default: set the object down on the bench *beside* the target, on the
        # side nearer the robot base so the placement stays inside the reachable
        # workspace.
        beside = float(params["beside_offset"])
        site = np.array([located["rim_center"][0] - beside,
                         located["rim_center"][1],
                         0.0])
        print(f"[Place] Placing beside '{target}' at {beside * 100:.0f}cm "
              f"toward the base (set on_top=true to stack instead)")
        return site, min(located["base_z"], table_z), located
