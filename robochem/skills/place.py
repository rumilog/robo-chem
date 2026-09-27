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
            # Feel for the seat instead of trusting the computed release height.
            #
            # Set this (newtons) to descend in small steps and stop the moment
            # the tool pushes back that hard, then release there. It removes the
            # two things release_clearance cannot know: how far the held object
            # sticks out below the TCP, which changes with every grasp, and the
            # few millimetres of error in the scanned surface height. Dropping a
            # stirrer into its holder is the case it was written for -- contact
            # means the head has landed on the holder's rim and it is seated.
            #
            # None (the default) keeps the plain position-controlled descent,
            # which is right for setting a beaker down on open bench.
            "stop_force_n": None,
            "probe_above": 0.015,      # start probing this far above the target
            "probe_below": 0.020,      # give up this far below it
            "probe_step": 0.002,
            "probe_seconds": 1.0,      # per step; also what the reading settles over
            # Put a tool back into a holder straight down, e.g. the stirrer.
            # reset_joints first (holding on), level the wrist to fingers-down,
            # cross in XY at the home height, and only then descend vertically.
            # The plain path moves diagonally from wherever the last skill
            # left the arm, which drags a hanging rod through whatever is
            # between there and the holder.
            "vertical_insert": False,
            # The TCP position of the pick this object came from. The executor
            # fills it in when target_location names a remembered pick. With
            # vertical_insert the release is measured from it, not from the
            # centroid: putting the TCP back where it was holding the seated
            # tool re-seats the tool.
            "pick_grasp_tcp": None,
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

        insert = bool(params["vertical_insert"])
        if insert:
            print("[Place] Vertical insert: reset_joints (home) first, holding on...")
            if not self.go_home():
                return False, {"error": "Failed to reset_joints before inserting",
                               "still_holding": True}
            self.wait(0.4)
            # Straight down, keeping the closing axis' yaw, so the rod goes
            # into the bore along its own axis.
            hold_rotation = self.tool_down_rotation()
        else:
            # Keep the orientation the object was picked with. Re-flattening
            # the wrist here would twist a held beaker against the table on
            # the way down; pick_up already left us in a sane top-down pose.
            hold_rotation = np.asarray(self.get_current_pose().rotation, dtype=float)

        target_pos, surface_z, located = self._resolve_target(params, target)
        if target_pos is None:
            return False, {"error": f"Cannot locate target '{target}'"}

        grasp_tcp = params.get("pick_grasp_tcp")
        if insert and grasp_tcp is not None:
            grasp_tcp = np.asarray(grasp_tcp, dtype=float)
            print(f"[Place] Returning the TCP to where it grasped: "
                  f"{np.round(grasp_tcp, 4)} (centroid was "
                  f"{np.round(target_pos, 4)})")
            target_pos = grasp_tcp.copy()
            surface_z = float(grasp_tcp[2])

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
        if insert:
            # Cross at the home height FIRST, so the only motion near the bench
            # is the straight-down one.
            here = np.asarray(self.get_current_pose().translation, dtype=float)
            transit = hover.copy()
            transit[2] = max(float(here[2]), hover[2])
            print(f"[Place] Crossing in XY at z={transit[2]:.4f} to "
                  f"{np.round(transit, 4)}...")
            if not self.goto_pose_rigid(transit, hold_rotation, duration=4.0):
                return False, {"error": "Failed to command the XY transit",
                               "still_holding": True}
            arrived, err = self.reached(transit, hover_tol)
            if not arrived:
                return False, {
                    "error": (f"XY transit above '{target}' unreachable (off by "
                              f"{err * 1000:.0f}mm); still holding the object"),
                    "still_holding": True,
                }
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
        stop_force = params.get("stop_force_n")
        seated_by_force, contact_force, contact_z = False, None, None

        if stop_force is not None:
            if self.ee_wrench() is None:
                return False, {
                    "error": ("stop_force_n was given but this arm reports no "
                              "force; refusing to probe blind"),
                    "still_holding": True,
                }
            stop_force = float(stop_force)
            step = float(params["probe_step"])
            z_from = release_z + float(params["probe_above"])
            z_to = release_z - float(params["probe_below"])
            print(f"[Place] Feeling for the seat: stepping {step * 1000:.0f}mm "
                  f"from z={z_from:.4f} down to z={z_to:.4f}, stopping above "
                  f"{stop_force:.1f}N")

            probe = target_pos.copy()
            # Contact is a CHANGE from a reading at rest above the seat: the
            # raw estimate on the real arm carries a pose-dependent bias of
            # its own, which an absolute threshold would read as contact (a
            # release in mid-air) or mask (no release at all).
            probe[2] = z_from
            if not self.goto_pose_rigid(probe, hold_rotation, duration=2.0):
                self.goto_pose_rigid(hover, hold_rotation, duration=3.0)
                return False, {"error": "Failed to reach the probe start",
                               "still_holding": True}
            self.wait(0.3)
            baseline = self.mean_push_up_n(samples=5, gap=0.05)
            print(f"[Place]   baseline {baseline:+.2f}N at z={z_from:.4f}")
            z = z_from
            while z >= z_to - 1e-9:
                probe[2] = z
                if not self.goto_pose_rigid(probe, hold_rotation,
                                            duration=float(params["probe_seconds"])):
                    self.goto_pose_rigid(hover, hold_rotation, duration=3.0)
                    return False, {"error": "Failed to command a probe step",
                                   "still_holding": True}
                push = self.mean_push_up_n() - baseline
                print(f"[Place]   z={z:.4f}  push={push:+.2f}N")
                if push is not None and push > stop_force:
                    seated_by_force, contact_force, contact_z = True, push, z
                    print(f"[Place] Contact at z={z:.4f} ({push:.2f}N > "
                          f"{stop_force:.1f}N) — seated, releasing here")
                    break
                z -= step

            if not seated_by_force:
                # Never felt anything. Releasing anyway would drop the object
                # from however high the scan happened to put us.
                self.goto_pose_rigid(hover, hold_rotation, duration=3.0)
                return False, {
                    "error": (f"Probed from z={z_from:.4f} to z={z_to:.4f} and "
                              f"never reached {stop_force:.1f}N. Nothing was "
                              f"there to seat against, so the object is still "
                              f"held rather than dropped."),
                    "still_holding": True,
                    "probe_from": z_from,
                    "probe_to": z_to,
                }
            target_pos = probe.copy()

        elif not self.goto_pose_rigid(target_pos, hold_rotation, duration=4.0):
            self.goto_pose_rigid(hover, hold_rotation, duration=3.0)
            return False, {"error": "Failed to command release pose",
                           "still_holding": True}
        # The probe already stopped exactly where it meant to; only the
        # position-controlled descent needs its arrival checked.
        arrived, err = (True, 0.0) if seated_by_force else self.reached(
            target_pos, descend_tol)
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
            "vertical_insert": insert,
            "returned_to_grasp": bool(insert and grasp_tcp is not None),
            "seated_by_force": seated_by_force,
            "contact_force_n": contact_force,
            "contact_z": contact_z,
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
