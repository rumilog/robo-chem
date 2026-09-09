"""
Pick Up Skill

Picks up an object from the workspace using SAM segmentation,
pointcloud localization, and grasp analysis.
"""

from typing import Dict, Any, Tuple, List
import numpy as np

from .base_skill import BaseSkill


def width_along_closing_axis(points, grasp_pose, slab=0.01,
                             lo_pct=3, hi_pct=97, max_radius=0.06):
    """
    How wide the object is where the fingers will actually meet it.

    Measured along the gripper's closing axis (its local y), over a thin
    horizontal slab at the grasp height. Points farther than ``max_radius``
    from the grasp XY are dropped so a bloated SAM cloud (table, neighbour
    objects) cannot invent a 70mm+ "beaker".
    """
    centre = grasp_pose[:3, 3]
    closing = grasp_pose[:3, 1]

    horizontal = closing[:2]
    norm = np.linalg.norm(horizontal)
    if norm < 1e-6:
        return None
    horizontal = horizontal / norm

    # Keep only a neighbourhood of the grasp — fused clouds often include
    # far-away surface from a fat mask.
    xy_dist = np.linalg.norm(points[:, :2] - centre[:2], axis=1)
    near = points[(xy_dist < max_radius) & (np.abs(points[:, 2] - centre[2]) < slab)]
    if len(near) < 20:
        near = points[np.abs(points[:, 2] - centre[2]) < slab]
    if len(near) < 20:
        return None

    projected = (near[:, :2] - centre[:2]) @ horizontal
    return float(np.percentile(projected, hi_pct)
                 - np.percentile(projected, lo_pct))


def upright_from_pitched(pose: np.ndarray) -> np.ndarray:
    """
    Same XY/closing as a pitched cup grasp, but tool Z straight down.

    Used for the high hover so we only ask for the 25° wrist tip once the
    arm is already above the cup — much easier for IK near the workspace edge.
    """
    out = pose.copy()
    y = pose[:3, 1].copy()
    y[2] = 0.0
    y = y / (np.linalg.norm(y) + 1e-9)
    z = np.array([0.0, 0.0, -1.0])
    x = np.cross(y, z)
    x = x / (np.linalg.norm(x) + 1e-9)
    y = np.cross(z, x)
    out[:3, 0] = x
    out[:3, 1] = y
    out[:3, 2] = z
    return out


class PickUpSkill(BaseSkill):
    """
    Pick up an object from the workspace.
    
    Pipeline:
    1. SAM3 segment the target object in scene images
    2. Get pointcloud of object from multi-camera fusion
    3. Analyze pointcloud for grasp affordances
    4. Plan approach trajectory (approach from above)
    5. Open gripper
    6. Move to pre-grasp position
    7. Move to grasp position
    8. Close gripper
    9. Lift object
    """
    
    name = "pick_up"
    required_params = ["object_name"]
    
    @property
    def optional_params(self) -> Dict[str, Any]:
        return {
            "grasp_type": "top",  # upright top-down; "side" still available if needed
            "approach_height": 0.10,  # Height above object for approach
            "lift_height": 0.15,  # Height to lift after grasping
            # Max gripper force (N). With force_limited=True the jaws stop as
            # soon as contact force hits this — keep it low for "any resistance".
            "grasp_force": 5.0,
            # Only used when grasp_type="side" (pour-style tipped grasps).
            "pitch_deg": 0.0,
            # Close mode: True = squeeze until contact force (good for rigid
            # objects / flat spoons). False = close to measured diameter minus
            # squeeze (needed for soft paper cups that give no force feedback).
            "force_limited": True,
            # How much narrower than the measured object diameter to close
            # (metres). Only used when force_limited=False.
            "squeeze": 0.004,
            # Below this fraction of the measured object width, treat the
            # grasp as having crushed the object (width-close mode only).
            "crush_fraction": 0.7,
            # World-Z shift of the grasp, metres. Negative lowers the fingers.
            # Keep near 0 for flat objects (spoons); +0.02 floats above them.
            "z_offset": 0.0,
            # Clear the cameras before scanning via frankapy reset_joints
            # (home), not a hardcoded XYZ park.
            "reset_before_scan": True,
            # Optional override: if set, refuse grasps whose measured closing
            # width exceeds this (metres). Default None = no extra cap beyond
            # the gripper's physical opening. The old 55mm default was only a
            # bandaid for one bloated SAM mask (~74mm reported on a ~40mm
            # beaker); width measurement now crops to a local neighbourhood.
            "max_object_width": None,
        }
    
    def check_preconditions(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Lightweight gate before execute. Does not scan or refuse on gripper
        width: execute opens the gripper and retracts the arm before looking,
        so a stuck mid-width grasp is cleared rather than blocking the skill.
        """
        return self.validate_params(params)
    
    def execute(self, params: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Execute the pick up skill.
        
        Returns:
            (success, result_dict) where result_dict contains:
            - grasped_object: Name of the object that was grasped
            - grasp_pose: The pose where the grasp was executed
            - grasp_type: The type of grasp used
        """
        params = self.get_params_with_defaults(params)
        object_name = params["object_name"]
        grasp_type = params["grasp_type"]
        approach_height = params["approach_height"]
        lift_height = params["lift_height"]
        grasp_force = params["grasp_force"]
        
        print(f"[PickUp] Starting pick up of '{object_name}'")

        # 1. Drop anything held RIGHT HERE before any arm motion. A leftover
        # goto_pose from a failed previous run used to keep running until that
        # old target was reached, so open_gripper appeared to wait until then
        # and the object never fell at the start.
        print(f"[PickUp] Opening gripper at current pose (drop first)...")
        if not self.open_gripper():
            return False, {"error": "Failed to open gripper"}
        self.wait(0.4)

        # 2. Clear the cage cameras BEFORE scanning. Default: frankapy
        # reset_joints (home). Scanning with the arm over the table occludes
        # the object and pollutes the fused cloud.
        if params.get("reset_before_scan", True):
            print("[PickUp] reset_joints (home) clear of cameras before scan...")
            if not self.go_home():
                self.open_gripper()
                return False, {"error": "Failed to reset_joints before scanning"}
        else:
            retract = np.array(params.get("retract_xyz", [0.35, 0.0, 0.45]), dtype=float)
            print(f"[PickUp] Retracting clear of cameras to {np.round(retract, 3)}...")
            if not self.move_to_position(retract, maintain_orientation=True):
                self.open_gripper()
                return False, {"error": "Failed to retract before scanning"}
        self.wait(0.4)

        # 3. Fresh multi-camera scan with the arm out of the way.
        print(f"[PickUp] Localizing object (all cameras, arm at home)...")
        self.vision.clear_cache()
        located = self.vision.locate(object_name, force_refresh=True)
        
        if located is None:
            return False, {"error": f"Could not segment '{object_name}' in scene"}
        
        object_pc = located["points"]
        centroid = located["centroid"]
        dimensions = located["dimensions"]
        
        confidences = located.get("confidences") or {}
        if confidences:
            print(f"[PickUp] Object segmented with confidence: {max(confidences.values()):.2f}")
        cams = located.get("cameras") or []
        print(f"[PickUp] Trusting fused median of cameras {cams} "
              f"(rejected {located.get('rejected_cameras') or []})")
        print(f"[PickUp] Object centroid: {np.round(centroid, 4)}, "
              f"dimensions: {np.round(dimensions, 4)}")
        
        # 4. Upright top-down grasp by default (no pour tip / spout align).
        print(f"[PickUp] Computing grasp pose (type: {grasp_type})...")
        if grasp_type == "side":
            candidates = self.vision.compute_grasp_candidates(
                object_pc,
                object_name,
                grasp_type="side",
                pitch_deg=params["pitch_deg"],
            )
        else:
            # "top", "auto", "handle" → single upright (or handle) pose
            gtype = "top" if grasp_type in ("auto", "top") else grasp_type
            pose = self.vision.compute_grasp_pose(
                object_pc,
                object_name,
                grasp_type=gtype,
                pitch_deg=0.0,
            )
            candidates = [pose] if pose is not None else []

        if not candidates:
            return False, {"error": "Could not compute valid grasp pose"}

        z_offset = params["z_offset"]
        grasp_pose = None
        for i, cand in enumerate(candidates):
            pose = cand.copy()
            if z_offset:
                pose[2, 3] = float(np.clip(
                    pose[2, 3] + z_offset,
                    self.workspace_min[2],
                    self.workspace_max[2],
                ))

            tip = pose[:3, 2]
            tip_from_vert = float(np.degrees(np.arccos(np.clip(-tip[2], -1, 1))))

            hover_pose = pose.copy()
            hover_pose[2, 3] = self.safe_height
            # Keep wrist upright on the way down when this is a top grasp.
            if tip_from_vert < 5.0:
                hover_pose[:3, :3] = pose[:3, :3]
            else:
                hover_pose = upright_from_pitched(pose)
                hover_pose[2, 3] = self.safe_height

            approach_pose = pose.copy()
            approach_pose[2, 3] += approach_height

            print(f"[PickUp] Candidate {i + 1}/{len(candidates)}: "
                  f"wrist tip {tip_from_vert:.0f} deg from vertical, "
                  f"grasp {np.round(pose[:3, 3], 4)}")

            print(f"[PickUp]   hover {np.round(hover_pose[:3, 3], 4)}...")
            if not self.move_to_pose(hover_pose, reach_tol=0.05):
                print(f"[PickUp]   unreachable at hover — trying next")
                continue
            print(f"[PickUp]   approach {np.round(approach_pose[:3, 3], 4)}...")
            if not self.move_to_pose(approach_pose, reach_tol=0.04):
                print(f"[PickUp]   unreachable at approach — trying next")
                continue
            print(f"[PickUp]   grasp {np.round(pose[:3, 3], 4)}...")
            if not self.move_to_pose(pose, speed="slow", reach_tol=0.03):
                print(f"[PickUp]   unreachable at grasp — trying next")
                continue

            grasp_pose = pose
            print(f"[PickUp] Using candidate {i + 1} "
                  f"(wrist tip {tip_from_vert:.0f} deg from vertical)")
            break

        if grasp_pose is None:
            return False, {
                "error": f"No reachable grasp among {len(candidates)} candidates",
            }

        force_limited = bool(params["force_limited"])
        # Best-effort width for logging / width-close mode. Flat objects often
        # have no cloud at grasp height — that used to abort the whole pick.
        expected = width_along_closing_axis(object_pc, grasp_pose)
        target_width = None

        if force_limited:
            print(f"[PickUp] Closing until contact "
                  f"(force_limited, max {grasp_force:.1f} N)"
                  + (f"; cloud width ~{expected * 1000:.1f}mm"
                     if expected is not None else
                     "; no cloud width at grasp height"))
            if not self.close_gripper(
                force=grasp_force, force_limited=True
            ):
                return False, {"error": "Failed to close gripper"}
        else:
            # Soft cups: no useful contact force, so aim for diameter − squeeze.
            if expected is None:
                return False, {
                    "error": "Could not measure object width at grasp height; "
                             "refusing to close blind on a crushable object "
                             "(set force_limited=True to close on contact instead)",
                }

            max_w = params.get("max_object_width")
            if max_w is not None and expected > float(max_w):
                self.open_gripper()
                return False, {
                    "error": (
                        f"Measured width {expected * 1000:.1f}mm > "
                        f"max_object_width {float(max_w) * 1000:.0f}mm — "
                        f"segmentation is probably bloated; refusing false grasp"
                    ),
                    "expected_width": expected,
                }

            gripper_max = 0.08
            if expected > gripper_max:
                self.open_gripper()
                return False, {
                    "error": (
                        f"Measured width {expected * 1000:.1f}mm > gripper opening "
                        f"{gripper_max * 1000:.0f}mm"
                    ),
                    "expected_width": expected,
                }

            target_width = max(0.0, expected - params["squeeze"])
            print(f"[PickUp] Object ~{expected * 1000:.1f}mm across; "
                  f"closing to {target_width * 1000:.1f}mm "
                  f"(diameter - {params['squeeze'] * 1000:.0f}mm)...")

            if not self.close_gripper(
                force=grasp_force, target_width=target_width, force_limited=False
            ):
                return False, {"error": "Failed to close gripper"}

        self.wait(0.3)

        gripper_width = self.get_gripper_width()
        grasped = self.gripper_is_grasped()
        print(f"[PickUp] Holding at {gripper_width * 1000:.1f}mm "
              f"(gripper_is_grasped={grasped}"
              + (f", target {target_width * 1000:.1f}mm" if target_width is not None
                 else "")
              + ")")

        if gripper_width < 0.008:
            self.open_gripper()
            return False, {
                "error": "Gripper closed on nothing",
                "gripper_width": gripper_width,
                "expected_width": expected,
                "target_width": target_width,
            }
        if (not force_limited and expected is not None
                and gripper_width < expected * params["crush_fraction"]):
            print("[PickUp] Crush detected — opening gripper to release")
            self.open_gripper()
            return False, {
                "error": (f"Gripper closed to {gripper_width * 1000:.1f}mm on a "
                          f"{expected * 1000:.1f}mm object: crushed it"),
                "gripper_width": gripper_width,
                "expected_width": expected,
                "target_width": target_width,
            }
        # Jaws barely left fully open and no grasp contact → empty/missed.
        if gripper_width > 0.07 and not grasped:
            self.open_gripper()
            return False, {
                "error": (
                    f"Gripper still {gripper_width * 1000:.1f}mm open "
                    f"(not grasped) — did not close on the object"
                ),
                "gripper_width": gripper_width,
                "expected_width": expected,
                "target_width": target_width,
            }
        
        # Step 11: Lift object
        print(f"[PickUp] Lifting object...")
        lift_pose = grasp_pose.copy()
        lift_pose[2, 3] += lift_height
        if not self.move_to_pose(lift_pose):
            print("[PickUp] Lift failed — opening gripper to release")
            self.open_gripper()
            return False, {"error": "Failed to lift object"}

        # Re-check after the lift: an object can slip out on the way up.
        lifted_width = self.get_gripper_width()
        if lifted_width < 0.008:
            return False, {
                "error": (f"Object lost during the lift: gripper went from "
                          f"{gripper_width * 1000:.1f}mm to "
                          f"{lifted_width * 1000:.1f}mm"),
                "gripper_width": lifted_width,
                "expected_width": expected,
            }
        if (not force_limited and expected is not None
                and lifted_width < expected * params["crush_fraction"]):
            print("[PickUp] Crush after lift — opening gripper to release")
            self.open_gripper()
            return False, {
                "error": (f"Object crushed during the lift: "
                          f"{lifted_width * 1000:.1f}mm vs expected "
                          f"{expected * 1000:.1f}mm"),
                "gripper_width": lifted_width,
                "expected_width": expected,
            }

        print(f"[PickUp] Successfully picked up '{object_name}'")
        
        # The object has moved, so any cached segmentation is now stale.
        self.vision.clear_cache()
        
        result = {
            "grasped_object": object_name,
            "grasp_pose": grasp_pose.tolist(),
            "grasp_type": grasp_type,
            "gripper_width": lifted_width,
            "expected_width": expected,
            "target_width": target_width,
            "grasp_force": grasp_force,
            "centroid": centroid.tolist(),
            "dimensions": dimensions.tolist() if dimensions is not None else None,
        }
        return True, result
