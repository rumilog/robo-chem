"""
Base Skill Class

Abstract base class that all robot skills must inherit from.
Provides a consistent interface for the VLM orchestrator to invoke skills.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import time

try:
    from autolab_core import RigidTransform
except ImportError:
    RigidTransform = None


# frankapy rejects any tool pose that isn't expressed between these two frames.
FRANKA_TOOL_FRAME = "franka_tool"
WORLD_FRAME = "world"

# Named speeds mapped to goto_pose durations in seconds.
SPEED_DURATIONS = {"slow": 6.0, "normal": 3.0, "fast": 1.5}


def _orthonormalize(rotation: np.ndarray) -> np.ndarray:
    """
    Project a rotation matrix onto SO(3).

    Grasp rotations are assembled from PCA eigenvectors and VLM output, which
    drift far enough from orthonormal that RigidTransform rejects them.
    """
    u, _, vt = np.linalg.svd(rotation)
    oriented = u @ vt
    if np.linalg.det(oriented) < 0:
        u[:, -1] *= -1
        oriented = u @ vt
    return oriented


def to_rigid_transform(pose) -> "RigidTransform":
    """
    Coerce a pose into the RigidTransform that frankapy's goto_pose requires.

    Skills build poses as 4x4 numpy matrices, but frankapy validates
    from_frame/to_frame on an autolab_core RigidTransform and raises otherwise.

    Args:
        pose: 4x4 numpy matrix or RigidTransform

    Returns:
        RigidTransform from franka_tool to world
    """
    if RigidTransform is None:
        raise ImportError(
            "autolab_core is required to send poses to the robot. "
            "Install it into the frankapy environment."
        )

    if isinstance(pose, RigidTransform):
        rotation, translation = pose.rotation, pose.translation
    else:
        matrix = np.asarray(pose, dtype=float)
        if matrix.shape != (4, 4):
            raise ValueError(f"Expected a 4x4 pose matrix, got shape {matrix.shape}")
        rotation, translation = matrix[:3, :3], matrix[:3, 3]

    return RigidTransform(
        rotation=_orthonormalize(np.asarray(rotation, dtype=float)),
        translation=np.asarray(translation, dtype=float),
        from_frame=FRANKA_TOOL_FRAME,
        to_frame=WORLD_FRAME,
    )


def to_matrix(pose) -> np.ndarray:
    """
    Convert a pose into a 4x4 numpy matrix.

    Args:
        pose: RigidTransform or 4x4 numpy matrix

    Returns:
        4x4 numpy matrix
    """
    if RigidTransform is not None and isinstance(pose, RigidTransform):
        matrix = np.eye(4)
        matrix[:3, :3] = pose.rotation
        matrix[:3, 3] = pose.translation
        return matrix

    matrix = np.asarray(pose, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 pose matrix, got shape {matrix.shape}")
    return matrix


class BaseSkill(ABC):
    """
    Abstract base class for all robot manipulation skills.
    
    Each skill follows the pattern:
    1. Pre-conditions check (is this skill valid to execute?)
    2. Visual grounding (where is the target object?)
    3. Grasp/motion planning (how do we interact with it?)
    4. Execution (move the robot)
    5. Post-conditions check (did it work?)
    
    Subclasses must implement:
    - name property
    - required_params property
    - check_preconditions method
    - execute method
    """
    
    def __init__(self, robot_interface, vision_system, config: Dict[str, Any] = None):
        """
        Initialize the skill.
        
        Args:
            robot_interface: Robot arm interface (e.g., FrankaArm)
            vision_system: Vision system for object localization
            config: Optional skill-specific configuration
        """
        self.robot = robot_interface
        self.vision = vision_system
        self.config = config or {}
        
        # Default safety parameters
        self.max_velocity = config.get("max_velocity", 0.3)  # m/s
        self.approach_height = config.get("approach_height", 0.10)  # 10cm above target
        self.safe_height = config.get("safe_height", 0.30)  # 30cm safe height
        
        # Workspace bounds in the robot base frame, as [x, y, z] in meters
        self.workspace_min = np.array(config.get("workspace_min", [0.25, -0.40, 0.015]))
        self.workspace_max = np.array(config.get("workspace_max", [0.75, 0.40, 0.70]))

        # Gripper widths that read as empty. Below the lower bound the jaws are
        # shut on nothing; above the upper bound they are open. Anything in
        # between is most likely clamped around an object.
        self.gripper_empty_min = config.get("gripper_empty_min", 0.005)
        self.gripper_empty_max = config.get("gripper_empty_max", 0.070)
        
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Skill name for orchestrator reference.
        Must be unique across all skills.
        """
        pass
    
    @property
    @abstractmethod
    def required_params(self) -> List[str]:
        """
        List of required parameter names for this skill.
        The orchestrator will ensure these are provided before execution.
        """
        pass
    
    @property
    def optional_params(self) -> Dict[str, Any]:
        """
        Dictionary of optional parameters with their default values.
        Override in subclasses to define optional parameters.
        """
        return {}
    
    @abstractmethod
    def check_preconditions(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check if the skill can be executed given current state.
        
        Args:
            params: Parameters for the skill execution
            
        Returns:
            Tuple of (can_execute: bool, message: str)
            If can_execute is False, message explains why.
        """
        pass
    
    @abstractmethod
    def execute(self, params: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Execute the skill.
        
        Args:
            params: Parameters for the skill (validated against required_params)
            
        Returns:
            Tuple of (success: bool, result_data: dict)
            result_data contains execution details and any outputs.
        """
        pass
    
    def validate_params(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate that all required parameters are provided.
        
        Args:
            params: Parameters to validate
            
        Returns:
            Tuple of (valid: bool, message: str)
        """
        missing = [p for p in self.required_params if p not in params]
        if missing:
            return False, f"Missing required parameters: {missing}"
        return True, "Parameters valid"
    
    def get_params_with_defaults(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge provided params with default values for optional params.
        
        Args:
            params: Provided parameters
            
        Returns:
            Complete parameter dictionary with defaults filled in
        """
        complete_params = self.optional_params.copy()
        complete_params.update(params)
        return complete_params
    
    # ==================== Helper Methods ====================
    
    def get_object_pose(self, object_name: str) -> Optional[np.ndarray]:
        """
        Use vision system to localize an object in the scene.
        
        Args:
            object_name: Semantic name of the object to find
            
        Returns:
            4x4 pose matrix in robot base frame, or None if not found
        """
        return self.vision.localize_object(object_name)
    
    def get_object_centroid(self, object_name: str) -> Optional[np.ndarray]:
        """
        Get the 3D centroid of an object.
        
        Args:
            object_name: Semantic name of the object
            
        Returns:
            3D point (x, y, z) in robot base frame, or None if not found
        """
        return self.vision.get_object_centroid(object_name)
    
    def get_grasp_pose(self, object_name: str, grasp_type: str = "auto") -> Optional[np.ndarray]:
        """
        Compute optimal grasp pose for an object.
        
        Args:
            object_name: Object to grasp
            grasp_type: Type of grasp ("top", "side", "handle", "auto")
            
        Returns:
            4x4 grasp pose matrix, or None if not computed
        """
        return self.vision.compute_grasp_pose(object_name, grasp_type)
    
    def get_object_dimensions(self, object_name: str) -> Optional[np.ndarray]:
        """
        Get bounding box dimensions of an object.
        
        Args:
            object_name: Object to measure
            
        Returns:
            Array of [length, width, height] in meters, or None
        """
        return self.vision.get_object_dimensions(object_name)
    
    def locate_container(self, object_name: str,
                         force_refresh: bool = True,
                         rim_band: float = 0.015) -> Optional[Dict[str, Any]]:
        """
        Localize a container and measure the geometry skills actually need.

        ``get_object_dimensions`` returns PCA extents *sorted descending*, so
        ``dimensions[2]`` is the object's smallest extent, not its height. For
        an upright beaker (100mm tall, 40mm across) that is the diameter — so
        every skill that treated ``dims[2]`` as height was aiming ~60mm low.
        This helper measures in the world frame instead.

        Args:
            object_name: Semantic name to segment
            force_refresh: Re-segment rather than trusting the cache. Default
                True because containers move between skills.
            rim_band: Thickness of the top slab used to estimate the opening.

        Returns:
            Dict with points, centroid, top_z, base_z, height, rim_center
            (XY of the opening) and rim_radius, or None if not located.
        """
        located = self.vision.locate(object_name, force_refresh=force_refresh)
        if located is None:
            return None

        points = located.get("points")
        if points is None or len(points) < 20:
            print(f"[Vision] '{object_name}' located but has too few points "
                  f"({0 if points is None else len(points)}) to measure")
            return None

        points = np.asarray(points, dtype=float)
        # Percentiles, not min/max: a handful of depth outliers otherwise set
        # the rim height and the arm dives into or hovers well above the cup.
        top_z = float(np.percentile(points[:, 2], 97))
        base_z = float(np.percentile(points[:, 2], 3))

        rim = points[points[:, 2] > top_z - rim_band]
        if len(rim) < 10:
            rim = points
        rim_center = np.array([float(np.median(rim[:, 0])),
                               float(np.median(rim[:, 1]))])
        radial = np.linalg.norm(rim[:, :2] - rim_center, axis=1)
        rim_radius = float(np.percentile(radial, 90))

        return {
            "name": object_name,
            "points": points,
            "centroid": np.asarray(located["centroid"], dtype=float),
            "dimensions": located.get("dimensions"),
            "top_z": top_z,
            "base_z": base_z,
            "height": top_z - base_z,
            # Median of the rim band, not the full-cloud centroid: cage
            # coverage is one-sided, so the fused cloud leans toward whichever
            # cameras saw the object and the centroid sits off the opening.
            "rim_center": rim_center,
            "rim_radius": rim_radius,
            "cameras": located.get("cameras"),
        }

    def clear_cameras(self, params: Dict[str, Any], tag: str = "Skill") -> bool:
        """
        Get the arm out of the cage's view before a scan.

        The pick and pour skills both converged on joint-space ``reset_joints``
        rather than a hardcoded XYZ park: from an arbitrary pose a cartesian
        retract is often unreachable, and scanning with the arm over the table
        occludes the object and pollutes the fused cloud. Safe while holding an
        object — pour does exactly this with the beaker in the gripper.
        """
        if params.get("reset_before_scan", True):
            print(f"[{tag}] reset_joints (home) clear of cameras before scan...")
            if not self.go_home():
                return False
        else:
            retract = np.array(params.get("retract_xyz", [0.35, 0.0, 0.45]), dtype=float)
            print(f"[{tag}] Retracting clear of cameras to {np.round(retract, 3)}...")
            if not self.move_to_position(retract, reach_tol=0.06):
                return False
        self.wait(0.4)
        return True

    def held_width(self) -> float:
        """Gripper opening right now, in metres. Reads as the held object's width."""
        return float(self.get_gripper_width())

    def check_still_holding(self, reference_width: float, tag: str = "Skill",
                            drop_floor: float = 0.003,
                            slip_fraction: float = 0.6) -> Tuple[bool, str]:
        """
        Confirm the tool is still in the gripper.

        Every skill after ``pick_up`` runs with something already held, and the
        commonest silent failure is that it was dropped or squeezed out three
        motions ago while the skill happily reported success. Compare against
        the width recorded when the skill started.

        Thin tools (spoon handles ~7mm) sit near the old 8mm "empty" floor, so
        emptiness is only absolute shut jaws (~3mm). A width that has not
        collapsed relative to the start is still a hold.

        Returns:
            (still_holding, message)
        """
        width = self.held_width()
        # Jaws essentially shut — nothing left, even for a thin spoon.
        if width < drop_floor:
            return False, (f"gripper closed to {width * 1000:.1f}mm — the held "
                           f"object is gone (was {reference_width * 1000:.1f}mm)")
        if reference_width > 0 and width < reference_width * slip_fraction:
            return False, (f"gripper collapsed from {reference_width * 1000:.1f}mm "
                           f"to {width * 1000:.1f}mm — object slipped or was crushed")
        if width > 0.075:
            return False, (f"gripper is {width * 1000:.1f}mm open — nothing held "
                           f"(was {reference_width * 1000:.1f}mm)")
        return True, f"still holding at {width * 1000:.1f}mm"

    def tool_down_rotation(self) -> np.ndarray:
        """
        Current wrist rotation flattened so the tool points straight down.

        Keeps the finger closing axis where it is (so a held pipette or stirrer
        is not spun in the jaws) and only re-aims tool Z at the table. Used by
        the insertion skills, which need a vertical tool but must not assume the
        wrist arrived upright from whatever the previous skill left behind.
        """
        R = _orthonormalize(np.asarray(self.get_current_pose().rotation, dtype=float))
        y = R[:3, 1].copy()
        y[2] = 0.0
        norm = np.linalg.norm(y)
        if norm < 1e-6:
            # Wrist is rolled fully on its side; fall back to the base Y axis.
            y = np.array([0.0, 1.0, 0.0])
        else:
            y = y / norm
        z = np.array([0.0, 0.0, -1.0])
        x = np.cross(y, z)
        x = x / (np.linalg.norm(x) + 1e-9)
        y = np.cross(z, x)
        return np.column_stack([x, y, z])

    def tool_tip_deg(self) -> float:
        """Angle of the tool axis from straight down, in degrees."""
        R = np.asarray(self.get_current_pose().rotation, dtype=float)
        return float(np.degrees(np.arccos(np.clip(-R[2, 2], -1.0, 1.0))))

    def goto_pose_rigid(self, xyz: np.ndarray, rotation: np.ndarray,
                        duration: float = 3.0,
                        use_impedance: bool = False) -> bool:
        """
        Command an explicit position + orientation.

        ``use_impedance=False`` by default: the pour skill only started tracking
        commanded orientation once it stopped using the impedance controller,
        which quietly settles tens of degrees short. Pass True for contact-rich
        motions (digging into powder) where compliance is the point.
        """
        try:
            pose = self.get_current_pose()
            pose.translation = self._clamp_position(np.asarray(xyz, dtype=float))
            pose.rotation = _orthonormalize(np.asarray(rotation, dtype=float))
            self.robot.goto_pose(
                to_rigid_transform(pose),
                duration=float(duration),
                use_impedance=use_impedance,
            )
            self.wait(0.1)
            return True
        except Exception as e:
            print(f"Oriented motion failed: {e}")
            return False

    def reached(self, target_xyz: np.ndarray, tol: float) -> Tuple[bool, float]:
        """Did the arm actually get within ``tol`` metres of ``target_xyz``?"""
        actual = np.asarray(self.get_current_pose().translation, dtype=float)
        err = float(np.linalg.norm(actual - np.asarray(target_xyz, dtype=float)))
        return err <= tol, err

    def capture_scene_image(self) -> np.ndarray:
        """Capture current scene image for verification."""
        return self.vision.capture_scene()[0]  # Return first camera
    
    def get_current_pose(self) -> "RigidTransform":
        """Get current robot end-effector pose as a franka_tool -> world transform."""
        return to_rigid_transform(self.robot.get_pose())
    
    def get_gripper_width(self) -> float:
        """Get current gripper opening width."""
        return self.robot.get_gripper_width()
    
    def is_gripper_holding(self) -> bool:
        """Check if gripper is holding something (width > threshold)."""
        width = self.get_gripper_width()
        return 0.001 < width < 0.075  # Between fully closed and fully open
    
    # ==================== Motion Primitives ====================
    
    def _clamp_position(self, position: np.ndarray) -> np.ndarray:
        """
        Clamp a target position into the configured workspace box.

        Grasp targets come from vision, so a bad segmentation can otherwise
        drive the arm into the table or the cage.
        """
        clamped = np.clip(
            np.asarray(position, dtype=float),
            self.workspace_min,
            self.workspace_max,
        )
        if not np.allclose(clamped, position, atol=1e-6):
            print(f"[Safety] Clamped target {np.round(position, 4)} -> {np.round(clamped, 4)}")
        return clamped
    
    def move_to_pose(self, pose, speed: str = "normal",
                     reach_tol: float = 0.035) -> bool:
        """
        Move end-effector to a target pose.
        
        Args:
            pose: Target pose (4x4 matrix or RigidTransform)
            speed: Movement speed ("slow", "normal", "fast")
            reach_tol: Max acceptable translation error after the move (m).
                frankapy impedance often settles a couple centimetres off,
                especially near the workspace edge; 20mm was rejecting poses
                that were visually on target.
            
        Returns:
            True if motion completed successfully
        """
        try:
            target = to_rigid_transform(pose)
            target.translation = self._clamp_position(target.translation)
            self.robot.goto_pose(
                target, duration=SPEED_DURATIONS.get(speed, SPEED_DURATIONS["normal"])
            )
            actual = to_rigid_transform(self.robot.get_pose()).translation
            err = float(np.linalg.norm(actual - target.translation))
            if err > reach_tol:
                print(f"[Safety] Pose not reached: commanded "
                      f"{np.round(target.translation, 4)}, actual "
                      f"{np.round(actual, 4)} (err={err * 1000:.1f} mm, "
                      f"tol={reach_tol * 1000:.0f} mm).")
                return False
            if err > 0.01:
                print(f"[PickUp] Reached within {err * 1000:.1f} mm of target "
                      f"{np.round(target.translation, 4)}")
            return True
        except Exception as e:
            print(f"Motion failed: {e}")
            return False
    
    def move_to_position(self, position: np.ndarray, maintain_orientation: bool = True,
                         reach_tol: Optional[float] = None,
                         speed: str = "normal") -> bool:
        """
        Move to a 3D position, optionally maintaining current orientation.

        Args:
            position: Target [x, y, z] position
            maintain_orientation: If True, keep current end-effector orientation
            reach_tol: If given, re-read the pose afterwards and return False
                unless the arm actually got within this many metres of the
                target. Left as None the call is fire-and-forget, which is how
                the early skills used it — and why they reported success after
                motions that never happened. New code should always pass a
                tolerance.
            speed: Named speed, mapped to a goto_pose duration.

        Returns:
            True if motion completed successfully
        """
        try:
            pose = to_rigid_transform(self.robot.get_pose())
            target = self._clamp_position(np.asarray(position, dtype=float))
            pose.translation = target
            self.robot.goto_pose(
                pose, duration=SPEED_DURATIONS.get(speed, SPEED_DURATIONS["normal"])
            )
            if reach_tol is None:
                return True
            actual = to_rigid_transform(self.robot.get_pose()).translation
            err = float(np.linalg.norm(actual - target))
            if err > reach_tol:
                print(f"[Safety] Position not reached: commanded "
                      f"{np.round(target, 4)}, actual {np.round(actual, 4)} "
                      f"(err={err * 1000:.1f} mm, tol={reach_tol * 1000:.0f} mm).")
                return False
            return True
        except Exception as e:
            print(f"Motion failed: {e}")
            return False
    
    def move_relative(self, delta: np.ndarray) -> bool:
        """
        Move relative to current position.
        
        Args:
            delta: [dx, dy, dz] offset in meters
            
        Returns:
            True if motion completed successfully
        """
        try:
            pose = to_rigid_transform(self.robot.get_pose())
            pose.translation = self._clamp_position(
                pose.translation + np.asarray(delta, dtype=float)
            )
            self.robot.goto_pose(pose)
            return True
        except Exception as e:
            print(f"Motion failed: {e}")
            return False
    
    def stop_arm(self) -> None:
        """Cancel any active frankapy arm skill left over from a previous run."""
        try:
            if not self.robot.is_skill_done():
                print("[Safety] Stopping leftover arm skill from a previous command...")
                self.robot.stop_skill()
                self.wait(0.2)
        except Exception as e:
            print(f"[Safety] stop_arm: {e}")

    def open_gripper(self, width: float = 0.08) -> bool:
        """
        Open gripper to specified width at the *current* pose.

        Stops any active arm skill first so a leftover goto_pose from a failed
        previous run cannot delay the open until that old target is reached.
        """
        try:
            self.stop_arm()
            try:
                self.robot.stop_gripper()
            except Exception:
                pass
            self.robot.goto_gripper(width=width, grasp=False, block=True)
            final = self.get_gripper_width()
            print(f"[Gripper] Open commanded to {width * 1000:.0f}mm; "
                  f"now {final * 1000:.1f}mm")
            return final > width * 0.7
        except Exception as e:
            print(f"Gripper open failed: {e}")
            return False
    
    def close_gripper(self, force: float = 5.0, target_width: float = 0.0,
                      force_limited: bool = True) -> bool:
        """
        Close the gripper.

        Args:
            force: Max grasp force in newtons. Only used when force_limited=True.
            target_width: Opening to drive to, in metres. Used when
                force_limited=False.
            force_limited: If True, squeeze with grasp=True until contact force
                reaches ``force`` (useless on soft paper cups). If False, move
                the jaws to exactly ``target_width`` with grasp=False and stop —
                that is the correct mode for "diameter minus squeeze".
        """
        try:
            if force_limited:
                # grasp=True: hardware squeezes until force is met. width here
                # is only an expected-size check, not a stop position.
                self.robot.goto_gripper(
                    width=0.0, grasp=True, force=float(force), speed=0.03
                )
            else:
                # grasp=False: position command. Jaws go to target_width and
                # stop. Using grasp=True here was the crush bug — frankapy
                # kept squeezing with force, and epsilon (~80mm) made 23mm
                # look like a successful 57mm grasp.
                self.robot.goto_gripper(
                    width=float(target_width), grasp=False, speed=0.05
                )
            return True
        except Exception as e:
            print(f"Gripper close failed: {e}")
            return False

    def gripper_is_grasped(self) -> bool:
        """True if the Franka gripper reports a successful grasp."""
        try:
            return bool(self.robot.get_gripper_is_grasped())
        except Exception:
            return False
    
    def rotate_wrist(self, angle_degrees: float, axis: str = "z") -> bool:
        """
        Rotate end-effector around a *base-frame* axis (left-multiply).

        Prefer ``rotate_about_tool_axis`` for pour tips around the gripper's
        closing axis so the jaws stay table-parallel.
        """
        angle_rad = np.radians(angle_degrees)
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        
        if axis == "x":
            rotation = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        elif axis == "y":
            rotation = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        elif axis == "z":
            rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        else:
            raise ValueError(f"Invalid axis: {axis}. Must be 'x', 'y', or 'z'")
        
        try:
            pose = to_rigid_transform(self.robot.get_pose())
            pose.rotation = _orthonormalize(np.dot(rotation, pose.rotation))
            self.robot.goto_pose(pose)
            return True
        except Exception as e:
            print(f"Rotation failed: {e}")
            return False

    def rotate_about_tool_axis(self, angle_degrees: float, axis: str = "y") -> bool:
        """
        Rotate the end-effector about a *tool-frame* axis (right-multiply).

        For spout-aligned beaker grasps, tool Y is the finger closing axis
        (horizontal). Extra tip about Y increases pour out the spout without
        rolling the jaws off the table plane.
        """
        angle_rad = np.radians(angle_degrees)
        c, s = np.cos(angle_rad), np.sin(angle_rad)

        if axis == "x":
            delta = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        elif axis == "y":
            delta = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        elif axis == "z":
            delta = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        else:
            raise ValueError(f"Invalid axis: {axis}. Must be 'x', 'y', or 'z'")

        try:
            pose = to_rigid_transform(self.robot.get_pose())
            pose.rotation = _orthonormalize(np.dot(pose.rotation, delta))
            # Must pass a duration — bare goto_pose often barely moves the wrist.
            self.robot.goto_pose(pose, duration=max(1.5, abs(angle_degrees) / 20.0))
            return True
        except Exception as e:
            print(f"Tool-axis rotation failed: {e}")
            return False
    
    def go_home(self) -> bool:
        """Move robot to home position."""
        try:
            self.robot.reset_joints()
            return True
        except Exception as e:
            print(f"Go home failed: {e}")
            return False
    
    def wait(self, seconds: float):
        """Wait for specified duration."""
        time.sleep(seconds)
