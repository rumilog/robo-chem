"""
Base Skill Class

Abstract base class that all robot skills must inherit from.
Provides a consistent interface for the VLM orchestrator to invoke skills.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import time


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
    
    def capture_scene_image(self) -> np.ndarray:
        """Capture current scene image for verification."""
        return self.vision.capture_scene()[0]  # Return first camera
    
    def get_current_pose(self) -> np.ndarray:
        """Get current robot end-effector pose."""
        return self.robot.get_pose()
    
    def get_gripper_width(self) -> float:
        """Get current gripper opening width."""
        return self.robot.get_gripper_width()
    
    def is_gripper_holding(self) -> bool:
        """Check if gripper is holding something (width > threshold)."""
        width = self.get_gripper_width()
        return 0.001 < width < 0.075  # Between fully closed and fully open
    
    # ==================== Motion Primitives ====================
    
    def move_to_pose(self, pose, speed: str = "normal") -> bool:
        """
        Move end-effector to a target pose.
        
        Args:
            pose: Target pose (4x4 matrix or Pose object)
            speed: Movement speed ("slow", "normal", "fast")
            
        Returns:
            True if motion completed successfully
        """
        try:
            self.robot.goto_pose(pose)
            return True
        except Exception as e:
            print(f"Motion failed: {e}")
            return False
    
    def move_to_position(self, position: np.ndarray, maintain_orientation: bool = True) -> bool:
        """
        Move to a 3D position, optionally maintaining current orientation.
        
        Args:
            position: Target [x, y, z] position
            maintain_orientation: If True, keep current end-effector orientation
            
        Returns:
            True if motion completed successfully
        """
        try:
            pose = self.robot.get_pose()
            pose.translation = position
            self.robot.goto_pose(pose)
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
            pose = self.robot.get_pose()
            pose.translation = pose.translation + np.array(delta)
            self.robot.goto_pose(pose)
            return True
        except Exception as e:
            print(f"Motion failed: {e}")
            return False
    
    def open_gripper(self, width: float = 0.08) -> bool:
        """Open gripper to specified width (default: fully open)."""
        try:
            self.robot.goto_gripper(width=width)
            return True
        except Exception as e:
            print(f"Gripper open failed: {e}")
            return False
    
    def close_gripper(self, force: float = 15.0) -> bool:
        """Close gripper with specified force."""
        try:
            self.robot.goto_gripper(width=0.0, grasp=True, force=force)
            return True
        except Exception as e:
            print(f"Gripper close failed: {e}")
            return False
    
    def rotate_wrist(self, angle_degrees: float, axis: str = "z") -> bool:
        """
        Rotate end-effector around specified axis.
        
        Args:
            angle_degrees: Rotation angle in degrees
            axis: Rotation axis ("x", "y", or "z")
            
        Returns:
            True if rotation completed successfully
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
            pose = self.robot.get_pose()
            pose.rotation = np.dot(rotation, pose.rotation)
            self.robot.goto_pose(pose)
            return True
        except Exception as e:
            print(f"Rotation failed: {e}")
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
