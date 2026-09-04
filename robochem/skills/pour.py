"""
Pour Skill

Pours contents of held container into target container.
Critical for chemistry operations involving liquids and powders.
"""

from typing import Dict, Any, Tuple, List
import numpy as np
import time

from .base_skill import BaseSkill


class PourSkill(BaseSkill):
    """
    Pour contents of held container into target container.
    
    Pipeline:
    1. Verify gripper is holding a container
    2. Localize target container
    3. Move above target (with offset for pour trajectory)
    4. Execute controlled pour trajectory (tilt)
    5. Hold pour position
    6. Return to upright position
    """
    
    name = "pour"
    required_params = ["target_container"]
    
    @property
    def optional_params(self) -> Dict[str, Any]:
        return {
            "pour_angle": 90,  # Maximum tilt angle in degrees
            "pour_speed": "medium",  # "slow", "medium", "fast"
            "hold_duration": 1.5,  # Seconds to hold at max pour angle
            "approach_height": 0.12,  # Height above target
            "horizontal_offset": 0.05,  # Offset to account for pour trajectory
        }
    
    def check_preconditions(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check if pour can be executed:
        - Gripper is holding a container
        - Target container is visible
        """
        valid, msg = self.validate_params(params)
        if not valid:
            return False, msg
        
        if not self.is_gripper_holding():
            return False, "Gripper is not holding anything. Pick up a container first."
        
        target = params["target_container"]
        centroid = self.get_object_centroid(target)
        if centroid is None:
            return False, f"Cannot locate target container '{target}'"
        
        return True, "Preconditions met"
    
    def execute(self, params: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Execute the pour skill.
        """
        params = self.get_params_with_defaults(params)
        target = params["target_container"]
        pour_angle = params["pour_angle"]
        pour_speed = params["pour_speed"]
        hold_duration = params["hold_duration"]
        approach_height = params["approach_height"]
        h_offset = params["horizontal_offset"]
        
        print(f"[Pour] Starting pour into '{target}'")
        
        # Step 1: Localize target container
        target_pos = self.get_object_centroid(target)
        if target_pos is None:
            return False, {"error": f"Cannot locate target '{target}'"}
        
        # Get target dimensions for better positioning
        target_dims = self.get_object_dimensions(target)
        target_height = target_dims[2] if target_dims is not None else 0.05
        
        print(f"[Pour] Target position: {target_pos}, height: {target_height}")
        
        # Step 2: Calculate pour position
        # Position above and slightly offset from target
        pour_position = target_pos.copy()
        pour_position[0] += h_offset  # Offset in X for pour trajectory
        pour_position[2] += target_height + approach_height
        
        # Step 3: Move to pour position (maintaining upright orientation)
        print(f"[Pour] Moving to pour position...")
        current_pose = self.get_current_pose()
        current_pose.translation = pour_position
        if not self.move_to_pose(current_pose):
            return False, {"error": "Failed to move to pour position"}
        
        # Step 4: Execute pour trajectory
        print(f"[Pour] Executing pour (angle: {pour_angle}°, speed: {pour_speed})...")
        pour_steps = self._get_pour_trajectory(pour_angle, pour_speed)
        
        for step_angle in pour_steps:
            if not self.rotate_wrist(step_angle, axis="y"):
                return False, {"error": f"Failed during pour at angle {step_angle}"}
            self.wait(0.05)  # Small delay between steps
        
        # Step 5: Hold at max pour angle
        print(f"[Pour] Holding pour position for {hold_duration}s...")
        self.wait(hold_duration)
        
        # Step 6: Return to upright
        print(f"[Pour] Returning to upright position...")
        for step_angle in reversed(pour_steps):
            if not self.rotate_wrist(-step_angle, axis="y"):
                return False, {"error": "Failed returning to upright"}
            self.wait(0.05)
        
        print(f"[Pour] Successfully poured into '{target}'")
        
        return True, {
            "poured_into": target,
            "pour_angle": pour_angle,
            "pour_speed": pour_speed
        }
    
    def _get_pour_trajectory(self, max_angle: float, speed: str) -> List[float]:
        """
        Generate pour trajectory as list of incremental angles.
        
        Args:
            max_angle: Maximum tilt angle
            speed: "slow", "medium", or "fast"
            
        Returns:
            List of incremental angle steps
        """
        if speed == "slow":
            num_steps = 30
        elif speed == "medium":
            num_steps = 20
        else:  # fast
            num_steps = 10
        
        # Generate smooth trajectory using sine wave for acceleration/deceleration
        t = np.linspace(0, np.pi, num_steps)
        angles = max_angle * (1 - np.cos(t)) / 2
        
        # Convert to incremental angles
        increments = [angles[0]]
        for i in range(1, len(angles)):
            increments.append(angles[i] - angles[i-1])
        
        return increments
