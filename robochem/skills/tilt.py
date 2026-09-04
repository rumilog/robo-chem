"""
Tilt Skill

Tilts the end-effector/held object to a specified angle.
Useful for draining liquids, angling containers, etc.
"""

from typing import Dict, Any, Tuple
import numpy as np

from .base_skill import BaseSkill


class TiltSkill(BaseSkill):
    """
    Tilt the end-effector to a specified angle.
    
    Tilts around the specified axis while maintaining position.
    Useful for:
    - Draining liquid from a container
    - Angling a pour
    - Returning to upright position
    """
    
    name = "tilt"
    required_params = ["angle"]
    
    @property
    def optional_params(self) -> Dict[str, Any]:
        return {
            "axis": "y",  # Rotation axis: "x", "y", or "z"
            "speed": "normal",  # "slow", "normal", "fast"
            "hold_duration": 0.0,  # Seconds to hold at angle
        }
    
    def check_preconditions(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate parameters."""
        valid, msg = self.validate_params(params)
        if not valid:
            return False, msg
        
        angle = params.get("angle", 0)
        if not isinstance(angle, (int, float)):
            return False, "Angle must be a number"
        
        if abs(angle) > 180:
            return False, "Angle must be between -180 and 180 degrees"
        
        axis = params.get("axis", "y").lower()
        if axis not in ["x", "y", "z"]:
            return False, "Axis must be 'x', 'y', or 'z'"
        
        return True, "Ready to tilt"
    
    def execute(self, params: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Tilt to the specified angle.
        
        Returns:
            (success, result) with final orientation info
        """
        params = self.get_params_with_defaults(params)
        angle = params["angle"]
        axis = params["axis"].lower()
        speed = params["speed"]
        hold_duration = params["hold_duration"]
        
        print(f"[Tilt] Tilting {angle} degrees around {axis.upper()}-axis")
        
        try:
            # Get current pose
            current_pose = self.get_current_pose()
            original_rotation = current_pose.rotation.copy()
            
            # Compute rotation matrix
            angle_rad = np.radians(angle)
            c, s = np.cos(angle_rad), np.sin(angle_rad)
            
            if axis == "x":
                rotation = np.array([
                    [1, 0, 0],
                    [0, c, -s],
                    [0, s, c]
                ])
            elif axis == "y":
                rotation = np.array([
                    [c, 0, s],
                    [0, 1, 0],
                    [-s, 0, c]
                ])
            else:  # z
                rotation = np.array([
                    [c, -s, 0],
                    [s, c, 0],
                    [0, 0, 1]
                ])
            
            # Apply rotation
            new_rotation = np.dot(rotation, original_rotation)
            current_pose.rotation = new_rotation
            
            # Execute tilt
            if not self.move_to_pose(current_pose):
                return False, {"error": "Failed to tilt"}
            
            # Hold if requested
            if hold_duration > 0:
                print(f"[Tilt] Holding for {hold_duration}s...")
                self.wait(hold_duration)
            
            print(f"[Tilt] Tilt complete")
            
            return True, {
                "angle": angle,
                "axis": axis,
                "hold_duration": hold_duration
            }
            
        except Exception as e:
            return False, {"error": str(e)}
    
    def return_to_upright(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Convenience method to return end-effector to upright position.
        """
        # This would need to track the accumulated tilt and reverse it
        # For now, return to a default orientation
        return self.execute({"angle": 0, "axis": "y"})
