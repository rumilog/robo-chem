"""
Move To Skill

Moves the robot arm to a specified location without grasping.
Used for positioning, approaching, and retreating.
"""

from typing import Dict, Any, Tuple, Union
import numpy as np

from .base_skill import BaseSkill


class MoveToSkill(BaseSkill):
    """
    Move robot end-effector to a specified location.
    
    Can target:
    - An object (move relative to object location)
    - Explicit coordinates [x, y, z]
    - Named locations ("home", "safe")
    
    Pipeline:
    1. Determine target position
    2. Plan safe trajectory
    3. Execute movement
    """
    
    name = "move_to"
    required_params = ["target"]
    
    @property
    def optional_params(self) -> Dict[str, Any]:
        return {
            "offset": [0.0, 0.0, 0.0],  # XYZ offset from target
            "positioning": "above",  # "above", "behind", "front", "left", "right", "at"
            "maintain_orientation": True,  # Keep current end-effector orientation
            "speed": "normal",  # "slow", "normal", "fast"
        }
    
    def check_preconditions(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check if move_to can be executed:
        - Target is valid (object exists or coordinates valid)
        """
        valid, msg = self.validate_params(params)
        if not valid:
            return False, msg
        
        target = params["target"]
        
        # Check named locations
        if isinstance(target, str):
            if target.lower() == "home":
                return True, "Preconditions met"
            # It's an object name - check visibility
            centroid = self.get_object_centroid(target)
            if centroid is None:
                return False, f"Cannot locate target '{target}'"
        elif isinstance(target, (list, np.ndarray)):
            if len(target) != 3:
                return False, "Target coordinates must have 3 values [x, y, z]"
        else:
            return False, f"Invalid target type: {type(target)}"
        
        return True, "Preconditions met"
    
    def execute(self, params: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Execute the move_to skill.
        """
        params = self.get_params_with_defaults(params)
        target = params["target"]
        offset = np.array(params["offset"])
        positioning = params["positioning"]
        maintain_orientation = params["maintain_orientation"]
        speed = params["speed"]
        
        print(f"[MoveTo] Starting move to '{target}' ({positioning})")
        
        # Step 1: Determine target position
        if isinstance(target, str):
            if target.lower() == "home":
                print(f"[MoveTo] Moving to home position...")
                success = self.go_home()
                return success, {"moved_to": "home"}
            
            # Object name
            base_pos = self.get_object_centroid(target)
            if base_pos is None:
                return False, {"error": f"Cannot locate '{target}'"}
            
            # Apply positioning offset
            positioning_offset = self._get_positioning_offset(positioning, target)
            target_pos = base_pos + positioning_offset + offset
        else:
            # Explicit coordinates
            target_pos = np.array(target) + offset
        
        print(f"[MoveTo] Target position: {target_pos}")
        
        # Step 2: Execute movement
        if maintain_orientation:
            success = self.move_to_position(target_pos)
        else:
            # Create a pose with default orientation
            pose = np.eye(4)
            pose[:3, 3] = target_pos
            # Default: gripper pointing down
            pose[:3, :3] = np.array([
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, -1]
            ])
            success = self.move_to_pose(pose)
        
        if not success:
            return False, {"error": "Failed to move to target position"}
        
        print(f"[MoveTo] Successfully moved to target")
        
        return True, {
            "target": target if isinstance(target, str) else "coordinates",
            "final_position": target_pos.tolist(),
            "positioning": positioning
        }
    
    def _get_positioning_offset(self, positioning: str, object_name: str) -> np.ndarray:
        """
        Calculate offset based on positioning relative to object.
        """
        # Get object dimensions for smart offsets
        dims = self.get_object_dimensions(object_name)
        
        if dims is not None:
            x_offset = dims[0] / 2 + 0.05  # Half length + 5cm clearance
            y_offset = dims[1] / 2 + 0.05  # Half width + 5cm clearance
            z_offset = dims[2] + 0.10  # Full height + 10cm
        else:
            x_offset = 0.10
            y_offset = 0.10
            z_offset = 0.15
        
        offsets = {
            "above": np.array([0, 0, z_offset]),
            "behind": np.array([-x_offset, 0, 0]),
            "front": np.array([x_offset, 0, 0]),
            "left": np.array([0, y_offset, 0]),
            "right": np.array([0, -y_offset, 0]),
            "at": np.array([0, 0, 0]),
        }
        
        return offsets.get(positioning.lower(), np.array([0, 0, z_offset]))
