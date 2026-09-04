"""
Place Skill

Places a held object at a target location.
"""

from typing import Dict, Any, Tuple, List
import numpy as np

from .base_skill import BaseSkill


class PlaceSkill(BaseSkill):
    """
    Place a held object at a target location.
    
    Pipeline:
    1. Verify gripper is holding something
    2. Localize target location
    3. Move to above target
    4. Lower to place height
    5. Open gripper
    6. Retract
    """
    
    name = "place"
    required_params = ["target_location"]
    
    @property
    def optional_params(self) -> Dict[str, Any]:
        return {
            "place_height": 0.05,  # Height above target to release
            "approach_height": 0.15,  # Height for approach
            "offset": [0.0, 0.0, 0.0],  # XYZ offset from target
        }
    
    def check_preconditions(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check if place can be executed:
        - Gripper is holding something
        - Target location is valid
        """
        valid, msg = self.validate_params(params)
        if not valid:
            return False, msg
        
        # Check gripper is holding something
        if not self.is_gripper_holding():
            return False, "Gripper is not holding anything. Pick up an object first."
        
        # Check target location is valid
        target = params["target_location"]
        if isinstance(target, str):
            # It's an object name - check it's visible
            centroid = self.get_object_centroid(target)
            if centroid is None:
                return False, f"Cannot locate target '{target}' in scene"
        
        return True, "Preconditions met"
    
    def execute(self, params: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Execute the place skill.
        """
        params = self.get_params_with_defaults(params)
        target = params["target_location"]
        place_height = params["place_height"]
        approach_height = params["approach_height"]
        offset = np.array(params["offset"])
        
        print(f"[Place] Starting place at '{target}'")
        
        # Step 1: Determine target position
        if isinstance(target, str):
            # Target is an object name
            target_pos = self.get_object_centroid(target)
            if target_pos is None:
                return False, {"error": f"Cannot locate target '{target}'"}
            # Get target dimensions to place on top
            target_dims = self.get_object_dimensions(target)
            if target_dims is not None:
                # Adjust Z to be on top of target
                target_pos[2] += target_dims[2] / 2 + place_height
            else:
                target_pos[2] += place_height
        elif isinstance(target, (list, np.ndarray)):
            # Target is explicit coordinates
            target_pos = np.array(target)
        else:
            return False, {"error": f"Invalid target type: {type(target)}"}
        
        # Apply offset
        target_pos = target_pos + offset
        
        print(f"[Place] Target position: {target_pos}")
        
        # Step 2: Move to approach position (above target)
        approach_pos = target_pos.copy()
        approach_pos[2] = max(target_pos[2] + approach_height, self.safe_height)
        
        print(f"[Place] Moving to approach position...")
        if not self.move_to_position(approach_pos):
            return False, {"error": "Failed to move to approach position"}
        
        # Step 3: Move down to place position
        place_pos = target_pos.copy()
        
        print(f"[Place] Moving to place position...")
        if not self.move_to_position(place_pos):
            return False, {"error": "Failed to move to place position"}
        
        # Step 4: Open gripper to release
        print(f"[Place] Opening gripper to release...")
        if not self.open_gripper():
            return False, {"error": "Failed to open gripper"}
        
        # Brief pause for object to settle
        self.wait(0.2)
        
        # Step 5: Retract
        print(f"[Place] Retracting...")
        retract_pos = place_pos.copy()
        retract_pos[2] += approach_height
        if not self.move_to_position(retract_pos):
            return False, {"error": "Failed to retract"}
        
        print(f"[Place] Successfully placed object at '{target}'")
        
        return True, {
            "placed_at": target if isinstance(target, str) else "coordinates",
            "place_position": target_pos.tolist()
        }
