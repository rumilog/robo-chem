"""
Scoop Skill

Scoops powder or granular material using a scoop/spoon tool.
Critical for handling dry chemistry reagents.
"""

from typing import Dict, Any, Tuple, List
import numpy as np

from .base_skill import BaseSkill


class ScoopSkill(BaseSkill):
    """
    Scoop powder/granular material with a scoop or spoon tool.
    
    Pipeline:
    1. Verify holding a scoop-type tool
    2. Localize powder source container
    3. Approach from behind the powder
    4. Lower scoop to surface level
    5. Push forward (scooping motion)
    6. Tilt up slightly to retain material
    7. Lift
    """
    
    name = "scoop"
    required_params = ["powder_source"]
    
    @property
    def optional_params(self) -> Dict[str, Any]:
        return {
            "scoop_depth": 0.02,  # How deep to scoop (meters)
            "scoop_distance": 0.05,  # Forward distance of scoop motion
            "approach_offset": 0.08,  # Distance behind the target to approach from
            "lift_angle": 25,  # Degrees to tilt up after scooping
            "lift_height": 0.10,  # Height to lift after scooping
        }
    
    def check_preconditions(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check if scoop can be executed:
        - Gripper is holding a scoop-type tool
        - Powder source is visible
        """
        valid, msg = self.validate_params(params)
        if not valid:
            return False, msg
        
        if not self.is_gripper_holding():
            return False, "Gripper is not holding anything. Pick up a scoop/spoon first."
        
        source = params["powder_source"]
        centroid = self.get_object_centroid(source)
        if centroid is None:
            return False, f"Cannot locate powder source '{source}'"
        
        return True, "Preconditions met"
    
    def execute(self, params: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Execute the scoop skill.
        """
        params = self.get_params_with_defaults(params)
        powder_source = params["powder_source"]
        scoop_depth = params["scoop_depth"]
        scoop_distance = params["scoop_distance"]
        approach_offset = params["approach_offset"]
        lift_angle = params["lift_angle"]
        lift_height = params["lift_height"]
        
        print(f"[Scoop] Starting scoop from '{powder_source}'")
        
        # Step 1: Localize powder source
        source_pos = self.get_object_centroid(powder_source)
        if source_pos is None:
            return False, {"error": f"Cannot locate '{powder_source}'"}
        
        # Get source dimensions for better positioning
        source_dims = self.get_object_dimensions(powder_source)
        source_height = source_dims[2] if source_dims is not None else 0.03
        
        print(f"[Scoop] Source position: {source_pos}")
        
        # Step 2: Calculate approach position (behind the powder)
        approach_pos = source_pos.copy()
        approach_pos[0] -= approach_offset  # Behind in X direction
        approach_pos[2] += source_height + 0.05  # Above the source
        
        # Step 3: Move to approach position
        print(f"[Scoop] Moving to approach position...")
        if not self.move_to_position(approach_pos):
            return False, {"error": "Failed to move to approach position"}
        
        # Ensure scoop is level (parallel to table)
        # Note: May need to reset wrist orientation here depending on current state
        
        # Step 4: Lower to scooping level
        print(f"[Scoop] Lowering to scoop level...")
        scoop_start_pos = source_pos.copy()
        scoop_start_pos[0] -= approach_offset / 2  # Still slightly behind
        scoop_start_pos[2] = source_pos[2] - scoop_depth + source_height
        
        if not self.move_to_position(scoop_start_pos):
            return False, {"error": "Failed to lower to scoop level"}
        
        # Step 5: Execute forward scooping motion
        print(f"[Scoop] Executing scoop motion...")
        scoop_end_pos = scoop_start_pos.copy()
        scoop_end_pos[0] += scoop_distance  # Move forward
        
        if not self.move_to_position(scoop_end_pos):
            return False, {"error": "Failed during scoop motion"}
        
        # Step 6: Tilt up to retain material
        print(f"[Scoop] Tilting up to retain material...")
        if not self.rotate_wrist(lift_angle, axis="y"):
            return False, {"error": "Failed to tilt scoop up"}
        
        # Step 7: Lift while maintaining tilt
        print(f"[Scoop] Lifting scoop...")
        current_pose = self.get_current_pose()
        lift_pos = current_pose.translation.copy()
        lift_pos[2] += lift_height
        current_pose.translation = lift_pos
        
        if not self.move_to_pose(current_pose):
            return False, {"error": "Failed to lift scoop"}
        
        print(f"[Scoop] Successfully scooped from '{powder_source}'")
        
        return True, {
            "scooped_from": powder_source,
            "scoop_depth": scoop_depth,
            "lift_angle": lift_angle
        }
