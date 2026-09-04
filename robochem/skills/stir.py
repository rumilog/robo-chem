"""
Stir Skill

Stirs contents of a container using a stirring tool or the held object.
Important for mixing chemistry reagents.
"""

from typing import Dict, Any, Tuple
import numpy as np
import time

from .base_skill import BaseSkill


class StirSkill(BaseSkill):
    """
    Stir contents of a container with a stirring tool.
    
    Pipeline:
    1. Verify holding a stirring tool
    2. Localize target container
    3. Move above container center
    4. Lower into container
    5. Execute circular stirring motion
    6. Lift and retract
    """
    
    name = "stir"
    required_params = ["target_container"]
    
    @property
    def optional_params(self) -> Dict[str, Any]:
        return {
            "duration": 5.0,  # Stirring duration in seconds
            "stir_radius": 0.015,  # Radius of circular motion (meters)
            "stir_depth": 0.03,  # How deep to insert stirrer (meters)
            "stir_speed": 2.0,  # Rotations per second
        }
    
    def check_preconditions(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check if stir can be executed:
        - Gripper is holding a stirring tool
        - Target container is visible
        """
        valid, msg = self.validate_params(params)
        if not valid:
            return False, msg
        
        if not self.is_gripper_holding():
            return False, "Gripper is not holding anything. Pick up a stirrer first."
        
        target = params["target_container"]
        centroid = self.get_object_centroid(target)
        if centroid is None:
            return False, f"Cannot locate target container '{target}'"
        
        return True, "Preconditions met"
    
    def execute(self, params: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Execute the stir skill.
        """
        params = self.get_params_with_defaults(params)
        target = params["target_container"]
        duration = params["duration"]
        stir_radius = params["stir_radius"]
        stir_depth = params["stir_depth"]
        stir_speed = params["stir_speed"]
        
        print(f"[Stir] Starting stir in '{target}' for {duration}s")
        
        # Step 1: Localize target container
        target_pos = self.get_object_centroid(target)
        if target_pos is None:
            return False, {"error": f"Cannot locate '{target}'"}
        
        # Get target dimensions
        target_dims = self.get_object_dimensions(target)
        target_height = target_dims[2] if target_dims is not None else 0.05
        
        center = target_pos.copy()
        center[2] += target_height  # Top of container
        
        print(f"[Stir] Container center: {center}")
        
        # Step 2: Move above container
        print(f"[Stir] Moving above container...")
        approach_pos = center.copy()
        approach_pos[2] += 0.05  # 5cm above
        
        if not self.move_to_position(approach_pos):
            return False, {"error": "Failed to move above container"}
        
        # Step 3: Lower into container
        print(f"[Stir] Lowering into container...")
        stir_pos = center.copy()
        stir_pos[2] -= stir_depth  # Below surface
        
        if not self.move_to_position(stir_pos):
            return False, {"error": "Failed to lower into container"}
        
        # Step 4: Execute circular stirring motion
        print(f"[Stir] Stirring (radius: {stir_radius*100:.1f}cm, speed: {stir_speed} rps)...")
        
        start_time = time.time()
        rotations = 0
        
        while time.time() - start_time < duration:
            t = time.time() - start_time
            angle = 2 * np.pi * stir_speed * t
            
            x = center[0] + stir_radius * np.cos(angle)
            y = center[1] + stir_radius * np.sin(angle)
            z = stir_pos[2]
            
            # Use fast movement for smooth stirring
            try:
                current_pose = self.get_current_pose()
                current_pose.translation = np.array([x, y, z])
                self.robot.goto_pose(current_pose)  # Fast, no blocking
            except Exception as e:
                print(f"[Stir] Warning during stir motion: {e}")
            
            self.wait(0.02)  # 50Hz update rate
        
        rotations = int(duration * stir_speed)
        
        # Step 5: Return to center
        print(f"[Stir] Returning to center...")
        if not self.move_to_position(stir_pos):
            print("[Stir] Warning: Failed to return to center")
        
        # Step 6: Lift out of container
        print(f"[Stir] Lifting out...")
        lift_pos = center.copy()
        lift_pos[2] += 0.08  # 8cm above
        
        if not self.move_to_position(lift_pos):
            return False, {"error": "Failed to lift out of container"}
        
        print(f"[Stir] Successfully stirred '{target}' ({rotations} rotations)")
        
        return True, {
            "stirred_container": target,
            "duration": duration,
            "rotations": rotations,
            "stir_radius": stir_radius
        }
