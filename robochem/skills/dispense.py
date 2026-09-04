"""
Dispense Skill

Dispenses drops from a pipette or dropper into a target container.
Critical for precise liquid chemistry operations.
"""

from typing import Dict, Any, Tuple
import numpy as np

from .base_skill import BaseSkill


class DispenseSkill(BaseSkill):
    """
    Dispense drops from a pipette/dropper into target container.
    
    Note: This skill assumes the pipette has a squeeze bulb that can be
    actuated by the gripper. The number of "drops" is simulated by
    gripper micro-movements (squeeze pulses).
    
    Pipeline:
    1. Verify holding a pipette/dropper
    2. Localize target container
    3. Position above target opening
    4. Execute squeeze pulses to dispense drops
    """
    
    name = "dispense"
    required_params = ["target_container", "num_drops"]
    
    @property
    def optional_params(self) -> Dict[str, Any]:
        return {
            "drop_interval": 0.5,  # Seconds between drops
            "dispense_height": 0.05,  # Height above container to dispense
            "squeeze_amount": 0.002,  # Gripper movement for each drop (meters)
        }
    
    def check_preconditions(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check if dispense can be executed:
        - Gripper is holding a pipette/dropper
        - Target container is visible
        - num_drops is positive
        """
        valid, msg = self.validate_params(params)
        if not valid:
            return False, msg
        
        if not self.is_gripper_holding():
            return False, "Gripper is not holding anything. Pick up a pipette first."
        
        num_drops = params["num_drops"]
        if not isinstance(num_drops, int) or num_drops < 1:
            return False, "num_drops must be a positive integer"
        
        target = params["target_container"]
        centroid = self.get_object_centroid(target)
        if centroid is None:
            return False, f"Cannot locate target container '{target}'"
        
        return True, "Preconditions met"
    
    def execute(self, params: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Execute the dispense skill.
        """
        params = self.get_params_with_defaults(params)
        target = params["target_container"]
        num_drops = params["num_drops"]
        drop_interval = params["drop_interval"]
        dispense_height = params["dispense_height"]
        squeeze_amount = params["squeeze_amount"]
        
        print(f"[Dispense] Starting dispense of {num_drops} drops into '{target}'")
        
        # Step 1: Localize target container
        target_pos = self.get_object_centroid(target)
        if target_pos is None:
            return False, {"error": f"Cannot locate '{target}'"}
        
        # Get target dimensions
        target_dims = self.get_object_dimensions(target)
        target_height = target_dims[2] if target_dims is not None else 0.05
        
        # Step 2: Calculate dispense position (centered above target opening)
        dispense_pos = target_pos.copy()
        dispense_pos[2] += target_height + dispense_height
        
        # Step 3: Move to dispense position
        print(f"[Dispense] Moving to dispense position...")
        if not self.move_to_position(dispense_pos):
            return False, {"error": "Failed to move to dispense position"}
        
        # Ensure pipette is pointing down
        # May need to adjust orientation here
        
        # Step 4: Dispense drops
        print(f"[Dispense] Dispensing {num_drops} drops...")
        initial_width = self.get_gripper_width()
        drops_dispensed = 0
        
        for i in range(num_drops):
            print(f"[Dispense] Drop {i+1}/{num_drops}")
            
            # Squeeze (close gripper slightly)
            current_width = self.get_gripper_width()
            squeeze_width = max(current_width - squeeze_amount, 0.001)
            
            try:
                self.robot.goto_gripper(width=squeeze_width)
                self.wait(0.1)  # Brief hold
                
                # Release (return to previous width)
                self.robot.goto_gripper(width=current_width)
                drops_dispensed += 1
                
            except Exception as e:
                print(f"[Dispense] Warning: Squeeze failed: {e}")
            
            # Wait between drops
            if i < num_drops - 1:
                self.wait(drop_interval)
        
        print(f"[Dispense] Successfully dispensed {drops_dispensed} drops into '{target}'")
        
        return True, {
            "target": target,
            "drops_requested": num_drops,
            "drops_dispensed": drops_dispensed,
            "dispense_position": dispense_pos.tolist()
        }
