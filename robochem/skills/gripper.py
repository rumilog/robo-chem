"""
Gripper Skill

Controls gripper open/close operations.
"""

from typing import Dict, Any, Tuple

from .base_skill import BaseSkill


class GripperSkill(BaseSkill):
    """
    Control gripper open/close operations.
    
    This skill handles both opening and closing the gripper,
    with configurable width and force parameters.
    """
    
    name = "gripper"
    required_params = ["action"]  # "open" or "close"
    
    @property
    def optional_params(self) -> Dict[str, Any]:
        return {
            "width": 0.08,  # Target width for open (meters)
            "force": 15.0,  # Grasp force for close (N)
        }
    
    def check_preconditions(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check if gripper operation can be executed.
        """
        valid, msg = self.validate_params(params)
        if not valid:
            return False, msg
        
        action = params["action"].lower()
        if action not in ["open", "close"]:
            return False, f"Invalid gripper action: {action}. Must be 'open' or 'close'"
        
        return True, "Preconditions met"
    
    def execute(self, params: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Execute the gripper operation.
        """
        params = self.get_params_with_defaults(params)
        action = params["action"].lower()
        width = params["width"]
        force = params["force"]
        
        print(f"[Gripper] Executing gripper {action}")
        
        if action == "open":
            success = self.open_gripper(width)
            result = {"action": "open", "target_width": width}
        else:  # close
            success = self.close_gripper(force)
            result = {"action": "close", "force": force}
        
        if success:
            # Get final gripper width
            final_width = self.get_gripper_width()
            result["final_width"] = final_width
            print(f"[Gripper] {action.capitalize()} successful. Width: {final_width:.4f}m")
        else:
            result["error"] = f"Gripper {action} failed"
            print(f"[Gripper] {action.capitalize()} failed")
        
        return success, result
