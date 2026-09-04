"""
Wait Skill

Explicit timed pause as a skill.
Useful for waiting for reactions, settling, or timing.
"""

from typing import Dict, Any, Tuple
import time

from .base_skill import BaseSkill


class WaitSkill(BaseSkill):
    """
    Wait for a specified duration.
    
    This skill pauses execution for a set time, useful for:
    - Waiting for chemical reactions to complete
    - Allowing liquids to settle
    - Timing between additions
    - Observing changes over time
    """
    
    name = "wait"
    required_params = ["duration"]
    
    @property
    def optional_params(self) -> Dict[str, Any]:
        return {
            "reason": "",  # Optional description of why waiting
        }
    
    def check_preconditions(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate duration is positive."""
        valid, msg = self.validate_params(params)
        if not valid:
            return False, msg
        
        duration = params.get("duration", 0)
        if not isinstance(duration, (int, float)) or duration < 0:
            return False, "Duration must be a positive number"
        
        if duration > 300:  # 5 minute max
            return False, "Duration cannot exceed 300 seconds (5 minutes)"
        
        return True, "Ready to wait"
    
    def execute(self, params: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Wait for the specified duration.
        
        Returns:
            (success, result) with actual wait time
        """
        params = self.get_params_with_defaults(params)
        duration = params["duration"]
        reason = params.get("reason", "")
        
        if reason:
            print(f"[Wait] Waiting {duration}s - {reason}")
        else:
            print(f"[Wait] Waiting {duration}s...")
        
        start_time = time.time()
        time.sleep(duration)
        actual_duration = time.time() - start_time
        
        print(f"[Wait] Done waiting")
        
        return True, {
            "requested_duration": duration,
            "actual_duration": actual_duration,
            "reason": reason
        }
