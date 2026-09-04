"""
Analyze Scene Skill

Perception-only skill that analyzes the current scene without manipulation.
Useful for understanding what's present before planning actions.
"""

from typing import Dict, Any, Tuple, List
import numpy as np

from .base_skill import BaseSkill


class AnalyzeSceneSkill(BaseSkill):
    """
    Analyze the current workspace scene.
    
    This is a perception-only skill (no robot movement) that:
    1. Captures images from all cameras
    2. Uses VLM to identify all objects, containers, tools
    3. Returns structured scene understanding
    
    Useful for:
    - Initial scene understanding before task execution
    - Re-analyzing after unexpected events
    - Verifying scene state matches expectations
    """
    
    name = "analyze_scene"
    required_params = []  # No required params - analyzes whole scene
    
    @property
    def optional_params(self) -> Dict[str, Any]:
        return {
            "focus_area": None,  # Optional: "left", "right", "center"
            "object_types": None,  # Optional: filter to specific types ["containers", "tools"]
        }
    
    def check_preconditions(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        """Scene analysis can always be performed."""
        return True, "Ready to analyze scene"
    
    def execute(self, params: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Analyze the current scene.
        
        Returns:
            (success, result) where result contains:
            - objects: List of identified objects with positions
            - containers: Containers and their contents
            - tools: Available tools
            - layout: Spatial description
        """
        params = self.get_params_with_defaults(params)
        
        print("[AnalyzeScene] Capturing scene images...")
        
        try:
            # Capture scene
            images = self.vision.capture_scene()
            
            if not images:
                return False, {"error": "Failed to capture scene images"}
            
            # Get scene description from VLM
            print("[AnalyzeScene] Analyzing with VLM...")
            scene_description = self.vision.scene_analyzer.get_scene_description(images)
            
            # Extract structured info
            result = {
                "objects": scene_description.get("containers", []) + 
                          scene_description.get("tools", []) +
                          scene_description.get("reagents", []),
                "containers": scene_description.get("containers", []),
                "tools": scene_description.get("tools", []),
                "reagents": scene_description.get("reagents", []),
                "layout": scene_description.get("layout", ""),
                "raw_description": scene_description
            }
            
            # Count objects
            total_objects = len(result["objects"])
            print(f"[AnalyzeScene] Found {total_objects} objects in scene")
            
            return True, result
            
        except Exception as e:
            return False, {"error": str(e)}
