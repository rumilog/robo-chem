"""
Check Container Skill

Perception-only skill that analyzes a container's state.
Useful for checking fill levels, contents, and changes.
"""

from typing import Dict, Any, Tuple
import numpy as np

from .base_skill import BaseSkill


class CheckContainerSkill(BaseSkill):
    """
    Check the state of a container (fill level, color, etc.).
    
    This is a perception-only skill (no robot movement) that:
    1. Locates the target container
    2. Analyzes its contents (color, fill level)
    3. Compares to previous state if available
    
    Useful for:
    - Checking if a pour completed successfully
    - Verifying chemistry reaction progress (color change)
    - Monitoring fill levels
    """
    
    name = "check_container"
    required_params = ["container_name"]
    
    @property
    def optional_params(self) -> Dict[str, Any]:
        return {
            "check_color": True,  # Analyze dominant color
            "check_fill_level": True,  # Estimate fill level
            "previous_image": None,  # For comparison
        }
    
    def check_preconditions(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate parameters."""
        valid, msg = self.validate_params(params)
        if not valid:
            return False, msg
        return True, "Ready to check container"
    
    def execute(self, params: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Check the state of a container.
        
        Returns:
            (success, result) where result contains:
            - found: bool
            - fill_level: estimated fill (0.0-1.0)
            - color: dominant color RGB
            - color_category: "acidic"/"neutral"/"basic" for pH indicators
        """
        params = self.get_params_with_defaults(params)
        container_name = params["container_name"]
        check_color = params["check_color"]
        check_fill_level = params["check_fill_level"]
        
        print(f"[CheckContainer] Checking '{container_name}'...")
        
        try:
            # Step 1: Capture scene
            images = self.vision.capture_scene()
            
            if not images:
                return False, {"error": "Failed to capture images"}
            
            # Step 2: Locate container
            masks, confidences = self.vision.segment_object(images, container_name)
            
            if not masks or len(masks) == 0:
                return True, {
                    "found": False,
                    "container_name": container_name,
                    "message": f"Could not find '{container_name}'"
                }
            
            result = {
                "found": True,
                "container_name": container_name,
            }
            
            # Get the primary image and mask
            primary_cam = list(masks.keys())[0]
            image = images[0] if isinstance(images, list) else images
            mask = masks[primary_cam]
            
            # Step 3: Analyze color
            if check_color:
                print(f"[CheckContainer] Analyzing color...")
                
                # Import color detector
                from ..verification.color_detector import ColorDetector
                color_detector = ColorDetector()
                
                # Get bounding box from mask
                rows = np.any(mask, axis=1)
                cols = np.any(mask, axis=0)
                if rows.any() and cols.any():
                    rmin, rmax = np.where(rows)[0][[0, -1]]
                    cmin, cmax = np.where(cols)[0][[0, -1]]
                    region = (cmin, rmin, cmax - cmin, rmax - rmin)
                    
                    # Get dominant color
                    color = color_detector.get_dominant_color(image, region)
                    result["color"] = color
                    
                    # Classify for pH
                    color_category = color_detector.classify_ph_color(color)
                    result["color_category"] = color_category
            
            # Step 4: Analyze fill level
            if check_fill_level:
                print(f"[CheckContainer] Estimating fill level...")
                
                from ..verification.volume_detector import VolumeDetector
                volume_detector = VolumeDetector()
                
                # Get region from mask
                rows = np.any(mask, axis=1)
                cols = np.any(mask, axis=0)
                if rows.any() and cols.any():
                    rmin, rmax = np.where(rows)[0][[0, -1]]
                    cmin, cmax = np.where(cols)[0][[0, -1]]
                    region = (cmin, rmin, cmax - cmin, rmax - rmin)
                    
                    fill_level = volume_detector.measure_content_height(image, region)
                    result["fill_level"] = fill_level
            
            print(f"[CheckContainer] Analysis complete")
            return True, result
            
        except Exception as e:
            return False, {"error": str(e), "found": False}
