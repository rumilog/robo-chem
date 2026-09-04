"""
Locate Object Skill

Perception-only skill that finds and localizes a specific object.
Uses SAM segmentation and pointcloud analysis without moving the robot.
"""

from typing import Dict, Any, Tuple, Optional
import numpy as np

from .base_skill import BaseSkill


class LocateObjectSkill(BaseSkill):
    """
    Locate a specific object in the workspace.
    
    This is a perception-only skill (no robot movement) that:
    1. Segments the target object using SAM
    2. Builds 3D pointcloud from multi-camera views
    3. Computes centroid, dimensions, and suggested grasp poses
    
    Useful for:
    - Finding where something is before picking it up
    - Checking if an object is present
    - Getting precise 3D location for planning
    """
    
    name = "locate_object"
    required_params = ["object_name"]
    
    @property
    def optional_params(self) -> Dict[str, Any]:
        return {
            "compute_grasp": True,  # Also compute grasp poses?
            "save_pointcloud": False,  # Save pointcloud to file?
        }
    
    def check_preconditions(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate parameters."""
        valid, msg = self.validate_params(params)
        if not valid:
            return False, msg
        return True, "Ready to locate object"
    
    def execute(self, params: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Locate the specified object.
        
        Returns:
            (success, result) where result contains:
            - found: bool
            - centroid: [x, y, z] position
            - dimensions: [length, width, height]
            - confidence: detection confidence
            - grasp_poses: suggested grasp poses (if compute_grasp=True)
        """
        params = self.get_params_with_defaults(params)
        object_name = params["object_name"]
        compute_grasp = params["compute_grasp"]
        
        print(f"[LocateObject] Looking for '{object_name}'...")
        
        try:
            # Step 1: Capture scene
            images = self.vision.capture_scene()
            
            if not images:
                return False, {"error": "Failed to capture images", "found": False}
            
            # Step 2: Segment object using SAM
            print(f"[LocateObject] Segmenting '{object_name}'...")
            masks, confidences = self.vision.segment_object(images, object_name)
            
            if not masks or len(masks) == 0:
                print(f"[LocateObject] Object '{object_name}' not found in scene")
                return True, {
                    "found": False,
                    "object_name": object_name,
                    "message": f"Could not find '{object_name}' in the scene"
                }
            
            # Step 3: Get 3D pointcloud
            print(f"[LocateObject] Building 3D pointcloud...")
            object_pc = self.vision.get_object_pointcloud(masks)
            
            if object_pc is None or len(object_pc) < 10:
                return True, {
                    "found": False,
                    "object_name": object_name,
                    "message": "Object detected but insufficient depth data"
                }
            
            # Step 4: Compute centroid and dimensions
            centroid = np.mean(object_pc, axis=0)
            dimensions = self.vision.compute_dimensions(object_pc)
            
            # Get best confidence score
            best_confidence = max(confidences.values()) if confidences else 0.0
            
            result = {
                "found": True,
                "object_name": object_name,
                "centroid": centroid.tolist(),
                "dimensions": dimensions.tolist() if dimensions is not None else None,
                "confidence": best_confidence,
                "num_points": len(object_pc),
            }
            
            # Step 5: Compute grasp poses if requested
            if compute_grasp:
                print(f"[LocateObject] Computing grasp poses...")
                
                grasp_poses = {}
                for grasp_type in ["top", "side", "handle"]:
                    try:
                        pose = self.vision.compute_grasp_pose(
                            object_pc, object_name, grasp_type=grasp_type
                        )
                        if pose is not None:
                            grasp_poses[grasp_type] = pose.tolist()
                    except:
                        pass
                
                result["grasp_poses"] = grasp_poses
            
            # Cache the result for quick access by other skills
            self.vision.object_localizer.cache_object(
                object_name, object_pc, centroid
            )
            
            print(f"[LocateObject] Found '{object_name}' at {centroid}")
            
            return True, result
            
        except Exception as e:
            return False, {"error": str(e), "found": False}
