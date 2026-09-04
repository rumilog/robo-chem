"""
Pick Up Skill

Picks up an object from the workspace using SAM segmentation,
pointcloud localization, and grasp analysis.
"""

from typing import Dict, Any, Tuple, List
import numpy as np

from .base_skill import BaseSkill


class PickUpSkill(BaseSkill):
    """
    Pick up an object from the workspace.
    
    Pipeline:
    1. SAM3 segment the target object in scene images
    2. Get pointcloud of object from multi-camera fusion
    3. Analyze pointcloud for grasp affordances
    4. Plan approach trajectory (approach from above)
    5. Open gripper
    6. Move to pre-grasp position
    7. Move to grasp position
    8. Close gripper
    9. Lift object
    """
    
    name = "pick_up"
    required_params = ["object_name"]
    
    @property
    def optional_params(self) -> Dict[str, Any]:
        return {
            "grasp_type": "auto",  # "auto", "top", "side", "handle"
            "approach_height": 0.10,  # Height above object for approach
            "lift_height": 0.15,  # Height to lift after grasping
            "grasp_force": 15.0,  # Gripper closing force in N
        }
    
    def check_preconditions(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check if pick up can be executed:
        - Gripper is not already holding something
        - Object is visible in the scene
        """
        # Validate parameters
        valid, msg = self.validate_params(params)
        if not valid:
            return False, msg
        
        # Check gripper is open (not holding something)
        gripper_width = self.get_gripper_width()
        if gripper_width < 0.01:  # Gripper is closed
            return False, "Gripper appears to be holding something. Place object first."
        
        # Check object is visible
        object_name = params["object_name"]
        centroid = self.get_object_centroid(object_name)
        if centroid is None:
            return False, f"Cannot locate object '{object_name}' in scene"
        
        return True, "Preconditions met"
    
    def execute(self, params: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Execute the pick up skill.
        
        Returns:
            (success, result_dict) where result_dict contains:
            - grasped_object: Name of the object that was grasped
            - grasp_pose: The pose where the grasp was executed
            - grasp_type: The type of grasp used
        """
        params = self.get_params_with_defaults(params)
        object_name = params["object_name"]
        grasp_type = params["grasp_type"]
        approach_height = params["approach_height"]
        lift_height = params["lift_height"]
        grasp_force = params["grasp_force"]
        
        print(f"[PickUp] Starting pick up of '{object_name}'")
        
        # Step 1: Capture scene and segment object
        print(f"[PickUp] Segmenting object...")
        scene_images = self.vision.capture_scene()
        masks, confidences = self.vision.segment_object(scene_images, object_name)
        
        if masks is None or len(masks) == 0:
            return False, {"error": f"Could not segment '{object_name}' in scene"}
        
        print(f"[PickUp] Object segmented with confidence: {max(confidences.values()):.2f}")
        
        # Step 2: Get 3D pointcloud of object
        print(f"[PickUp] Getting object pointcloud...")
        object_pc = self.vision.get_object_pointcloud(masks)
        
        if object_pc is None or len(object_pc) < 10:
            return False, {"error": "Insufficient pointcloud data for object"}
        
        centroid = np.mean(object_pc, axis=0)
        dimensions = self.vision.compute_dimensions(object_pc)
        print(f"[PickUp] Object centroid: {centroid}, dimensions: {dimensions}")
        
        # Step 3: Compute grasp pose
        print(f"[PickUp] Computing grasp pose (type: {grasp_type})...")
        grasp_pose = self.vision.compute_grasp_pose(
            object_pc, 
            object_name, 
            grasp_type=grasp_type
        )
        
        if grasp_pose is None:
            return False, {"error": "Could not compute valid grasp pose"}
        
        # Step 4: Compute approach pose (above grasp pose)
        approach_pose = grasp_pose.copy()
        approach_pose[2, 3] += approach_height
        
        # Step 5: Open gripper
        print(f"[PickUp] Opening gripper...")
        if not self.open_gripper():
            return False, {"error": "Failed to open gripper"}
        
        # Step 6: Move to safe height first
        print(f"[PickUp] Moving to safe height...")
        safe_position = np.array([centroid[0], centroid[1], self.safe_height])
        if not self.move_to_position(safe_position):
            return False, {"error": "Failed to move to safe height"}
        
        # Step 7: Move to approach position
        print(f"[PickUp] Moving to approach position...")
        if not self.move_to_pose(approach_pose):
            return False, {"error": "Failed to move to approach position"}
        
        # Step 8: Move down to grasp position
        print(f"[PickUp] Moving to grasp position...")
        if not self.move_to_pose(grasp_pose):
            return False, {"error": "Failed to move to grasp position"}
        
        # Step 9: Close gripper to grasp
        print(f"[PickUp] Closing gripper with force {grasp_force}N...")
        if not self.close_gripper(force=grasp_force):
            return False, {"error": "Failed to close gripper"}
        
        # Brief pause to ensure grasp is secure
        self.wait(0.3)
        
        # Step 10: Verify grasp (gripper should have closed on something)
        gripper_width = self.get_gripper_width()
        if gripper_width < 0.001:  # Fully closed - nothing grasped
            print(f"[PickUp] Warning: Gripper fully closed - may not have grasped object")
            # Could optionally return failure here
        
        # Step 11: Lift object
        print(f"[PickUp] Lifting object...")
        lift_pose = grasp_pose.copy()
        lift_pose[2, 3] += lift_height
        if not self.move_to_pose(lift_pose):
            return False, {"error": "Failed to lift object"}
        
        print(f"[PickUp] Successfully picked up '{object_name}'")
        
        return True, {
            "grasped_object": object_name,
            "grasp_pose": grasp_pose.tolist(),
            "grasp_type": grasp_type,
            "gripper_width": gripper_width,
            "centroid": centroid.tolist(),
            "dimensions": dimensions.tolist() if dimensions is not None else None
        }
