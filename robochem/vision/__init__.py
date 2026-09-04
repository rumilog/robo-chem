"""
RoboChem Vision Module

Provides visual perception capabilities:
- Scene analysis using SAM3 for segmentation
- Object localization using multi-camera pointcloud fusion
- Grasp pose computation
- Instruction parsing from images
"""

from .scene_analyzer import SceneAnalyzer
from .object_localizer import ObjectLocalizer
from .grasp_analyzer import GraspAnalyzer
from .instruction_parser import InstructionParser


class VisionSystem:
    """
    Unified vision system that combines all visual perception capabilities.
    """
    
    def __init__(
        self,
        cameras: dict = None,
        sam_checkpoint: str = None,
        sam_model_type: str = "vit_h",
    ):
        """
        Initialize the vision system.
        
        Args:
            cameras: Dictionary mapping camera IDs to CameraClass instances
            sam_checkpoint: Path to SAM model checkpoint
            sam_model_type: SAM model type ("vit_h", "vit_l", "vit_b")
        """
        self.cameras = cameras or {}
        
        # Initialize components
        self.scene_analyzer = SceneAnalyzer(
            sam_checkpoint=sam_checkpoint,
            model_type=sam_model_type
        ) if sam_checkpoint else None
        
        self.object_localizer = ObjectLocalizer(
            cameras=cameras
        ) if cameras else None
        
        self.grasp_analyzer = GraspAnalyzer()
        self.instruction_parser = InstructionParser()
        
    def capture_scene(self):
        """Capture images from all cameras."""
        if self.object_localizer:
            return self.object_localizer.capture_images()
        return []
    
    def segment_object(self, images, object_query):
        """Segment an object across multiple camera views."""
        if self.scene_analyzer:
            return self.scene_analyzer.segment_object(images, object_query)
        return None, {}
    
    def get_object_pointcloud(self, masks):
        """Get 3D pointcloud of segmented object."""
        if self.object_localizer:
            return self.object_localizer.get_object_pointcloud(masks)
        return None
    
    def localize_object(self, object_name):
        """Get 4x4 pose matrix of object."""
        if self.object_localizer:
            return self.object_localizer.localize_object(object_name)
        return None
    
    def get_object_centroid(self, object_name):
        """Get 3D centroid of object."""
        if self.object_localizer:
            return self.object_localizer.get_object_centroid(object_name)
        return None
    
    def get_object_dimensions(self, object_name):
        """Get bounding box dimensions of object."""
        if self.object_localizer:
            return self.object_localizer.get_object_dimensions(object_name)
        return None
    
    def compute_grasp_pose(self, object_points, object_name=None, grasp_type="auto"):
        """Compute optimal grasp pose for object."""
        return self.grasp_analyzer.compute_optimal_grasp(
            object_points, 
            object_type=object_name,
            grasp_preference=grasp_type
        )
    
    def compute_dimensions(self, points):
        """Compute bounding box dimensions from pointcloud."""
        return self.grasp_analyzer.get_principal_dimensions(points)
