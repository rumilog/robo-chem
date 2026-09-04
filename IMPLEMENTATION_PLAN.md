# Autonomous Robotic Chemist - Implementation Plan

## Executive Summary

This project implements a **VLM-as-Orchestrator** system for wet-chemistry manipulation using a Franka Emika arm. The system reads instructions (from paper/screen), analyzes the scene, and executes chemistry tasks with closed-loop visual verification.

**Key Architecture**: Instruction Parser → Scene Analyzer → VLM Orchestrator → Skills Library → Visual Verification → Replan Loop

---

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         AUTONOMOUS ROBOTIC CHEMIST                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐      │
│  │  INSTRUCTION     │───▶│  VLM ORCHESTRATOR │───▶│  SKILLS          │      │
│  │  PARSER          │    │  (GPT-4o/Claude)  │    │  EXECUTOR        │      │
│  │  (OCR + VLM)     │    │                   │    │                  │      │
│  └──────────────────┘    └────────┬──────────┘    └────────┬─────────┘      │
│                                   │                        │                │
│                                   ▼                        ▼                │
│  ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐      │
│  │  SCENE           │◀──▶│  VISUAL          │◀───│  FRANKA ARM      │      │
│  │  COMPREHENSION   │    │  VERIFICATION    │    │  + GRIPPER       │      │
│  │  (SAM3 + PC)     │    │  (Chemistry)     │    │                  │      │
│  └──────────────────┘    └──────────────────┘    └──────────────────┘      │
│                                   │                                         │
│                                   ▼                                         │
│                          ┌──────────────────┐                               │
│                          │  REPLAN LOOP     │                               │
│                          │  (if verification│                               │
│                          │   fails)         │                               │
│                          └──────────────────┘                               │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Core Skills Library

### 1.1 Directory Structure

```
robo-chem/
├── robochem/
│   ├── __init__.py
│   ├── skills/
│   │   ├── __init__.py
│   │   ├── base_skill.py          # Abstract base class for all skills
│   │   ├── pick_up.py             # Pick up object skill
│   │   ├── place.py               # Place object skill  
│   │   ├── pour.py                # Pour liquid/powder skill
│   │   ├── scoop.py               # Scoop powder skill
│   │   ├── dispense.py            # Dispense drops (pipette) skill
│   │   ├── stir.py                # Stir/agitate skill
│   │   ├── move_to.py             # Move to position skill
│   │   └── gripper.py             # Open/close gripper skill
│   ├── vision/
│   │   ├── __init__.py
│   │   ├── scene_analyzer.py      # SAM3 + scene understanding
│   │   ├── object_localizer.py    # Pointcloud-based 3D localization
│   │   ├── grasp_analyzer.py      # PC analysis for grasp planning
│   │   └── instruction_parser.py  # OCR + VLM for reading instructions
│   ├── verification/
│   │   ├── __init__.py
│   │   ├── color_detector.py      # pH indicator color classification
│   │   ├── effervescence_detector.py  # Bubble/foam detection
│   │   ├── volume_detector.py     # Liquid level / volume change
│   │   └── chemistry_verifier.py  # Main verification orchestrator
│   ├── orchestrator/
│   │   ├── __init__.py
│   │   ├── vlm_orchestrator.py    # Main VLM planning loop
│   │   ├── task_planner.py        # High-level task decomposition
│   │   └── step_planner.py        # Low-level action generation
│   └── utils/
│       ├── __init__.py
│       ├── camera_utils.py        # Camera management
│       └── logging_utils.py       # Experiment logging
├── configs/
│   ├── robot_config.yaml
│   ├── camera_config.yaml
│   └── skill_configs/
│       ├── pour.yaml
│       ├── scoop.yaml
│       └── ...
├── scripts/
│   ├── run_experiment.py
│   └── calibrate_cameras.py
└── tests/
    └── ...
```

### 1.2 Base Skill Class

```python
# robochem/skills/base_skill.py

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
import numpy as np

class BaseSkill(ABC):
    """
    Abstract base class for all robot skills.
    
    Each skill follows the pattern:
    1. Pre-conditions check (is this skill valid to execute?)
    2. Visual grounding (where is the target object?)
    3. Grasp/motion planning (how do we interact with it?)
    4. Execution (move the robot)
    5. Post-conditions check (did it work?)
    """
    
    def __init__(self, robot_interface, vision_system, config: Dict[str, Any] = None):
        self.robot = robot_interface
        self.vision = vision_system
        self.config = config or {}
        
    @property
    @abstractmethod
    def name(self) -> str:
        """Skill name for orchestrator reference."""
        pass
    
    @property
    @abstractmethod
    def required_params(self) -> list:
        """List of required parameters for this skill."""
        pass
    
    @abstractmethod
    def check_preconditions(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Check if skill can be executed.
        Returns: (success, message)
        """
        pass
    
    @abstractmethod
    def execute(self, params: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Execute the skill.
        Returns: (success, result_data)
        """
        pass
    
    def get_object_pose(self, object_name: str) -> Optional[np.ndarray]:
        """
        Use vision system to localize an object.
        Returns 4x4 pose matrix or None if not found.
        """
        return self.vision.localize_object(object_name)
    
    def get_grasp_pose(self, object_name: str, grasp_type: str = "top") -> Optional[np.ndarray]:
        """
        Use vision system to compute optimal grasp pose.
        """
        return self.vision.compute_grasp_pose(object_name, grasp_type)
```

### 1.3 Core Skills to Implement

#### Skill 1: `pick_up`

```python
# robochem/skills/pick_up.py

class PickUpSkill(BaseSkill):
    """
    Pick up an object from the workspace.
    
    Pipeline:
    1. SAM3 segment the target object in scene images
    2. Get pointcloud of object from multi-camera fusion
    3. Analyze pointcloud for grasp affordances
    4. Plan approach trajectory
    5. Execute grasp
    """
    
    name = "pick_up"
    required_params = ["object_name"]
    
    def execute(self, params: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        object_name = params["object_name"]
        grasp_type = params.get("grasp_type", "auto")  # auto, top, side
        
        # 1. Capture scene images from all cameras
        scene_images = self.vision.capture_scene()
        
        # 2. SAM3 segmentation to find object
        masks, confidence = self.vision.segment_object(scene_images, object_name)
        if masks is None:
            return False, {"error": f"Could not find {object_name} in scene"}
        
        # 3. Get 3D pointcloud of object
        object_pc = self.vision.get_object_pointcloud(masks)
        centroid = np.mean(object_pc, axis=0)
        
        # 4. Analyze for grasp pose
        if grasp_type == "auto":
            grasp_pose = self.vision.compute_optimal_grasp(object_pc, object_name)
        else:
            grasp_pose = self.vision.compute_grasp_pose(object_pc, grasp_type)
        
        # 5. Plan and execute
        # Pre-grasp approach (10cm above)
        approach_pose = grasp_pose.copy()
        approach_pose[2, 3] += 0.10  # 10cm above
        
        self.robot.goto_pose(approach_pose)
        self.robot.open_gripper()
        self.robot.goto_pose(grasp_pose)
        self.robot.close_gripper()
        
        # Lift object
        lift_pose = grasp_pose.copy()
        lift_pose[2, 3] += 0.15
        self.robot.goto_pose(lift_pose)
        
        return True, {"grasped_object": object_name, "grasp_pose": grasp_pose}
```

#### Skill 2: `pour`

```python
# robochem/skills/pour.py

class PourSkill(BaseSkill):
    """
    Pour contents of held container into target container.
    
    Pipeline:
    1. Verify gripper is holding a container
    2. Localize target container
    3. Move above target
    4. Execute controlled pour trajectory (tilt)
    5. Return to upright position
    """
    
    name = "pour"
    required_params = ["target_container"]
    
    def execute(self, params: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        target = params["target_container"]
        pour_angle = params.get("pour_angle", 90)  # degrees
        pour_speed = params.get("pour_speed", "medium")
        
        # 1. Localize target container
        target_pose = self.get_object_pose(target)
        if target_pose is None:
            return False, {"error": f"Cannot find target container: {target}"}
        
        # 2. Move above target (offset for pour trajectory)
        pour_position = target_pose[:3, 3] + np.array([0.05, 0, 0.15])
        
        current_pose = self.robot.get_pose()
        current_pose.translation = pour_position
        self.robot.goto_pose(current_pose)
        
        # 3. Execute pour (tilt around Y axis)
        pour_angles = self._get_pour_trajectory(pour_angle, pour_speed)
        for angle in pour_angles:
            current_pose = self.robot.get_pose()
            rotated = self._rotate_y(current_pose.rotation, np.radians(angle))
            current_pose.rotation = rotated
            self.robot.goto_pose(current_pose)
        
        # 4. Hold pour position briefly
        time.sleep(1.0)
        
        # 5. Return to upright
        for angle in reversed(pour_angles):
            current_pose = self.robot.get_pose()
            rotated = self._rotate_y(current_pose.rotation, np.radians(-angle))
            current_pose.rotation = rotated
            self.robot.goto_pose(current_pose)
        
        return True, {"poured_into": target, "angle": pour_angle}
```

#### Skill 3: `scoop`

```python
# robochem/skills/scoop.py

class ScoopSkill(BaseSkill):
    """
    Scoop powder/granular material with a scoop/spoon tool.
    
    Pipeline:
    1. Verify holding scoop tool
    2. Localize powder container
    3. Approach from behind powder
    4. Lower scoop to surface level
    5. Push forward (scoop motion)
    6. Tilt up to retain material
    7. Lift
    """
    
    name = "scoop"
    required_params = ["powder_source"]
    
    def execute(self, params: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        powder_source = params["powder_source"]
        scoop_depth = params.get("scoop_depth", 0.02)  # 2cm default
        
        # Localize powder container
        source_pose = self.get_object_pose(powder_source)
        if source_pose is None:
            return False, {"error": f"Cannot find powder source: {powder_source}"}
        
        source_pos = source_pose[:3, 3]
        
        # Approach from behind
        approach_pos = source_pos + np.array([-0.10, 0, 0.05])  # 10cm behind, 5cm above
        self.robot.goto_position(approach_pos)
        
        # Lower to scoop level (inside container)
        scoop_pos = source_pos + np.array([-0.05, 0, -scoop_depth])
        self.robot.goto_position(scoop_pos)
        
        # Push forward (scoop motion)
        end_pos = source_pos + np.array([0.03, 0, -scoop_depth])
        self.robot.goto_position(end_pos)
        
        # Tilt up to retain material
        current_pose = self.robot.get_pose()
        current_pose.rotation = self._rotate_y(current_pose.rotation, np.radians(30))
        self.robot.goto_pose(current_pose)
        
        # Lift
        lift_pos = source_pos + np.array([0, 0, 0.15])
        current_pose.translation = lift_pos
        self.robot.goto_pose(current_pose)
        
        return True, {"scooped_from": powder_source}
```

#### Skill 4: `dispense` (Pipette)

```python
# robochem/skills/dispense.py

class DispenseSkill(BaseSkill):
    """
    Dispense drops from a pipette into target container.
    """
    
    name = "dispense"
    required_params = ["target_container", "num_drops"]
    
    def execute(self, params: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        target = params["target_container"]
        num_drops = params["num_drops"]
        
        # Localize target
        target_pose = self.get_object_pose(target)
        target_pos = target_pose[:3, 3]
        
        # Position above target
        dispense_pos = target_pos + np.array([0, 0, 0.08])  # 8cm above
        self.robot.goto_position(dispense_pos)
        
        # Dispense drops by squeezing (simulated via gripper micro-movements)
        for i in range(num_drops):
            # Squeeze action (gripper pulse)
            self.robot.gripper_squeeze_pulse()
            time.sleep(0.5)  # Wait between drops
        
        return True, {"dispensed_drops": num_drops, "into": target}
```

#### Skill 5: `stir`

```python
# robochem/skills/stir.py

class StirSkill(BaseSkill):
    """
    Stir contents of a container with a stirring tool.
    """
    
    name = "stir"
    required_params = ["target_container"]
    
    def execute(self, params: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        target = params["target_container"]
        stir_duration = params.get("duration", 5.0)  # seconds
        stir_radius = params.get("radius", 0.02)  # 2cm circular motion
        
        # Localize target
        target_pose = self.get_object_pose(target)
        center = target_pose[:3, 3]
        
        # Lower into container
        stir_height = center[2] - 0.03  # 3cm below surface
        self.robot.goto_position([center[0], center[1], stir_height])
        
        # Execute circular stirring motion
        start_time = time.time()
        while time.time() - start_time < stir_duration:
            t = (time.time() - start_time) * 2 * np.pi / 2  # 2 second period
            x = center[0] + stir_radius * np.cos(t)
            y = center[1] + stir_radius * np.sin(t)
            self.robot.goto_position([x, y, stir_height], speed="fast")
        
        # Return to center and lift
        self.robot.goto_position([center[0], center[1], center[2] + 0.10])
        
        return True, {"stirred": target, "duration": stir_duration}
```

### 1.4 Skills Registry

```python
# robochem/skills/__init__.py

from .base_skill import BaseSkill
from .pick_up import PickUpSkill
from .place import PlaceSkill
from .pour import PourSkill
from .scoop import ScoopSkill
from .dispense import DispenseSkill
from .stir import StirSkill
from .move_to import MoveToSkill
from .gripper import GripperSkill

SKILL_REGISTRY = {
    "pick_up": PickUpSkill,
    "place": PlaceSkill,
    "pour": PourSkill,
    "scoop": ScoopSkill,
    "dispense": DispenseSkill,
    "stir": StirSkill,
    "move_to": MoveToSkill,
    "open_gripper": GripperSkill,
    "close_gripper": GripperSkill,
}

def get_skill(skill_name: str) -> type:
    """Get skill class by name."""
    if skill_name not in SKILL_REGISTRY:
        raise ValueError(f"Unknown skill: {skill_name}. Available: {list(SKILL_REGISTRY.keys())}")
    return SKILL_REGISTRY[skill_name]
```

---

## Phase 2: Vision Pipeline

### 2.1 Scene Analyzer (SAM3 Integration)

```python
# robochem/vision/scene_analyzer.py

from segment_anything import sam_model_registry, SamPredictor
from typing import List, Dict, Tuple, Optional
import numpy as np
import cv2

class SceneAnalyzer:
    """
    Analyzes the workspace scene using SAM3 for segmentation
    and VLM for object identification.
    """
    
    def __init__(self, sam_checkpoint: str, model_type: str = "vit_h"):
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.predictor = SamPredictor(self.sam)
        self.grounding_model = None  # GroundingDINO or similar
        
    def segment_object(
        self, 
        images: List[np.ndarray], 
        object_query: str,
        cameras: List[int] = [2, 3, 4, 5]
    ) -> Dict[int, np.ndarray]:
        """
        Segment object across multiple camera views.
        
        Args:
            images: List of images from each camera
            object_query: Text description of object to find
            cameras: Camera indices
            
        Returns:
            Dict mapping camera_id -> segmentation mask
        """
        masks = {}
        confidences = {}
        
        for cam_id, image in zip(cameras, images):
            # Get bounding box from grounding model (e.g., GroundingDINO)
            bbox, confidence = self._get_bbox_from_text(image, object_query)
            
            if bbox is not None:
                # Use SAM to get precise mask from bbox
                self.predictor.set_image(image)
                mask, _, _ = self.predictor.predict(box=bbox)
                masks[cam_id] = mask[0]  # Best mask
                confidences[cam_id] = confidence
        
        return masks, confidences
    
    def _get_bbox_from_text(self, image: np.ndarray, query: str) -> Tuple[np.ndarray, float]:
        """Use GroundingDINO or similar to get bbox from text query."""
        # Placeholder - integrate with GroundingDINO
        pass
    
    def get_all_objects(self, image: np.ndarray) -> List[Dict]:
        """
        Detect and describe all objects in scene using VLM.
        Returns list of {name, bbox, confidence, has_handle}
        """
        pass
```

### 2.2 Object Localizer (Pointcloud-based)

```python
# robochem/vision/object_localizer.py

import open3d as o3d
import numpy as np
from robomail.vision import Vision3D, CameraClass

class ObjectLocalizer:
    """
    Localizes objects in 3D using multi-camera pointcloud fusion.
    Leverages existing robomail infrastructure.
    """
    
    def __init__(self, camera_ids: List[int] = [2, 3, 4, 5]):
        self.cameras = {i: CameraClass(cam_number=i) for i in camera_ids}
        self.vision_3d = Vision3D()
        
    def capture_pointclouds(self) -> Dict[int, o3d.geometry.PointCloud]:
        """Capture pointclouds from all cameras."""
        pointclouds = {}
        images = {}
        depth_images = {}
        
        for cam_id, cam in self.cameras.items():
            img, depth, pc, verts, intrinsics = cam.get_next_frame()
            pointclouds[cam_id] = pc
            images[cam_id] = img
            depth_images[cam_id] = depth
            
        return pointclouds, images, depth_images
    
    def get_fused_pointcloud(self) -> o3d.geometry.PointCloud:
        """Get fused pointcloud from all cameras in world frame."""
        pcs, _, _ = self.capture_pointclouds()
        
        # Use robomail's fusion (assumes cameras 2-5)
        fused = self.vision_3d.unnormalize_fuse_point_clouds(
            pcs[2], pcs[3], pcs[4], pcs[5]
        )
        return fused
    
    def get_object_pointcloud(
        self, 
        masks: Dict[int, np.ndarray],
        depth_images: Dict[int, np.ndarray]
    ) -> np.ndarray:
        """
        Extract object pointcloud using segmentation masks.
        
        Args:
            masks: Camera ID -> segmentation mask
            depth_images: Camera ID -> depth image
            
        Returns:
            Nx3 array of object points in world coordinates
        """
        object_points = []
        
        for cam_id, mask in masks.items():
            depth = depth_images[cam_id]
            intrinsics = self.cameras[cam_id].get_intrinsics()
            extrinsics = self.cameras[cam_id].get_cam_extrinsics()
            
            # Project masked depth to 3D
            points_cam = self._depth_to_points(depth, mask, intrinsics)
            
            # Transform to world frame
            points_world = self._transform_points(points_cam, extrinsics)
            object_points.append(points_world)
        
        # Combine and clean
        all_points = np.vstack(object_points)
        cleaned = self._remove_outliers(all_points)
        
        return cleaned
    
    def get_object_centroid(self, object_points: np.ndarray) -> np.ndarray:
        """Get centroid of object pointcloud."""
        return np.mean(object_points, axis=0)
    
    def get_object_dimensions(self, object_points: np.ndarray) -> np.ndarray:
        """Get bounding box dimensions (length, width, height) via PCA."""
        centered = object_points - np.mean(object_points, axis=0)
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Rotate to principal axes
        rotated = centered @ eigenvectors
        dimensions = rotated.max(axis=0) - rotated.min(axis=0)
        
        return np.sort(dimensions)[::-1]  # [length, width, height]
```

### 2.3 Grasp Analyzer

```python
# robochem/vision/grasp_analyzer.py

import numpy as np
from typing import Dict, Tuple, Optional

class GraspAnalyzer:
    """
    Analyzes pointclouds to determine optimal grasp poses.
    Uses geometric analysis and optionally learned grasp prediction.
    """
    
    def __init__(self, gripper_width: float = 0.08):
        self.gripper_width = gripper_width
        
    def compute_optimal_grasp(
        self, 
        object_points: np.ndarray,
        object_type: str = "unknown",
        grasp_preference: str = "auto"
    ) -> np.ndarray:
        """
        Compute optimal 4x4 grasp pose for object.
        
        Args:
            object_points: Nx3 pointcloud of object
            object_type: Semantic type (cup, bottle, spoon, etc.)
            grasp_preference: "top", "side", "handle", or "auto"
            
        Returns:
            4x4 grasp pose matrix
        """
        centroid = np.mean(object_points, axis=0)
        dimensions = self._get_principal_dimensions(object_points)
        
        if grasp_preference == "auto":
            grasp_preference = self._determine_grasp_type(object_type, dimensions)
        
        if grasp_preference == "top":
            return self._compute_top_grasp(centroid, dimensions)
        elif grasp_preference == "side":
            return self._compute_side_grasp(centroid, dimensions)
        elif grasp_preference == "handle":
            return self._compute_handle_grasp(object_points, centroid)
        
    def _compute_top_grasp(self, centroid: np.ndarray, dimensions: np.ndarray) -> np.ndarray:
        """Top-down grasp pose."""
        pose = np.eye(4)
        pose[:3, 3] = centroid
        pose[2, 3] += dimensions[2] / 2  # Offset to top
        
        # Gripper pointing down (rotate 180 around X)
        pose[:3, :3] = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ])
        
        return pose
    
    def _compute_side_grasp(self, centroid: np.ndarray, dimensions: np.ndarray) -> np.ndarray:
        """Side grasp (for tall objects like bottles)."""
        pose = np.eye(4)
        pose[:3, 3] = centroid
        
        # Gripper approaching from side
        pose[:3, :3] = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [-1, 0, 0]
        ])
        
        return pose
    
    def _compute_handle_grasp(self, points: np.ndarray, centroid: np.ndarray) -> np.ndarray:
        """Grasp by handle (for tools with handles)."""
        # Find handle as the elongated part away from centroid
        # Using PCA to find principal axis
        centered = points - centroid
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Principal axis (handle direction)
        handle_axis = eigenvectors[:, -1]
        
        # Find handle end (furthest point along principal axis)
        projections = centered @ handle_axis
        handle_idx = np.argmax(np.abs(projections))
        handle_point = points[handle_idx]
        
        pose = np.eye(4)
        pose[:3, 3] = handle_point
        
        # Align gripper with handle axis
        # ... (detailed rotation calculation)
        
        return pose
    
    def _determine_grasp_type(self, object_type: str, dimensions: np.ndarray) -> str:
        """Heuristically determine best grasp type."""
        aspect_ratio = dimensions[0] / dimensions[2] if dimensions[2] > 0 else 1
        
        # Object-specific preferences
        if object_type in ["spoon", "scoop", "spatula", "pipette"]:
            return "handle"
        elif object_type in ["bottle", "cup", "beaker"]:
            if aspect_ratio > 2:  # Tall object
                return "side"
            return "top"
        elif object_type in ["bowl", "container"]:
            return "side"
        
        # Default: if tall, side grasp; otherwise top
        return "side" if aspect_ratio > 1.5 else "top"
```

### 2.4 Instruction Parser

```python
# robochem/vision/instruction_parser.py

from openai import OpenAI
import base64
from PIL import Image
from typing import List, Dict, Tuple

class InstructionParser:
    """
    Reads and parses instructions from images (paper/screen).
    Uses OCR + VLM to extract structured task information.
    """
    
    def __init__(self, api_key: str = None):
        self.client = OpenAI(api_key=api_key)
        
    def parse_instruction_image(self, image_path: str) -> Dict:
        """
        Parse an image containing chemistry instructions.
        
        Returns:
            {
                "goal": str,  # Overall task goal
                "steps": List[str],  # Ordered steps if present
                "reagents": List[str],  # Identified reagents
                "equipment": List[str],  # Identified equipment
                "safety_notes": List[str],  # Any safety notes
                "expected_outcome": str  # Expected result description
            }
        """
        base64_image = self._encode_image(image_path)
        
        prompt = """You are analyzing an image containing chemistry experiment instructions.
        
Extract the following information in a structured format:
1. The overall goal/objective of the experiment
2. The step-by-step procedure (if shown)
3. All reagents/chemicals mentioned
4. All equipment needed
5. Any safety notes or warnings
6. The expected outcome/result

Return as JSON with keys: goal, steps, reagents, equipment, safety_notes, expected_outcome"""

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)
    
    def _encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
```

---

## Phase 3: VLM Orchestrator

### 3.1 Main Orchestrator

```python
# robochem/orchestrator/vlm_orchestrator.py

from openai import OpenAI
from typing import List, Dict, Any, Optional
import json

class VLMOrchestrator:
    """
    Main VLM-based orchestrator that:
    1. Reads instructions (from paper/verbal/text)
    2. Analyzes the scene
    3. Plans high-level tasks
    4. Decomposes into skills
    5. Executes with visual verification
    6. Replans on failure
    """
    
    def __init__(
        self, 
        skills_executor,
        vision_system,
        verifier,
        model: str = "gpt-4o"
    ):
        self.client = OpenAI()
        self.model = model
        self.skills = skills_executor
        self.vision = vision_system
        self.verifier = verifier
        
        self.conversation_history = []
        self.execution_log = []
        
    def run_task(self, task_description: str, instruction_image: str = None) -> Dict:
        """
        Main entry point for executing a chemistry task.
        
        Args:
            task_description: Natural language task (or parsed from image)
            instruction_image: Optional image of written instructions
            
        Returns:
            Execution result with success status and logs
        """
        # 1. Parse instructions if image provided
        if instruction_image:
            parsed = self.vision.instruction_parser.parse_instruction_image(instruction_image)
            task_description = parsed["goal"]
            self.execution_log.append({"step": "instruction_parse", "result": parsed})
        
        # 2. Capture and analyze initial scene
        scene_state = self._analyze_scene()
        self.execution_log.append({"step": "initial_scene", "state": scene_state})
        
        # 3. Generate high-level plan
        plan = self._generate_plan(task_description, scene_state)
        self.execution_log.append({"step": "plan_generated", "plan": plan})
        
        # 4. Execute plan with verification loop
        for i, step in enumerate(plan["steps"]):
            print(f"\n=== Executing Step {i+1}: {step['action']} ===")
            
            # Take pre-action image
            pre_images = self.vision.capture_scene()
            
            # Execute the skill
            success, result = self._execute_step(step)
            
            # Take post-action image
            post_images = self.vision.capture_scene()
            
            # Verify outcome
            verification = self._verify_step(step, pre_images, post_images)
            
            self.execution_log.append({
                "step_num": i + 1,
                "step": step,
                "success": success,
                "result": result,
                "verification": verification
            })
            
            # Handle failure - replan if needed
            if not success or not verification["success"]:
                replan_result = self._handle_failure(step, verification, scene_state)
                if not replan_result["recovered"]:
                    return {"success": False, "log": self.execution_log, "failure_point": i}
        
        # 5. Final verification of chemistry outcome
        final_verification = self.verifier.verify_chemistry_outcome(
            task_description,
            self.vision.capture_scene()
        )
        
        return {
            "success": final_verification["success"],
            "log": self.execution_log,
            "final_verification": final_verification
        }
    
    def _analyze_scene(self) -> Dict:
        """Capture scene and identify all objects."""
        images = self.vision.capture_scene()
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": """Analyze this chemistry workspace scene. Identify:
                    1. All containers (beakers, cups, bowls) and their contents if visible
                    2. All tools (spoons, pipettes, scoops, stirrers)
                    3. All reagents/chemicals with their containers
                    4. The spatial layout (what's where)
                    
                    Return as JSON with: containers, tools, reagents, layout"""
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{self._encode_image(images[0])}"}}
                    ]
                }
            ],
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)
    
    def _generate_plan(self, task: str, scene_state: Dict) -> Dict:
        """Generate high-level task plan."""
        
        available_skills = list(self.skills.SKILL_REGISTRY.keys())
        
        prompt = f"""You are planning a chemistry manipulation task for a robot arm.

Task: {task}

Current scene state:
{json.dumps(scene_state, indent=2)}

Available robot skills: {available_skills}

Generate a step-by-step plan. Each step should specify:
- action: The skill to use
- params: Parameters for the skill (object names, targets, etc.)
- expected_outcome: What should happen
- verification_type: How to verify (visual/position/chemistry)

Return as JSON with key "steps" containing the ordered list."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        return json.loads(response.choices[0].message.content)
    
    def _execute_step(self, step: Dict) -> Tuple[bool, Dict]:
        """Execute a single plan step using the skills library."""
        action = step["action"]
        params = step["params"]
        
        skill_class = self.skills.get_skill(action)
        skill = skill_class(self.robot, self.vision)
        
        return skill.execute(params)
    
    def _verify_step(self, step: Dict, pre_images: List, post_images: List) -> Dict:
        """Verify step execution using visual comparison."""
        verification_type = step.get("verification_type", "visual")
        
        if verification_type == "chemistry":
            return self.verifier.verify_chemistry_change(
                step["expected_outcome"],
                pre_images,
                post_images
            )
        else:
            return self._visual_verification(step, pre_images, post_images)
    
    def _handle_failure(self, failed_step: Dict, verification: Dict, scene_state: Dict) -> Dict:
        """Handle execution failure by replanning."""
        prompt = f"""A robot manipulation step failed.

Failed step: {json.dumps(failed_step)}
Verification result: {json.dumps(verification)}
Current scene: {json.dumps(scene_state)}

What recovery action should be taken? Options:
1. Retry the same step
2. Add a preparatory step and retry
3. Skip and continue (if safe)
4. Abort the task

Return JSON with: action, reason, modified_step (if applicable)"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        
        recovery = json.loads(response.choices[0].message.content)
        
        if recovery["action"] == "retry":
            success, result = self._execute_step(failed_step)
            return {"recovered": success, "action": "retry"}
        elif recovery["action"] == "abort":
            return {"recovered": False, "action": "abort", "reason": recovery["reason"]}
        
        return {"recovered": False, "action": recovery["action"]}
```

---

## Phase 4: Chemistry Verification Module

### 4.1 Main Verifier

```python
# robochem/verification/chemistry_verifier.py

from typing import List, Dict, Tuple
import numpy as np
import cv2

class ChemistryVerifier:
    """
    Verifies chemistry experiment outcomes using computer vision.
    Supports: color change, effervescence, volume change.
    """
    
    def __init__(self):
        self.color_detector = ColorDetector()
        self.effervescence_detector = EffervescenceDetector()
        self.volume_detector = VolumeDetector()
        
        # Reference colors for pH indicator
        self.ph_reference = {
            "acidic": (255, 100, 150),    # Pink/Red
            "neutral": (180, 100, 180),   # Purple
            "basic": (100, 180, 100)      # Blue/Green
        }
        
    def verify_chemistry_outcome(
        self, 
        expected_outcome: str,
        images: List[np.ndarray],
        pre_images: List[np.ndarray] = None
    ) -> Dict:
        """
        Main verification entry point.
        
        Args:
            expected_outcome: Description of expected result
            images: Current scene images
            pre_images: Optional pre-experiment images for comparison
            
        Returns:
            {success: bool, confidence: float, details: str}
        """
        # Determine verification type from expected outcome
        if any(word in expected_outcome.lower() for word in ["color", "blue", "green", "pink", "purple"]):
            return self.verify_color_change(expected_outcome, images, pre_images)
        elif any(word in expected_outcome.lower() for word in ["fizz", "bubble", "effervesce", "foam"]):
            return self.verify_effervescence(images)
        elif any(word in expected_outcome.lower() for word in ["expand", "swell", "volume", "grow"]):
            return self.verify_volume_change(images, pre_images)
        else:
            return self._vlm_verification(expected_outcome, images)
    
    def verify_color_change(
        self, 
        expected: str, 
        images: List[np.ndarray],
        pre_images: List[np.ndarray]
    ) -> Dict:
        """Verify pH indicator color change."""
        
        # Detect target container region
        container_region = self._detect_container_region(images[0])
        
        # Get dominant color in region
        current_color = self.color_detector.get_dominant_color(images[0], container_region)
        
        if pre_images:
            pre_color = self.color_detector.get_dominant_color(pre_images[0], container_region)
            color_changed = self.color_detector.colors_different(current_color, pre_color)
        else:
            color_changed = True
        
        # Match against expected
        if "basic" in expected.lower() or "blue" in expected.lower() or "green" in expected.lower():
            target = "basic"
        elif "acidic" in expected.lower() or "pink" in expected.lower() or "red" in expected.lower():
            target = "acidic"
        else:
            target = "neutral"
        
        matches_target = self.color_detector.color_matches_category(current_color, target, self.ph_reference)
        
        return {
            "success": color_changed and matches_target,
            "confidence": 0.9 if matches_target else 0.3,
            "details": f"Color detected: RGB{current_color}, Target: {target}",
            "current_color": current_color,
            "target_category": target
        }
    
    def verify_effervescence(self, images: List[np.ndarray]) -> Dict:
        """Verify fizzing/bubbling reaction."""
        # Need video frames, not single image
        # This should be called with a sequence of frames
        
        bubble_score = self.effervescence_detector.detect_bubbles(images)
        
        return {
            "success": bubble_score > 0.5,
            "confidence": bubble_score,
            "details": f"Effervescence score: {bubble_score:.2f}"
        }
    
    def verify_volume_change(
        self, 
        images: List[np.ndarray],
        pre_images: List[np.ndarray]
    ) -> Dict:
        """Verify volume/height change (e.g., instant snow expansion)."""
        
        pre_height = self.volume_detector.measure_content_height(pre_images[0])
        post_height = self.volume_detector.measure_content_height(images[0])
        
        ratio = post_height / pre_height if pre_height > 0 else 1.0
        
        return {
            "success": ratio > 1.5,  # Significant expansion
            "confidence": min(ratio / 3.0, 1.0),  # Scale confidence
            "details": f"Volume ratio: {ratio:.2f}x",
            "pre_height": pre_height,
            "post_height": post_height
        }
```

### 4.2 Color Detector

```python
# robochem/verification/color_detector.py

import cv2
import numpy as np
from sklearn.cluster import KMeans

class ColorDetector:
    """Detects and classifies colors in images."""
    
    def get_dominant_color(self, image: np.ndarray, region: Tuple = None) -> Tuple[int, int, int]:
        """Get dominant color in image or region."""
        if region:
            x, y, w, h = region
            roi = image[y:y+h, x:x+w]
        else:
            roi = image
        
        # Reshape for clustering
        pixels = roi.reshape(-1, 3)
        
        # K-means to find dominant color
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans.fit(pixels)
        
        # Get most common cluster
        counts = np.bincount(kmeans.labels_)
        dominant_idx = np.argmax(counts)
        dominant_color = kmeans.cluster_centers_[dominant_idx]
        
        return tuple(int(c) for c in dominant_color)
    
    def color_matches_category(
        self, 
        color: Tuple[int, int, int], 
        category: str,
        reference: Dict
    ) -> bool:
        """Check if color matches a reference category."""
        target = reference[category]
        distance = np.sqrt(sum((c1 - c2) ** 2 for c1, c2 in zip(color, target)))
        return distance < 100  # Threshold for color match
    
    def colors_different(self, c1: Tuple, c2: Tuple, threshold: float = 50) -> bool:
        """Check if two colors are significantly different."""
        distance = np.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))
        return distance > threshold
```

### 4.3 Effervescence Detector

```python
# robochem/verification/effervescence_detector.py

import cv2
import numpy as np
from typing import List

class EffervescenceDetector:
    """Detects bubbling/fizzing using optical flow."""
    
    def detect_bubbles(self, frames: List[np.ndarray]) -> float:
        """
        Detect bubble formation using frame differencing and optical flow.
        
        Args:
            frames: Sequence of frames (at least 2)
            
        Returns:
            Bubble score between 0 and 1
        """
        if len(frames) < 2:
            return 0.0
        
        total_motion = 0.0
        
        for i in range(len(frames) - 1):
            gray1 = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)
            
            # Compute optical flow
            flow = cv2.calcOpticalFlowFarneback(
                gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0
            )
            
            # Magnitude of flow vectors
            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            
            # High-motion regions suggest bubbles
            # Bubbles create upward motion patterns
            upward_flow = flow[..., 1]  # Y component (negative = upward)
            bubble_motion = np.sum(upward_flow < -1)  # Count upward motion pixels
            
            total_motion += bubble_motion / magnitude.size
        
        avg_motion = total_motion / (len(frames) - 1)
        
        # Normalize to 0-1 score
        return min(avg_motion * 10, 1.0)
```

---

## Phase 5: Integration & Main Entry Point

### 5.1 Main Execution Script

```python
# scripts/run_experiment.py

import argparse
from robochem.orchestrator import VLMOrchestrator
from robochem.vision import SceneAnalyzer, ObjectLocalizer, InstructionParser
from robochem.verification import ChemistryVerifier
from robochem.skills import SkillsExecutor
from robomail.vision import CameraClass
from frankapy import FrankaArm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, help="Task description")
    parser.add_argument("--instruction-image", type=str, help="Path to instruction image")
    parser.add_argument("--save-dir", type=str, default="./experiments")
    args = parser.parse_args()
    
    # Initialize robot
    print("Initializing Franka arm...")
    robot = FrankaArm()
    robot.reset_joints()
    robot.open_gripper()
    
    # Initialize cameras
    print("Initializing cameras...")
    cameras = {i: CameraClass(cam_number=i) for i in [2, 3, 4, 5]}
    
    # Initialize vision system
    print("Initializing vision system...")
    vision = VisionSystem(
        cameras=cameras,
        sam_checkpoint="path/to/sam_vit_h.pth"
    )
    
    # Initialize verifier
    verifier = ChemistryVerifier()
    
    # Initialize skills executor
    skills = SkillsExecutor(robot, vision)
    
    # Initialize orchestrator
    orchestrator = VLMOrchestrator(
        skills_executor=skills,
        vision_system=vision,
        verifier=verifier
    )
    
    # Run task
    print(f"\n{'='*60}")
    print(f"Starting task: {args.task or 'From instruction image'}")
    print(f"{'='*60}\n")
    
    result = orchestrator.run_task(
        task_description=args.task,
        instruction_image=args.instruction_image
    )
    
    # Log results
    print(f"\n{'='*60}")
    print(f"Task completed: {'SUCCESS' if result['success'] else 'FAILED'}")
    print(f"{'='*60}\n")
    
    # Save experiment log
    log_path = f"{args.save_dir}/experiment_{timestamp}.json"
    with open(log_path, 'w') as f:
        json.dump(result, f, indent=2)
    
    return result

if __name__ == "__main__":
    main()
```

---

## Implementation Order (Recommended)

### Week 1-2: Core Skills
1. ✅ Set up project structure
2. Implement `BaseSkill` class
3. Implement `pick_up` skill with basic vision
4. Implement `place` skill
5. Test pick-and-place pipeline

### Week 3-4: Vision Pipeline  
1. Integrate SAM3 for segmentation
2. Implement multi-camera pointcloud fusion (leverage robomail)
3. Implement grasp analyzer
4. Test full pick-up with SAM → PC → grasp pipeline

### Week 5-6: Chemistry Skills
1. Implement `pour` skill
2. Implement `scoop` skill
3. Implement `stir` skill
4. Test with actual chemistry kit items

### Week 7-8: VLM Orchestrator
1. Implement instruction parser
2. Implement scene analyzer
3. Implement task planner
4. Implement step-by-step execution loop

### Week 9-10: Verification & Integration
1. Implement color detector
2. Implement effervescence detector
3. Implement chemistry verifier
4. Full system integration testing

### Week 11+: Experiments & Paper
1. Run pH indicator experiments (Exp A)
2. Run effervescence experiments (Exp B)
3. Run powder expansion experiments (Exp C)
4. Run negative control experiments (Exp D)
5. Collect data and write paper

---

## Key Design Decisions

1. **Skill-based Architecture**: Each manipulation primitive is a self-contained skill with pre/post conditions, making the system modular and debuggable.

2. **SAM3 + Pointcloud Pipeline**: Use SAM3 for precise 2D segmentation, then project to 3D using multi-camera pointclouds for accurate localization.

3. **VLM-in-the-Loop**: The VLM (GPT-4o) is queried at each decision point, not just for planning. This enables adaptive replanning.

4. **Chemistry-Specific Verification**: Purpose-built detectors for color, bubbles, and volume changes provide reliable feedback for the closed-loop system.

5. **Leverage Existing Infrastructure**: Build on robomail's camera/pointcloud code and PLATO's planning patterns rather than rewriting.
