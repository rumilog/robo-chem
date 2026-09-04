"""
Scene Analyzer

Analyzes workspace scenes using SAM3 for segmentation and VLM for object identification.
"""

from typing import List, Dict, Tuple, Optional
import numpy as np
import cv2
from openai import OpenAI
import base64
import json


class SceneAnalyzer:
    """
    Analyzes the workspace scene using SAM3 for segmentation
    and VLM for object identification.
    """
    
    def __init__(self, sam_checkpoint: str = None, model_type: str = "vit_h"):
        """
        Initialize scene analyzer.
        
        Args:
            sam_checkpoint: Path to SAM model checkpoint
            model_type: SAM model type ("vit_h", "vit_l", "vit_b")
        """
        self.sam = None
        self.predictor = None
        self.sam_checkpoint = sam_checkpoint
        self.model_type = model_type
        
        self.vlm_client = OpenAI()
        
        # Lazy load SAM to avoid loading if not needed
        if sam_checkpoint:
            self._load_sam()
    
    def _load_sam(self):
        """Lazy load SAM model."""
        if self.sam is None and self.sam_checkpoint:
            try:
                from segment_anything import sam_model_registry, SamPredictor
                self.sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
                self.predictor = SamPredictor(self.sam)
                print(f"[SceneAnalyzer] Loaded SAM model ({self.model_type})")
            except ImportError:
                print("[SceneAnalyzer] Warning: segment_anything not installed")
            except Exception as e:
                print(f"[SceneAnalyzer] Warning: Failed to load SAM: {e}")
    
    def segment_object(
        self, 
        images: List[np.ndarray], 
        object_query: str,
        cameras: List[int] = None
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, float]]:
        """
        Segment object across multiple camera views.
        
        Uses GroundingDINO or VLM to get bounding box from text query,
        then SAM to get precise segmentation mask.
        
        Args:
            images: List of images from each camera
            object_query: Text description of object to find
            cameras: Camera indices (defaults to [2, 3, 4, 5])
            
        Returns:
            Tuple of (masks dict, confidences dict)
            - masks: camera_id -> segmentation mask (HxW boolean array)
            - confidences: camera_id -> detection confidence
        """
        if cameras is None:
            cameras = list(range(2, 2 + len(images)))
        
        masks = {}
        confidences = {}
        
        for cam_id, image in zip(cameras, images):
            # Get bounding box from VLM-based detection
            bbox, confidence = self._detect_object_bbox(image, object_query)
            
            if bbox is not None and confidence > 0.3:
                # Use SAM for precise segmentation
                mask = self._segment_from_bbox(image, bbox)
                if mask is not None:
                    masks[cam_id] = mask
                    confidences[cam_id] = confidence
        
        return masks, confidences
    
    def _detect_object_bbox(self, image: np.ndarray, query: str) -> Tuple[Optional[np.ndarray], float]:
        """
        Detect object bounding box using VLM.
        
        Args:
            image: Input image (HxWx3)
            query: Object description
            
        Returns:
            Tuple of (bbox [x1, y1, x2, y2], confidence)
        """
        # Encode image
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        base64_image = base64.b64encode(buffer).decode('utf-8')
        
        prompt = f"""Find the object described as "{query}" in this image.
        
If you can see the object, return the bounding box coordinates as JSON:
{{"found": true, "bbox": [x1, y1, x2, y2], "confidence": 0.0-1.0}}

The bbox coordinates should be in pixels, where (0,0) is top-left.
x1, y1 is top-left corner, x2, y2 is bottom-right corner.

If you cannot find the object, return:
{{"found": false, "confidence": 0.0}}"""

        try:
            response = self.vlm_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a precise object detector. Return only valid JSON."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }
                ],
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            if result.get("found", False):
                bbox = np.array(result["bbox"])
                confidence = result.get("confidence", 0.8)
                return bbox, confidence
            
        except Exception as e:
            print(f"[SceneAnalyzer] VLM detection error: {e}")
        
        return None, 0.0
    
    def _segment_from_bbox(self, image: np.ndarray, bbox: np.ndarray) -> Optional[np.ndarray]:
        """
        Get precise segmentation mask from bounding box using SAM.
        
        Args:
            image: Input image
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Boolean mask (HxW) or None if segmentation fails
        """
        if self.predictor is None:
            self._load_sam()
            
        if self.predictor is None:
            # Fall back to simple bbox mask if SAM not available
            h, w = image.shape[:2]
            mask = np.zeros((h, w), dtype=bool)
            x1, y1, x2, y2 = bbox.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            mask[y1:y2, x1:x2] = True
            return mask
        
        try:
            self.predictor.set_image(image)
            masks, scores, _ = self.predictor.predict(
                box=bbox,
                multimask_output=True
            )
            
            # Return best mask
            best_idx = np.argmax(scores)
            return masks[best_idx]
            
        except Exception as e:
            print(f"[SceneAnalyzer] SAM segmentation error: {e}")
            return None
    
    def get_scene_description(self, images: List[np.ndarray]) -> Dict:
        """
        Get comprehensive scene description using VLM.
        
        Args:
            images: List of scene images
            
        Returns:
            Dict with containers, tools, reagents, layout
        """
        # Use first image for scene analysis
        image = images[0] if images else None
        if image is None:
            return {}
        
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        base64_image = base64.b64encode(buffer).decode('utf-8')
        
        prompt = """Analyze this chemistry workspace scene. Identify:

1. All containers (beakers, cups, bowls, bottles) and their contents if visible
2. All tools (spoons, pipettes, scoops, stirrers)
3. All reagents/chemicals with their containers
4. The spatial layout (what's where relative to other objects)

Return as JSON with keys:
- containers: List of {name, contents, position}
- tools: List of {name, has_handle, position}
- reagents: List of {name, container, state (powder/liquid/solid)}
- layout: Description of spatial arrangement"""

        try:
            response = self.vlm_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are analyzing a chemistry workspace. Be precise and thorough."},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }
                ],
                response_format={"type": "json_object"}
            )
            
            return json.loads(response.choices[0].message.content)
            
        except Exception as e:
            print(f"[SceneAnalyzer] Scene description error: {e}")
            return {}
