"""
Chemistry Verifier

Main verification module for chemistry experiment outcomes.
"""

from typing import List, Dict, Optional
import numpy as np

from .color_detector import ColorDetector
from .effervescence_detector import EffervescenceDetector
from .volume_detector import VolumeDetector


class ChemistryVerifier:
    """
    Verifies chemistry experiment outcomes using computer vision.
    
    Supports verification types:
    - Color change (pH indicators, reactions)
    - Effervescence (bubbling, gas evolution)
    - Volume change (expansion, precipitation)
    """
    
    def __init__(self):
        """Initialize chemistry verifier with detector modules."""
        self.color_detector = ColorDetector()
        self.effervescence_detector = EffervescenceDetector()
        self.volume_detector = VolumeDetector()
    
    def verify_chemistry_outcome(
        self, 
        expected_outcome: str,
        current_images: List[np.ndarray],
        pre_images: List[np.ndarray] = None
    ) -> Dict:
        """
        Verify chemistry outcome based on expected result.
        
        Automatically determines verification type from expected outcome description.
        
        Args:
            expected_outcome: Description of expected result
            current_images: Current scene images
            pre_images: Optional pre-experiment images for comparison
            
        Returns:
            {
                "success": bool,
                "confidence": float,
                "verification_type": str,
                "details": str,
                ...additional type-specific data
            }
        """
        # Determine verification type
        outcome_lower = expected_outcome.lower()
        
        # Color-related keywords
        color_keywords = ["color", "blue", "green", "pink", "purple", "red", 
                         "yellow", "orange", "basic", "acidic", "indicator"]
        
        # Effervescence keywords
        effervescence_keywords = ["fizz", "bubble", "effervesce", "foam", 
                                  "gas", "react", "evolve"]
        
        # Volume keywords
        volume_keywords = ["expand", "swell", "grow", "volume", "increase",
                          "shrink", "dissolve", "precipitate"]
        
        if any(kw in outcome_lower for kw in color_keywords):
            return self.verify_color_change(expected_outcome, current_images, pre_images)
        
        elif any(kw in outcome_lower for kw in effervescence_keywords):
            return self.verify_effervescence(current_images)
        
        elif any(kw in outcome_lower for kw in volume_keywords):
            return self.verify_volume_change(current_images, pre_images)
        
        else:
            # Fall back to VLM-based verification
            return self._vlm_verification(expected_outcome, current_images)
    
    def verify_color_change(
        self,
        expected: str,
        current_images: List[np.ndarray],
        pre_images: List[np.ndarray] = None
    ) -> Dict:
        """
        Verify color change (e.g., pH indicator).
        
        Args:
            expected: Expected color/state description
            current_images: Current scene images
            pre_images: Pre-change images for comparison
            
        Returns:
            Verification result dict
        """
        # Use first image
        image = current_images[0] if current_images else None
        if image is None:
            return {"success": False, "error": "No image provided"}
        
        # Detect container region (could be enhanced with object detection)
        # For now, analyze central region
        h, w = image.shape[:2]
        region = (w//4, h//4, w//2, h//2)
        
        # Get current color
        current_color = self.color_detector.get_dominant_color(image, region)
        
        # Check for color change if pre-images available
        color_changed = True
        if pre_images:
            pre_color = self.color_detector.get_dominant_color(pre_images[0], region)
            color_changed = self.color_detector.colors_different(current_color, pre_color)
        
        # Determine target category
        target = self._extract_color_target(expected)
        
        # Check if current color matches target
        matches = self.color_detector.color_matches_category(current_color, target)
        
        return {
            "success": color_changed and matches,
            "confidence": 0.85 if matches else 0.3,
            "verification_type": "color_change",
            "details": f"Color detected: RGB{current_color}, Target: {target}",
            "current_color": current_color,
            "target_category": target,
            "color_changed": color_changed
        }
    
    def verify_effervescence(
        self,
        frames: List[np.ndarray]
    ) -> Dict:
        """
        Verify effervescence (bubbling reaction).
        
        Args:
            frames: Sequence of frames to analyze for motion
            
        Returns:
            Verification result dict
        """
        if len(frames) < 2:
            return {
                "success": False, 
                "error": "Need multiple frames for effervescence detection"
            }
        
        bubble_score = self.effervescence_detector.detect_bubbles(frames)
        
        return {
            "success": bubble_score > 0.5,
            "confidence": bubble_score,
            "verification_type": "effervescence",
            "details": f"Effervescence score: {bubble_score:.2f}",
            "bubble_score": bubble_score
        }
    
    def verify_volume_change(
        self,
        current_images: List[np.ndarray],
        pre_images: List[np.ndarray]
    ) -> Dict:
        """
        Verify volume/height change.
        
        Args:
            current_images: Current scene images
            pre_images: Pre-change images
            
        Returns:
            Verification result dict
        """
        if not pre_images:
            return {
                "success": False,
                "error": "Need pre-change images for volume comparison"
            }
        
        # Measure heights
        pre_height = self.volume_detector.measure_content_height(pre_images[0])
        post_height = self.volume_detector.measure_content_height(current_images[0])
        
        # Calculate ratio
        if pre_height > 0:
            ratio = post_height / pre_height
        else:
            ratio = 1.0
        
        # Significant change threshold
        success = ratio > 1.3 or ratio < 0.7  # 30% change
        
        return {
            "success": success,
            "confidence": min(abs(ratio - 1.0), 1.0),
            "verification_type": "volume_change",
            "details": f"Volume ratio: {ratio:.2f}x",
            "pre_height": pre_height,
            "post_height": post_height,
            "ratio": ratio
        }
    
    def _extract_color_target(self, expected: str) -> str:
        """Extract target color category from description."""
        expected_lower = expected.lower()
        
        if any(w in expected_lower for w in ["basic", "blue", "green"]):
            return "basic"
        elif any(w in expected_lower for w in ["acidic", "pink", "red"]):
            return "acidic"
        elif "yellow" in expected_lower:
            return "yellow"
        elif "orange" in expected_lower:
            return "orange"
        elif "purple" in expected_lower:
            return "neutral"
        
        return "unknown"
    
    def _vlm_verification(
        self,
        expected: str,
        images: List[np.ndarray]
    ) -> Dict:
        """
        Fall back to VLM-based verification.
        
        Args:
            expected: Expected outcome description
            images: Scene images
            
        Returns:
            Verification result dict
        """
        # This would use GPT-4V to verify the outcome
        # Simplified placeholder
        return {
            "success": True,  # Assume success for now
            "confidence": 0.5,
            "verification_type": "vlm",
            "details": f"VLM verification for: {expected}"
        }
