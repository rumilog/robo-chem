"""
Volume Detector

Detects volume/height changes in containers.
"""

from typing import Tuple, Optional
import numpy as np
import cv2


class VolumeDetector:
    """
    Detects volume and height changes in containers.
    
    Used for verifying:
    - Liquid addition/removal
    - Powder expansion (instant snow)
    - Precipitation
    - Dissolution
    """
    
    def __init__(self):
        """Initialize volume detector."""
        pass
    
    def measure_content_height(
        self, 
        image: np.ndarray,
        container_region: Tuple[int, int, int, int] = None
    ) -> float:
        """
        Measure the height of contents in a container.
        
        Uses edge detection and color analysis to find the
        content/air boundary.
        
        Args:
            image: Input image
            container_region: Optional (x, y, w, h) of container
            
        Returns:
            Relative height (0-1) or pixel height if region provided
        """
        # If no region specified, use center of image
        if container_region is None:
            h, w = image.shape[:2]
            container_region = (w//4, h//4, w//2, h//2)
        
        x, y, w, h = container_region
        roi = image[y:y+h, x:x+w]
        
        # Convert to grayscale
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        else:
            gray = roi
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find horizontal edges (content boundaries)
        kernel = np.ones((1, 5), np.uint8)
        horizontal_edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, kernel)
        
        # Find the highest significant horizontal edge
        # (represents top of content)
        row_sums = np.sum(horizontal_edges, axis=1)
        threshold = np.max(row_sums) * 0.3
        
        significant_rows = np.where(row_sums > threshold)[0]
        
        if len(significant_rows) == 0:
            return 0.5  # Default to middle
        
        # Top edge of content
        top_edge = significant_rows[0]
        
        # Height relative to container
        relative_height = 1.0 - (top_edge / h)
        
        return relative_height
    
    def measure_volume_change(
        self,
        pre_image: np.ndarray,
        post_image: np.ndarray,
        container_region: Tuple[int, int, int, int] = None
    ) -> Tuple[float, float, float]:
        """
        Measure volume change between two images.
        
        Args:
            pre_image: Image before change
            post_image: Image after change
            container_region: Optional container region
            
        Returns:
            Tuple of (pre_height, post_height, ratio)
        """
        pre_height = self.measure_content_height(pre_image, container_region)
        post_height = self.measure_content_height(post_image, container_region)
        
        if pre_height > 0:
            ratio = post_height / pre_height
        else:
            ratio = float('inf') if post_height > 0 else 1.0
        
        return pre_height, post_height, ratio
    
    def detect_precipitation(
        self,
        pre_image: np.ndarray,
        post_image: np.ndarray,
        container_region: Tuple[int, int, int, int] = None
    ) -> float:
        """
        Detect precipitation (sediment formation at bottom).
        
        Args:
            pre_image: Image before reaction
            post_image: Image after reaction
            container_region: Container region
            
        Returns:
            Precipitation score (0-1)
        """
        if container_region is None:
            h, w = pre_image.shape[:2]
            container_region = (w//4, h//4, w//2, h//2)
        
        x, y, w, h = container_region
        
        # Analyze bottom third of container
        bottom_region = (x, y + 2*h//3, w, h//3)
        bx, by, bw, bh = bottom_region
        
        pre_bottom = pre_image[by:by+bh, bx:bx+bw]
        post_bottom = post_image[by:by+bh, bx:bx+bw]
        
        # Compare brightness/density at bottom
        pre_brightness = np.mean(pre_bottom)
        post_brightness = np.mean(post_bottom)
        
        # Precipitation typically makes bottom darker/denser
        brightness_change = (pre_brightness - post_brightness) / 255.0
        
        # Also check for texture change (sediment has different texture)
        pre_std = np.std(pre_bottom)
        post_std = np.std(post_bottom)
        texture_change = abs(post_std - pre_std) / max(pre_std, 1)
        
        # Combine scores
        score = 0.6 * max(brightness_change, 0) + 0.4 * min(texture_change, 1)
        
        return min(score, 1.0)
    
    def detect_layer_separation(
        self,
        image: np.ndarray,
        container_region: Tuple[int, int, int, int] = None
    ) -> int:
        """
        Detect number of distinct layers in container.
        
        Args:
            image: Input image
            container_region: Container region
            
        Returns:
            Number of detected layers
        """
        if container_region is None:
            h, w = image.shape[:2]
            container_region = (w//4, h//4, w//2, h//2)
        
        x, y, w, h = container_region
        roi = image[y:y+h, x:x+w]
        
        # Compute vertical color profile
        if len(roi.shape) == 3:
            # Average color per row
            profile = np.mean(roi, axis=(1, 2))
        else:
            profile = np.mean(roi, axis=1)
        
        # Find significant changes (layer boundaries)
        gradient = np.abs(np.diff(profile))
        threshold = np.mean(gradient) + 2 * np.std(gradient)
        
        boundaries = np.where(gradient > threshold)[0]
        
        # Merge nearby boundaries
        if len(boundaries) == 0:
            return 1
        
        merged = [boundaries[0]]
        for b in boundaries[1:]:
            if b - merged[-1] > h // 10:  # Minimum 10% container height between layers
                merged.append(b)
        
        return len(merged) + 1  # Number of layers = boundaries + 1
