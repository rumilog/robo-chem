"""
Color Detector

Detects and classifies colors for pH indicator verification.
"""

from typing import Tuple, Dict, Optional
import numpy as np
import cv2


class ColorDetector:
    """
    Detects and classifies colors for chemistry verification.
    
    Optimized for pH indicator color classification:
    - Acidic: Pink/Red tones
    - Neutral: Purple/Violet tones  
    - Basic: Blue/Green tones
    """
    
    def __init__(self):
        """Initialize color detector with reference colors."""
        # Reference colors for pH indicator (cabbage juice)
        # Values in RGB
        self.reference_colors = {
            "acidic": (220, 100, 130),    # Pink/Red
            "neutral": (160, 100, 160),   # Purple
            "basic": (100, 160, 130),     # Blue/Green
            "yellow": (220, 200, 80),     # Yellow
            "orange": (230, 150, 80),     # Orange
        }
        
        # Color matching threshold (Euclidean distance)
        self.match_threshold = 80
        
        # Minimum difference to detect color change
        self.change_threshold = 40
    
    def get_dominant_color(
        self, 
        image: np.ndarray, 
        region: Tuple[int, int, int, int] = None
    ) -> Tuple[int, int, int]:
        """
        Get dominant color in image or region.
        
        Uses K-means clustering to find the most common color.
        
        Args:
            image: Input image (HxWx3, RGB)
            region: Optional (x, y, width, height) to analyze
            
        Returns:
            Dominant color as (R, G, B) tuple
        """
        # Extract region of interest
        if region:
            x, y, w, h = region
            h_img, w_img = image.shape[:2]
            x = max(0, min(x, w_img - 1))
            y = max(0, min(y, h_img - 1))
            w = min(w, w_img - x)
            h = min(h, h_img - y)
            roi = image[y:y+h, x:x+w]
        else:
            roi = image
        
        # Reshape for clustering
        pixels = roi.reshape(-1, 3).astype(np.float32)
        
        # Remove very dark and very light pixels (likely background/reflections)
        brightness = np.mean(pixels, axis=1)
        valid_mask = (brightness > 30) & (brightness < 240)
        pixels = pixels[valid_mask]
        
        if len(pixels) < 100:
            # Not enough pixels, return average
            return tuple(int(c) for c in np.mean(roi.reshape(-1, 3), axis=0))
        
        # K-means clustering
        try:
            from sklearn.cluster import KMeans
            
            n_clusters = min(5, len(pixels) // 100)
            n_clusters = max(1, n_clusters)
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            # Get most common cluster
            counts = np.bincount(kmeans.labels_)
            dominant_idx = np.argmax(counts)
            dominant_color = kmeans.cluster_centers_[dominant_idx]
            
        except ImportError:
            # Fallback: simple mean
            dominant_color = np.mean(pixels, axis=0)
        
        return tuple(int(c) for c in dominant_color)
    
    def color_matches_category(
        self, 
        color: Tuple[int, int, int], 
        category: str
    ) -> bool:
        """
        Check if color matches a reference category.
        
        Args:
            color: RGB color tuple
            category: Category name (e.g., "acidic", "basic")
            
        Returns:
            True if color is close to the reference category
        """
        if category not in self.reference_colors:
            return False
        
        reference = self.reference_colors[category]
        distance = self._color_distance(color, reference)
        
        return distance < self.match_threshold
    
    def colors_different(
        self, 
        color1: Tuple[int, int, int], 
        color2: Tuple[int, int, int]
    ) -> bool:
        """
        Check if two colors are significantly different.
        
        Args:
            color1: First RGB color
            color2: Second RGB color
            
        Returns:
            True if colors differ by more than threshold
        """
        distance = self._color_distance(color1, color2)
        return distance > self.change_threshold
    
    def classify_ph_color(self, color: Tuple[int, int, int]) -> str:
        """
        Classify a color as acidic, neutral, or basic.
        
        Args:
            color: RGB color tuple
            
        Returns:
            Classification string
        """
        # Calculate distance to each category
        distances = {}
        for category, ref_color in self.reference_colors.items():
            if category in ["acidic", "neutral", "basic"]:
                distances[category] = self._color_distance(color, ref_color)
        
        # Return closest category
        return min(distances, key=distances.get)
    
    def _color_distance(
        self, 
        c1: Tuple[int, int, int], 
        c2: Tuple[int, int, int]
    ) -> float:
        """
        Calculate Euclidean distance between two colors.
        
        Args:
            c1: First RGB color
            c2: Second RGB color
            
        Returns:
            Euclidean distance
        """
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(c1, c2)))
    
    def get_color_histogram(
        self, 
        image: np.ndarray,
        region: Tuple[int, int, int, int] = None
    ) -> Dict[str, float]:
        """
        Get distribution of colors in image.
        
        Args:
            image: Input image
            region: Optional region to analyze
            
        Returns:
            Dict mapping category -> proportion
        """
        if region:
            x, y, w, h = region
            roi = image[y:y+h, x:x+w]
        else:
            roi = image
        
        pixels = roi.reshape(-1, 3)
        
        # Classify each pixel
        counts = {cat: 0 for cat in ["acidic", "neutral", "basic", "other"]}
        
        for pixel in pixels[::10]:  # Sample every 10th pixel
            distances = {
                cat: self._color_distance(tuple(pixel), ref)
                for cat, ref in self.reference_colors.items()
                if cat in ["acidic", "neutral", "basic"]
            }
            
            closest = min(distances, key=distances.get)
            if distances[closest] < self.match_threshold * 1.5:
                counts[closest] += 1
            else:
                counts["other"] += 1
        
        total = sum(counts.values())
        return {k: v / total for k, v in counts.items()}
