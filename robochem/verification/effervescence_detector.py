"""
Effervescence Detector

Detects bubbling/fizzing reactions using motion analysis.
"""

from typing import List
import numpy as np
import cv2


class EffervescenceDetector:
    """
    Detects effervescence (bubbling) using frame differencing and optical flow.
    
    Bubbles create characteristic upward motion patterns that can be
    detected through temporal analysis of video frames.
    """
    
    def __init__(self):
        """Initialize effervescence detector."""
        # Detection thresholds
        self.motion_threshold = 0.02  # Minimum motion to consider
        self.bubble_threshold = 0.5   # Score threshold for positive detection
    
    def detect_bubbles(self, frames: List[np.ndarray]) -> float:
        """
        Detect bubble formation using frame differencing and optical flow.
        
        Args:
            frames: Sequence of frames (at least 2, preferably 5-10)
            
        Returns:
            Bubble score between 0 and 1
        """
        if len(frames) < 2:
            return 0.0
        
        total_bubble_score = 0.0
        num_pairs = 0
        
        for i in range(len(frames) - 1):
            frame1 = frames[i]
            frame2 = frames[i + 1]
            
            # Convert to grayscale if needed
            if len(frame1.shape) == 3:
                gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
            else:
                gray1 = frame1
                
            if len(frame2.shape) == 3:
                gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
            else:
                gray2 = frame2
            
            # Method 1: Frame differencing
            diff_score = self._frame_diff_score(gray1, gray2)
            
            # Method 2: Optical flow (upward motion)
            flow_score = self._optical_flow_score(gray1, gray2)
            
            # Combine scores
            pair_score = 0.4 * diff_score + 0.6 * flow_score
            total_bubble_score += pair_score
            num_pairs += 1
        
        avg_score = total_bubble_score / num_pairs if num_pairs > 0 else 0.0
        
        # Normalize to 0-1 range
        return min(avg_score, 1.0)
    
    def _frame_diff_score(
        self, 
        gray1: np.ndarray, 
        gray2: np.ndarray
    ) -> float:
        """
        Compute frame difference score.
        
        Bubbles create localized brightness changes.
        
        Args:
            gray1: First grayscale frame
            gray2: Second grayscale frame
            
        Returns:
            Normalized difference score
        """
        # Compute absolute difference
        diff = cv2.absdiff(gray1, gray2)
        
        # Threshold to get significant changes
        _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
        
        # Count changed pixels
        changed_ratio = np.sum(thresh > 0) / thresh.size
        
        # Normalize (bubbling typically affects 1-10% of pixels)
        return min(changed_ratio * 10, 1.0)
    
    def _optical_flow_score(
        self, 
        gray1: np.ndarray, 
        gray2: np.ndarray
    ) -> float:
        """
        Compute optical flow score focusing on upward motion.
        
        Bubbles rise upward, creating negative Y-component flow vectors.
        
        Args:
            gray1: First grayscale frame
            gray2: Second grayscale frame
            
        Returns:
            Upward motion score
        """
        # Compute dense optical flow
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0
        )
        
        # Extract flow components
        flow_x = flow[..., 0]
        flow_y = flow[..., 1]
        
        # Compute magnitude
        magnitude = np.sqrt(flow_x**2 + flow_y**2)
        
        # Focus on significant motion
        significant_motion = magnitude > 1.0
        
        if np.sum(significant_motion) < 100:
            return 0.0
        
        # Count upward motion (negative Y in image coordinates)
        upward_motion = (flow_y < -0.5) & significant_motion
        upward_ratio = np.sum(upward_motion) / np.sum(significant_motion)
        
        # Also consider general motion activity
        motion_activity = np.mean(magnitude[significant_motion]) / 10.0
        
        # Combine: prefer upward motion but general activity also counts
        score = 0.7 * upward_ratio + 0.3 * min(motion_activity, 1.0)
        
        return score
    
    def detect_foam(self, image: np.ndarray) -> float:
        """
        Detect foam formation in a single image.
        
        Foam has characteristic texture (many small bubbles).
        
        Args:
            image: Input image
            
        Returns:
            Foam detection score
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Detect circular structures (bubbles)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=5,
            param1=50,
            param2=30,
            minRadius=2,
            maxRadius=20
        )
        
        if circles is None:
            return 0.0
        
        num_circles = len(circles[0])
        
        # Normalize by image size
        # Many small circles suggest foam
        expected_bubbles_per_megapixel = 100
        megapixels = gray.size / 1e6
        
        score = num_circles / (expected_bubbles_per_megapixel * megapixels)
        
        return min(score, 1.0)
