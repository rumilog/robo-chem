"""
Grasp Analyzer

Analyzes pointclouds to determine optimal grasp poses.
Uses geometric analysis for grasp planning.
"""

from typing import Dict, Tuple, Optional
import numpy as np


class GraspAnalyzer:
    """
    Analyzes pointclouds to determine optimal grasp poses.
    
    Supports multiple grasp types:
    - Top grasp: Approach from above
    - Side grasp: Approach from side (for tall objects)
    - Handle grasp: Grasp by handle (for tools)
    """
    
    def __init__(self, gripper_width: float = 0.08):
        """
        Initialize grasp analyzer.
        
        Args:
            gripper_width: Maximum gripper opening width in meters
        """
        self.gripper_width = gripper_width
        
        # Object type to default grasp mapping
        self.grasp_preferences = {
            # Tools (grasp by handle)
            "spoon": "handle",
            "scoop": "handle", 
            "spatula": "handle",
            "pipette": "handle",
            "stirrer": "handle",
            "tongs": "handle",
            
            # Containers (usually side or top)
            "bottle": "side",
            "cup": "side",
            "beaker": "side",
            "flask": "side",
            "bowl": "side",
            "container": "side",
            
            # Small objects (top grasp)
            "powder": "top",
            "pill": "top",
            "sample": "top",
        }
    
    def compute_optimal_grasp(
        self, 
        object_points: np.ndarray,
        object_type: str = "unknown",
        grasp_preference: str = "auto"
    ) -> Optional[np.ndarray]:
        """
        Compute optimal 4x4 grasp pose for object.
        
        Args:
            object_points: Nx3 pointcloud of object
            object_type: Semantic type (cup, bottle, spoon, etc.)
            grasp_preference: "top", "side", "handle", or "auto"
            
        Returns:
            4x4 grasp pose matrix in world frame
        """
        if len(object_points) < 10:
            print("[GraspAnalyzer] Warning: Insufficient points for grasp analysis")
            return None
        
        centroid = np.mean(object_points, axis=0)
        dimensions = self.get_principal_dimensions(object_points)
        
        # Determine grasp type
        if grasp_preference == "auto":
            grasp_preference = self._determine_grasp_type(object_type, dimensions)
        
        print(f"[GraspAnalyzer] Computing {grasp_preference} grasp for {object_type}")
        
        if grasp_preference == "top":
            return self._compute_top_grasp(centroid, dimensions, object_points)
        elif grasp_preference == "side":
            return self._compute_side_grasp(centroid, dimensions, object_points)
        elif grasp_preference == "handle":
            return self._compute_handle_grasp(object_points, centroid)
        else:
            # Default to top
            return self._compute_top_grasp(centroid, dimensions, object_points)
    
    def get_principal_dimensions(self, points: np.ndarray) -> np.ndarray:
        """
        Get object dimensions along principal axes using PCA.
        
        Args:
            points: Nx3 pointcloud
            
        Returns:
            Array of [length, width, height] sorted descending
        """
        centered = points - np.mean(points, axis=0)
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Sort by eigenvalue (descending)
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Rotate to principal axes
        rotated = centered @ eigenvectors
        dimensions = rotated.max(axis=0) - rotated.min(axis=0)
        
        return np.sort(dimensions)[::-1]
    
    def _determine_grasp_type(self, object_type: str, dimensions: np.ndarray) -> str:
        """
        Determine best grasp type based on object type and shape.
        
        Args:
            object_type: Semantic object type
            dimensions: [length, width, height]
            
        Returns:
            Grasp type string
        """
        # Check known object types
        object_type_lower = object_type.lower()
        for key, grasp in self.grasp_preferences.items():
            if key in object_type_lower:
                return grasp
        
        # Use shape heuristics
        if len(dimensions) >= 3:
            aspect_ratio = dimensions[0] / dimensions[2] if dimensions[2] > 0 else 1
            
            # Elongated objects might be tools
            if aspect_ratio > 4:
                return "handle"
            # Tall objects
            elif aspect_ratio > 2:
                return "side"
        
        # Default to top grasp
        return "top"
    
    def _compute_top_grasp(
        self, 
        centroid: np.ndarray, 
        dimensions: np.ndarray,
        points: np.ndarray
    ) -> np.ndarray:
        """
        Compute top-down grasp pose.
        
        The gripper approaches from directly above and grasps.
        
        Args:
            centroid: Object centroid
            dimensions: Object dimensions [l, w, h]
            points: Object pointcloud
            
        Returns:
            4x4 grasp pose matrix
        """
        pose = np.eye(4)
        
        # Position at centroid, offset to top
        pose[:3, 3] = centroid.copy()
        
        if len(dimensions) >= 3:
            # Grasp at appropriate height (middle of object)
            # Offset slightly down from top surface
            top_z = points[:, 2].max()
            pose[2, 3] = top_z - dimensions[2] * 0.3
        
        # Gripper pointing down (rotate 180° around X-axis)
        pose[:3, :3] = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ])
        
        # Check if object fits in gripper
        grasp_width = dimensions[1] if len(dimensions) >= 2 else 0.05
        if grasp_width > self.gripper_width:
            print(f"[GraspAnalyzer] Warning: Object width ({grasp_width:.3f}m) exceeds gripper ({self.gripper_width}m)")
        
        return pose
    
    def _compute_side_grasp(
        self, 
        centroid: np.ndarray, 
        dimensions: np.ndarray,
        points: np.ndarray
    ) -> np.ndarray:
        """
        Compute side grasp pose (for tall objects like bottles).
        
        The gripper approaches from the side horizontally.
        
        Args:
            centroid: Object centroid
            dimensions: Object dimensions
            points: Object pointcloud
            
        Returns:
            4x4 grasp pose matrix
        """
        pose = np.eye(4)
        
        # Position at centroid
        pose[:3, 3] = centroid.copy()
        
        # Gripper approaching from +X direction
        # Orientation: gripper fingers along Y-axis
        pose[:3, :3] = np.array([
            [0, 0, 1],   # X points in Z direction (forward)
            [0, 1, 0],   # Y stays Y (gripper opening direction)
            [-1, 0, 0]   # Z points in -X direction (up from gripper's perspective)
        ])
        
        return pose
    
    def _compute_handle_grasp(
        self, 
        points: np.ndarray, 
        centroid: np.ndarray
    ) -> np.ndarray:
        """
        Compute grasp pose by handle (for tools).
        
        Uses PCA to find the principal axis (handle direction) and
        positions grasp near one end.
        
        Args:
            points: Object pointcloud
            centroid: Object centroid
            
        Returns:
            4x4 grasp pose matrix
        """
        # Find principal axis using PCA
        centered = points - centroid
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Principal axis is eigenvector with largest eigenvalue
        idx = eigenvalues.argsort()[::-1]
        eigenvectors = eigenvectors[:, idx]
        handle_axis = eigenvectors[:, 0]  # Longest axis
        
        # Ensure axis points consistently (e.g., toward positive X)
        if handle_axis[0] < 0:
            handle_axis = -handle_axis
        
        # Find handle end (typically the end closer to robot base)
        projections = centered @ handle_axis
        
        # Grasp point: 70% along the handle from the tool end
        # (tool end is usually further from robot)
        grasp_offset = 0.3 * (projections.max() - projections.min())
        grasp_point = centroid + handle_axis * (projections.min() + grasp_offset)
        
        pose = np.eye(4)
        pose[:3, 3] = grasp_point
        
        # Align gripper with handle axis
        # Z-axis (gripper approach) perpendicular to handle
        z_axis = np.array([0, 0, -1])  # Default: approach from above
        
        # Y-axis (gripper opening) along handle
        y_axis = handle_axis
        
        # X-axis perpendicular to both
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        # Recompute Z to ensure orthogonality
        z_axis = np.cross(x_axis, y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis)
        
        pose[:3, 0] = x_axis
        pose[:3, 1] = y_axis
        pose[:3, 2] = z_axis
        
        return pose
    
    def refine_grasp_pose(
        self, 
        initial_pose: np.ndarray,
        object_points: np.ndarray,
        collision_points: np.ndarray = None
    ) -> np.ndarray:
        """
        Refine grasp pose to avoid collisions and improve stability.
        
        Args:
            initial_pose: Initial grasp pose estimate
            object_points: Target object pointcloud
            collision_points: Points to avoid (e.g., table, other objects)
            
        Returns:
            Refined 4x4 grasp pose
        """
        # This could be extended with:
        # - Collision checking
        # - Grasp quality metrics
        # - Force closure analysis
        
        # For now, return initial pose
        return initial_pose
