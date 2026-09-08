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
            
            # Containers — upright top-down grasp (no pour tip)
            "bottle": "top",
            "cup": "top",
            "beaker": "top",
            "flask": "top",
            "bowl": "top",
            "container": "top",
            
            # Small objects (top grasp)
            "powder": "top",
            "pill": "top",
            "sample": "top",
        }
    
    def compute_optimal_grasp(
        self, 
        object_points: np.ndarray,
        object_type: str = "unknown",
        grasp_preference: str = "auto",
        pitch_deg: float = 25.0,
    ) -> Optional[np.ndarray]:
        """
        Compute optimal 4x4 grasp pose for object.
        
        Args:
            object_points: Nx3 pointcloud of object
            object_type: Semantic type (cup, bottle, spoon, etc.)
            grasp_preference: "top", "side", "handle", or "auto"
            pitch_deg: For side grasps, tip the wrist this many degrees from
                vertical so the cup rim clears the EEF for later pouring.
            
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
            candidates = self._compute_side_grasp_candidates(
                object_points, pitch_deg=pitch_deg
            )
            return candidates[0] if candidates else None
        elif grasp_preference == "handle":
            return self._compute_handle_grasp(object_points, centroid)
        else:
            # Default to top
            return self._compute_top_grasp(centroid, dimensions, object_points)

    def compute_side_grasp_candidates(
        self,
        object_points: np.ndarray,
        pitch_deg: float = 25.0,
        spout_xy: np.ndarray = None,
    ):
        """
        Pour-friendly side grasps at ``pitch_deg`` from vertical.

        Closing axis stays parallel to the table. If ``spout_xy`` is given
        (beaker), the wrist tips toward the spout and the fingers close
        across the axis perpendicular to it so a later pour exits the spout.
        """
        if len(object_points) < 10:
            return []
        return self._compute_side_grasp_candidates(
            object_points, pitch_deg=pitch_deg, spout_xy=spout_xy
        )

    def estimate_spout_direction_xy(self, points: np.ndarray) -> Optional[np.ndarray]:
        """
        Horizontal unit vector from object centre toward the spout.

        Uses the farthest XY point in the upper half of the cloud from the
        median centre — for a beaker that outlier is the spout lip. Returns
        None if the cloud is too round to have a clear protrusion.
        """
        if len(points) < 50:
            return None

        z_lo, z_hi = float(points[:, 2].min()), float(points[:, 2].max())
        # Prefer the rim band where the spout sticks out most.
        rim = points[points[:, 2] > z_lo + 0.45 * (z_hi - z_lo)]
        if len(rim) < 30:
            rim = points

        centre = np.median(rim[:, :2], axis=0)
        deltas = rim[:, :2] - centre
        dists = np.linalg.norm(deltas, axis=1)
        # Robust "radius" of the body vs the spout tip.
        body_r = float(np.percentile(dists, 70))
        tip_r = float(np.percentile(dists, 98))
        if tip_r < body_r * 1.08:
            print(f"[GraspAnalyzer] No clear spout protrusion "
                  f"(body_r={body_r * 1000:.1f}mm tip_r={tip_r * 1000:.1f}mm)")
            return None

        tip = deltas[int(np.argmax(dists))]
        direction = tip / (np.linalg.norm(tip) + 1e-9)
        print(f"[GraspAnalyzer] Spout direction XY ≈ "
              f"{np.round(direction, 3)} "
              f"(protrudes {(tip_r - body_r) * 1000:.1f}mm past body)")
        return direction
    
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
    
    def _compute_side_grasp_candidates(
        self,
        points: np.ndarray,
        pitch_deg: float = 25.0,
        spout_xy: np.ndarray = None,
    ):
        """
        Cup/beaker grasps with finger-closing axis parallel to the table.

        Without a spout: tip ±pitch around the narrow-axis closing direction.
        With a spout: tip toward the spout only, fingers close on the axis
        perpendicular to the spout so pour tilts liquid out the lip.
        """
        z_lo, z_hi = float(points[:, 2].min()), float(points[:, 2].max())
        position = np.array([
            float(np.median(points[:, 0])),
            float(np.median(points[:, 1])),
            0.5 * (z_lo + z_hi),
        ])

        pitch = np.radians(float(pitch_deg))
        z_lift = 0.004 * abs(np.sin(pitch)) / max(abs(np.sin(np.radians(25))), 1e-3)

        if spout_xy is not None:
            spout = np.asarray(spout_xy, dtype=float)[:2]
            spout = spout / (np.linalg.norm(spout) + 1e-9)
            # Fingers close across the spout axis (perpendicular in XY).
            closing0 = np.array([-spout[1], spout[0]])
            candidates = []
            for closing in (closing0, -closing0):
                # Choose signed pitch so tool Z leans toward spout.
                for signed in (pitch, -pitch):
                    pose = self._pose_pitch_around_closing(
                        position, closing, signed, z_lift
                    )
                    tip_h = pose[:2, 2].copy()
                    if np.linalg.norm(tip_h) < 1e-6:
                        continue
                    tip_h = tip_h / np.linalg.norm(tip_h)
                    if float(tip_h @ spout) > 0.3:
                        candidates.append(pose)
            unique = []
            for pose in candidates:
                if any(np.allclose(pose[:3, :3], u[:3, :3], atol=1e-3) for u in unique):
                    continue
                unique.append(pose)
            print(f"[GraspAnalyzer] {len(unique)} spout-aligned grasps "
                  f"(tip toward spout, closing ⊥ spout, ±{pitch_deg:.0f} deg)")
            return unique

        xy = points[:, :2] - np.median(points[:, :2], axis=0)
        if len(xy) >= 10:
            _, _, Vt = np.linalg.svd(xy, full_matrices=False)
            closing0 = Vt[-1]
        else:
            closing0 = np.array([0.0, 1.0])
        closing0 = closing0 / (np.linalg.norm(closing0) + 1e-9)

        candidates = []
        for closing in (closing0, -closing0):
            for signed_pitch in (pitch, -pitch):
                candidates.append(
                    self._pose_pitch_around_closing(
                        position, closing, signed_pitch, z_lift
                    )
                )

        unique = []
        for pose in candidates:
            R = pose[:3, :3]
            if any(np.allclose(R, u[:3, :3], atol=1e-3) for u in unique):
                continue
            unique.append(pose)

        def lean_toward_base_score(pose):
            return float(pose[0, 2])

        unique.sort(key=lean_toward_base_score)
        print(f"[GraspAnalyzer] {len(unique)} cup grasps: closing axis "
              f"parallel to table, wrist tip ±{pitch_deg:.0f} deg "
              f"(xy={np.round(position[:2], 4)}, z={position[2] + z_lift:.3f})")
        return unique

    @staticmethod
    def _pose_pitch_around_closing(position, closing_xy, pitch_rad, z_lift):
        """
        Top-down cup frame pitched around the horizontal closing axis.

        Tool Y (finger open/close) stays in the table plane. Tool Z tips
        from vertical by ``pitch_rad`` around Y — that is the wrist rotation
        that clears the rim for pouring without tilting the jaws.
        """
        # Closing axis: strictly horizontal.
        y_axis = np.array([closing_xy[0], closing_xy[1], 0.0], dtype=float)
        y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-9)

        down = np.array([0.0, 0.0, -1.0])
        # Upright along-cup axis (horizontal, ⊥ closing).
        x0 = np.cross(y_axis, down)
        x0 = x0 / (np.linalg.norm(x0) + 1e-9)

        # Rotate x and z around y by pitch (Rodrigues).
        c, s = np.cos(pitch_rad), np.sin(pitch_rad)
        # z starts as down; after pitch around y: z' = c*down - s*x0
        # (right-hand: positive pitch tips z toward -x0)
        z_axis = c * down - s * x0
        x_axis = c * x0 + s * down
        z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-9)
        x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-9)
        # Re-orthonormalize y against the tipped frame.
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-9)
        # Force closing back to horizontal if float drift introduced a Z.
        y_axis[2] = 0.0
        y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-9)
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-9)
        z_axis = np.cross(x_axis, y_axis)
        z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-9)

        pose = np.eye(4)
        pose[:3, 0] = x_axis
        pose[:3, 1] = y_axis
        pose[:3, 2] = z_axis
        pose[:3, 3] = position
        pose[2, 3] += z_lift
        return pose

    def _compute_side_grasp(
        self, 
        centroid: np.ndarray, 
        dimensions: np.ndarray,
        points: np.ndarray,
        pitch_deg: float = 25.0,
    ) -> np.ndarray:
        """Single preferred cup side-grasp (first candidate)."""
        candidates = self._compute_side_grasp_candidates(points, pitch_deg=pitch_deg)
        return candidates[0] if candidates else np.eye(4)
    
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
