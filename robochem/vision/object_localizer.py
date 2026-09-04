"""
Object Localizer

Localizes objects in 3D using multi-camera pointcloud fusion.
Leverages the existing robomail infrastructure.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    import open3d as o3d
except ImportError:
    o3d = None


class ObjectLocalizer:
    """
    Localizes objects in 3D using multi-camera pointcloud fusion.
    
    This class integrates with the existing robomail vision infrastructure
    to provide object localization for the skills library.
    """
    
    def __init__(self, cameras: Dict = None, camera_ids: List[int] = None):
        """
        Initialize object localizer.
        
        Args:
            cameras: Dictionary mapping camera IDs to CameraClass instances
            camera_ids: List of camera IDs to use (defaults to [2, 3, 4, 5])
        """
        self.cameras = cameras or {}
        self.camera_ids = camera_ids or [2, 3, 4, 5]
        
        # Try to import robomail Vision3D
        self.vision_3d = None
        try:
            import robomail.vision as vis
            self.vision_3d = vis.Vision3D()
            print("[ObjectLocalizer] Initialized with robomail Vision3D")
        except ImportError:
            print("[ObjectLocalizer] Warning: robomail not available")
        
        # Cache for scene analyzer segmentation results
        self._cached_segmentations = {}
        self._cached_pointclouds = {}
        self._cached_centroids = {}
    
    def capture_images(self) -> List[np.ndarray]:
        """
        Capture color images from all cameras.
        
        Returns:
            List of color images (HxWx3 numpy arrays)
        """
        images = []
        for cam_id in self.camera_ids:
            if cam_id in self.cameras:
                img, _, _, _, _ = self.cameras[cam_id].get_next_frame()
                images.append(img)
        return images
    
    def capture_pointclouds(self) -> Dict:
        """
        Capture pointclouds and images from all cameras.
        
        Returns:
            Dict with 'pointclouds', 'images', 'depth_images', 'intrinsics'
        """
        pointclouds = {}
        images = {}
        depth_images = {}
        intrinsics = {}
        
        for cam_id in self.camera_ids:
            if cam_id in self.cameras:
                cam = self.cameras[cam_id]
                img, depth, pc, verts, intr = cam.get_next_frame()
                
                pointclouds[cam_id] = pc
                images[cam_id] = img
                depth_images[cam_id] = depth
                intrinsics[cam_id] = intr
        
        return {
            'pointclouds': pointclouds,
            'images': images,
            'depth_images': depth_images,
            'intrinsics': intrinsics
        }
    
    def get_fused_pointcloud(self) -> Optional['o3d.geometry.PointCloud']:
        """
        Get fused pointcloud from all cameras in world frame.
        
        Uses robomail's multi-camera fusion.
        
        Returns:
            Open3D PointCloud object or None
        """
        if self.vision_3d is None or o3d is None:
            return None
        
        data = self.capture_pointclouds()
        pcs = data['pointclouds']
        
        if len(pcs) < 4:
            print("[ObjectLocalizer] Warning: Need 4 cameras for fusion")
            return None
        
        try:
            # Use robomail's fusion method
            # Assumes cameras 2, 3, 4, 5
            fused_points, center = self.vision_3d.unnormalize_fuse_point_clouds_no_base(
                pcs.get(2), pcs.get(3), pcs.get(4), pcs.get(5)
            )
            
            # Create Open3D pointcloud
            pointcloud = o3d.geometry.PointCloud()
            pointcloud.points = o3d.utility.Vector3dVector(fused_points)
            
            return pointcloud
            
        except Exception as e:
            print(f"[ObjectLocalizer] Fusion error: {e}")
            return None
    
    def get_object_pointcloud(
        self, 
        masks: Dict[int, np.ndarray],
        depth_images: Dict[int, np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """
        Extract object pointcloud using segmentation masks.
        
        Args:
            masks: Camera ID -> segmentation mask (HxW boolean)
            depth_images: Camera ID -> depth image (if not provided, captures fresh)
            
        Returns:
            Nx3 array of object points in world coordinates
        """
        if depth_images is None:
            data = self.capture_pointclouds()
            depth_images = data['depth_images']
            intrinsics = data['intrinsics']
        else:
            intrinsics = {}
            for cam_id in masks.keys():
                if cam_id in self.cameras:
                    intrinsics[cam_id] = self.cameras[cam_id].get_intrinsics()
        
        object_points = []
        
        for cam_id, mask in masks.items():
            if cam_id not in depth_images or cam_id not in self.cameras:
                continue
            
            depth = depth_images[cam_id]
            intr = intrinsics.get(cam_id)
            
            if intr is None:
                continue
            
            # Project masked depth to 3D
            points_cam = self._depth_to_points(depth, mask, intr)
            
            if len(points_cam) == 0:
                continue
            
            # Transform to world frame
            extrinsics = self.cameras[cam_id].get_cam_extrinsics()
            points_world = self._transform_points(points_cam, extrinsics)
            
            object_points.append(points_world)
        
        if len(object_points) == 0:
            return None
        
        # Combine points from all cameras
        all_points = np.vstack(object_points)
        
        # Remove outliers
        cleaned = self._remove_outliers(all_points)
        
        return cleaned
    
    def _depth_to_points(
        self, 
        depth: np.ndarray, 
        mask: np.ndarray,
        intrinsics
    ) -> np.ndarray:
        """
        Convert depth image to 3D points using camera intrinsics.
        
        Args:
            depth: Depth image (HxW)
            mask: Boolean mask (HxW)
            intrinsics: Camera intrinsics object
            
        Returns:
            Nx3 array of 3D points in camera frame
        """
        # Get masked depth values
        v, u = np.where(mask)
        z = depth[v, u] / 1000.0  # Convert mm to meters
        
        # Filter invalid depth
        valid = (z > 0.1) & (z < 2.0)
        v, u, z = v[valid], u[valid], z[valid]
        
        if len(z) == 0:
            return np.array([]).reshape(0, 3)
        
        # Get intrinsic parameters
        try:
            fx = intrinsics.fx
            fy = intrinsics.fy
            cx = intrinsics.ppx
            cy = intrinsics.ppy
        except:
            # Fallback to typical RealSense values
            fx = fy = 600.0
            cx, cy = depth.shape[1] / 2, depth.shape[0] / 2
        
        # Project to 3D
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        
        points = np.stack([x, y, z], axis=-1)
        return points
    
    def _transform_points(
        self, 
        points: np.ndarray, 
        transform
    ) -> np.ndarray:
        """
        Transform points from camera frame to world frame.
        
        Args:
            points: Nx3 array of points
            transform: 4x4 transformation matrix
            
        Returns:
            Nx3 array of transformed points
        """
        if isinstance(transform, np.ndarray):
            T = transform
        else:
            T = transform.matrix() if hasattr(transform, 'matrix') else np.eye(4)
        
        # Convert to homogeneous coordinates
        ones = np.ones((len(points), 1))
        points_h = np.hstack([points, ones])
        
        # Apply transform
        points_transformed = (T @ points_h.T).T
        
        return points_transformed[:, :3]
    
    def _remove_outliers(self, points: np.ndarray, neighbors: int = 20, std_ratio: float = 2.0) -> np.ndarray:
        """
        Remove statistical outliers from pointcloud.
        """
        if len(points) < neighbors:
            return points
        
        from sklearn.neighbors import NearestNeighbors
        
        nbrs = NearestNeighbors(n_neighbors=neighbors).fit(points)
        distances, _ = nbrs.kneighbors(points)
        
        mean_dist = np.mean(distances, axis=1)
        threshold = np.mean(mean_dist) + std_ratio * np.std(mean_dist)
        
        return points[mean_dist < threshold]
    
    def localize_object(self, object_name: str) -> Optional[np.ndarray]:
        """
        Get 4x4 pose matrix of object.
        
        Note: Currently returns pose at centroid with default orientation.
        Could be extended to estimate object orientation from pointcloud.
        
        Args:
            object_name: Name of object to localize
            
        Returns:
            4x4 pose matrix or None if not found
        """
        centroid = self.get_object_centroid(object_name)
        if centroid is None:
            return None
        
        # Create pose matrix (default orientation: Z-down)
        pose = np.eye(4)
        pose[:3, 3] = centroid
        pose[:3, :3] = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ])
        
        return pose
    
    def get_object_centroid(self, object_name: str) -> Optional[np.ndarray]:
        """
        Get 3D centroid of object.
        
        This method should be called after segment_object has cached
        the segmentation, or it will segment the object fresh.
        
        Args:
            object_name: Name of object
            
        Returns:
            3D point (x, y, z) or None
        """
        # Check cache
        if object_name in self._cached_centroids:
            return self._cached_centroids[object_name]
        
        # Need to segment and localize
        # This requires scene_analyzer to be set
        print(f"[ObjectLocalizer] Object '{object_name}' not in cache, need fresh segmentation")
        return None
    
    def get_object_dimensions(self, object_name: str) -> Optional[np.ndarray]:
        """
        Get bounding box dimensions of object.
        
        Args:
            object_name: Name of object
            
        Returns:
            Array of [length, width, height] in meters or None
        """
        # Check cache for pointcloud
        if object_name in self._cached_pointclouds:
            points = self._cached_pointclouds[object_name]
            return self._compute_dimensions(points)
        
        return None
    
    def _compute_dimensions(self, points: np.ndarray) -> np.ndarray:
        """
        Compute bounding box dimensions using PCA.
        
        Args:
            points: Nx3 array of points
            
        Returns:
            Array of [length, width, height] sorted descending
        """
        centered = points - np.mean(points, axis=0)
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        
        # Rotate to principal axes
        rotated = centered @ eigenvectors
        dimensions = rotated.max(axis=0) - rotated.min(axis=0)
        
        return np.sort(dimensions)[::-1]
    
    def cache_object(
        self, 
        object_name: str, 
        pointcloud: np.ndarray,
        centroid: np.ndarray = None
    ):
        """
        Cache object pointcloud and centroid for fast lookup.
        
        Args:
            object_name: Object identifier
            pointcloud: Nx3 array of points
            centroid: Optional precomputed centroid
        """
        self._cached_pointclouds[object_name] = pointcloud
        self._cached_centroids[object_name] = centroid if centroid is not None else np.mean(pointcloud, axis=0)
    
    def clear_cache(self):
        """Clear all cached data."""
        self._cached_segmentations.clear()
        self._cached_pointclouds.clear()
        self._cached_centroids.clear()
