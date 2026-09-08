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


_LIVE_INTRINSICS_CACHE = {}


def get_live_intrinsics(cam):
    """
    Colour-stream intrinsics read from the RealSense device itself.

    The .intr files shipped with the cage are mutually inconsistent: camera 2
    claimed fy was roughly half fx, which is impossible for a square-pixel
    sensor and skews the depth reconstruction projectively, so no rigid
    camera-to-world transform can fit it. The factory intrinsics stored on the
    devices agree to well under 1% across all four cameras, so trust those.

    Depth is aligned to the colour stream upstream, so the colour intrinsics are
    the correct ones for deprojecting the depth image.

    Falls back to the file intrinsics if the device cannot be queried.
    """
    cam_id = cam.get_cam_number()
    if cam_id in _LIVE_INTRINSICS_CACHE:
        return _LIVE_INTRINSICS_CACHE[cam_id]

    try:
        import pyrealsense2 as rs
        from perception import CameraIntrinsics

        profile = cam.pipeline.get_active_profile()
        rs_intr = (profile.get_stream(rs.stream.color)
                   .as_video_stream_profile()
                   .get_intrinsics())

        intr = CameraIntrinsics(
            frame=str(cam_id),
            fx=rs_intr.fx, fy=rs_intr.fy,
            cx=rs_intr.ppx, cy=rs_intr.ppy,
            height=rs_intr.height, width=rs_intr.width,
        )
    except Exception as e:
        print(f"[ObjectLocalizer] cam {cam_id}: device intrinsics unavailable "
              f"({e}); using file intrinsics")
        intr = cam.get_cam_intrinsics()

    _LIVE_INTRINSICS_CACHE[cam_id] = intr
    return intr


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
        data = self.capture_pointclouds()
        return [data["images"][c] for c in self.camera_ids if c in data["images"]]
    
    def _open_camera(self, cam_id: int):
        """Return (camera, owned). owned means this call started the pipeline."""
        if cam_id in self.cameras:
            return self.cameras[cam_id], False
        import robomail.vision as vis
        print(f"[ObjectLocalizer] opening cam {cam_id}...")
        return vis.CameraClass(cam_number=cam_id), True

    @staticmethod
    def _close_camera(cam):
        try:
            cam.stop_pipeline()
        except Exception:
            pass

    def capture_pointclouds(self, warmup: int = 8) -> Dict:
        """
        Capture colour and depth from each camera, one at a time.

        Opening all four RealSense pipelines together has repeatedly wedged
        this machine's USB controller. A static scene does not need a
        synchronised snapshot, so each camera is started, warmed up, read,
        and stopped before the next one starts. Persistent CameraClass
        instances (if the caller already opened them) are reused as-is.
        
        Returns:
            Dict with 'pointclouds', 'images', 'depth_images', 'intrinsics'
        """
        pointclouds = {}
        images = {}
        depth_images = {}
        intrinsics = {}
        
        for cam_id in self.camera_ids:
            cam, owned = self._open_camera(cam_id)
            try:
                n = warmup if owned else 1
                img, depth = None, None
                for _ in range(n):
                    img, depth, _, _ = cam.get_next_frame(
                        get_point_cloud=False, get_verts=False
                    )
                images[cam_id] = img.copy()
                depth_images[cam_id] = depth.copy()
                pointclouds[cam_id] = None
                intrinsics[cam_id] = get_live_intrinsics(cam)
            except Exception as e:
                print(f"[ObjectLocalizer] cam {cam_id} capture failed: "
                      f"{type(e).__name__}: {e}")
            finally:
                if owned:
                    self._close_camera(cam)
        
        return {
            'pointclouds': pointclouds,
            'images': images,
            'depth_images': depth_images,
            'intrinsics': intrinsics
        }

    def _extrinsics(self, cam_id: int):
        if cam_id in self.cameras:
            return self.cameras[cam_id].get_cam_extrinsics()
        from robomail.vision.cam_utils import get_cam_info
        return get_cam_info(cam_id)[1]
    
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
                    intrinsics[cam_id] = get_live_intrinsics(self.cameras[cam_id])
        
        by_camera = self.get_object_points_by_camera(masks, depth_images, intrinsics)
        
        if not by_camera:
            return None
        
        # Combine points from all cameras
        all_points = np.vstack(list(by_camera.values()))
        
        # Remove outliers
        cleaned = self._remove_outliers(all_points)
        
        return cleaned
    
    def get_object_points_by_camera(
        self,
        masks: Dict[int, np.ndarray],
        depth_images: Dict[int, np.ndarray] = None,
        intrinsics: Dict[int, object] = None,
    ) -> Dict[int, np.ndarray]:
        """
        Reconstruct the object separately for each camera.

        Keeping the views separate lets callers check whether the cameras
        actually agree before fusing them. Fusing first hides the case where
        one view is segmenting something entirely different.

        Args:
            masks: Camera ID -> segmentation mask (HxW boolean)
            depth_images: Camera ID -> depth image (captured fresh if omitted)
            intrinsics: Camera ID -> intrinsics (read from cameras if omitted)

        Returns:
            Camera ID -> Nx3 array of world-frame points
        """
        if depth_images is None:
            data = self.capture_pointclouds()
            depth_images = data['depth_images']
            intrinsics = data['intrinsics']
        elif intrinsics is None:
            intrinsics = {
                cam_id: get_live_intrinsics(self.cameras[cam_id])
                for cam_id in masks.keys()
                if cam_id in self.cameras
            }
        
        by_camera = {}
        
        for cam_id, mask in masks.items():
            if cam_id not in depth_images:
                continue
            
            intr = intrinsics.get(cam_id)
            if intr is None:
                continue
            
            # Project masked depth to 3D
            points_cam = self._depth_to_points(depth_images[cam_id], mask, intr)
            
            if len(points_cam) == 0:
                continue
            
            # Transform to world frame
            by_camera[cam_id] = self._transform_points(
                points_cam, self._extrinsics(cam_id)
            )
        
        return by_camera
    
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
        
        # perception.CameraIntrinsics exposes cx/cy; native pyrealsense2
        # intrinsics expose ppx/ppy. Guessing here silently produces
        # plausible-looking but wrong 3D points, so fail loudly instead.
        fx = getattr(intrinsics, "fx", None)
        fy = getattr(intrinsics, "fy", None)
        cx = getattr(intrinsics, "cx", getattr(intrinsics, "ppx", None))
        cy = getattr(intrinsics, "cy", getattr(intrinsics, "ppy", None))
        
        if None in (fx, fy, cx, cy):
            raise ValueError(
                f"Could not read fx/fy/cx/cy from intrinsics of type "
                f"{type(intrinsics).__name__}"
            )
        
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
