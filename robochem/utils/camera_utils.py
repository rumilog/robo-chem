"""
Camera Utilities

Helper functions for camera management and calibration.
"""

from typing import Dict, List, Optional
import numpy as np


class CameraManager:
    """
    Manages multiple cameras for the robotic chemistry system.
    
    Wraps robomail.vision.CameraClass instances and provides
    convenient access patterns.
    """
    
    def __init__(self, camera_ids: List[int] = None):
        """
        Initialize camera manager.
        
        Args:
            camera_ids: List of camera IDs to manage (default: [2, 3, 4, 5])
        """
        self.camera_ids = camera_ids or [2, 3, 4, 5]
        self.cameras: Dict = {}
        self._initialized = False
    
    def initialize(self) -> bool:
        """
        Initialize all cameras.
        
        Returns:
            True if all cameras initialized successfully
        """
        try:
            import robomail.vision as vis
            
            for cam_id in self.camera_ids:
                self.cameras[cam_id] = vis.CameraClass(cam_number=cam_id)
                print(f"[CameraManager] Initialized camera {cam_id}")
            
            self._initialized = True
            return True
            
        except ImportError:
            print("[CameraManager] Error: robomail not installed")
            return False
        except Exception as e:
            print(f"[CameraManager] Error initializing cameras: {e}")
            return False
    
    def capture_all(self) -> Dict:
        """
        Capture frames from all cameras.
        
        Returns:
            Dict with camera_id -> {image, depth, pointcloud, intrinsics}
        """
        if not self._initialized:
            self.initialize()
        
        frames = {}
        for cam_id, cam in self.cameras.items():
            try:
                img, depth, pc, verts, intr = cam.get_next_frame()
                frames[cam_id] = {
                    'image': img,
                    'depth': depth,
                    'pointcloud': pc,
                    'vertices': verts,
                    'intrinsics': intr
                }
            except Exception as e:
                print(f"[CameraManager] Error capturing from camera {cam_id}: {e}")
        
        return frames
    
    def get_camera(self, cam_id: int):
        """Get a specific camera instance."""
        if not self._initialized:
            self.initialize()
        return self.cameras.get(cam_id)
    
    def get_cameras_dict(self) -> Dict:
        """Get dictionary of all camera instances."""
        if not self._initialized:
            self.initialize()
        return self.cameras
    
    def close(self):
        """Release all camera resources."""
        for cam in self.cameras.values():
            try:
                if hasattr(cam, 'close'):
                    cam.close()
            except:
                pass
        self.cameras.clear()
        self._initialized = False
