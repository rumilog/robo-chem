"""
RoboChem Vision Module

Provides visual perception capabilities:
- Scene analysis using SAM3 for segmentation
- Object localization using multi-camera pointcloud fusion
- Grasp pose computation
- Instruction parsing from images
"""

import numpy as np

from .scene_analyzer import SceneAnalyzer
from .object_localizer import ObjectLocalizer
from .grasp_analyzer import GraspAnalyzer
from .instruction_parser import InstructionParser
from .label_resolver import LabelResolver, looks_like_label_query


class VisionSystem:
    """
    Unified vision system that combines all visual perception capabilities.
    """
    
    def __init__(
        self,
        cameras: dict = None,
        sam_checkpoint: str = None,
        sam_model_type: str = "vit_h",
        grounding_url: str = None,
        consensus_tolerance: float = 0.05,
        max_object_extent: float = 0.35,
        min_object_height: float = -0.01,
        min_object_points: int = 50,
        labeled_cup_category: str = "white paper cup",
    ):
        """
        Initialize the vision system.
        
        Args:
            cameras: Dictionary mapping camera IDs to CameraClass instances
            sam_checkpoint: Path to SAM model checkpoint
            sam_model_type: SAM model type ("vit_h", "vit_l", "vit_b")
            grounding_url: URL of the open-vocabulary grounding service
            consensus_tolerance: Max distance (m) a camera's centroid may sit
                from the median before that camera is discarded
            max_object_extent: Largest plausible object dimension (m)
            min_object_height: Lowest plausible centroid height (m); guards
                against reconstructions that land below the table
            min_object_points: Minimum surviving points for a valid detection
            labeled_cup_category: SAM prompt used when resolving reagent labels
                on paper under scoop-target cups
        """
        self.cameras = cameras or {}
        self.consensus_tolerance = consensus_tolerance
        self.max_object_extent = max_object_extent
        self.min_object_height = min_object_height
        self.min_object_points = min_object_points
        self.labeled_cup_category = labeled_cup_category
        
        # Initialize components
        self.scene_analyzer = SceneAnalyzer(
            sam_checkpoint=sam_checkpoint,
            model_type=sam_model_type,
            grounding_url=grounding_url,
        ) if (sam_checkpoint or grounding_url) else None
        
        # Always construct the localizer, even with an empty cameras dict.
        # Sequential capture opens each RealSense for one frame and closes it,
        # so we deliberately do not hold four pipelines open.
        self.object_localizer = ObjectLocalizer(
            cameras=self.cameras,
            camera_ids=list(self.cameras.keys()) if self.cameras else [2, 3, 4, 5],
        )
        
        self.grasp_analyzer = GraspAnalyzer()
        self.instruction_parser = InstructionParser()
        self.label_resolver = LabelResolver(
            vlm_client=getattr(self.scene_analyzer, "vlm_client", None)
            if self.scene_analyzer else None
        )
    def capture_scene(self):
        """Capture images from all cameras."""
        if self.object_localizer:
            return self.object_localizer.capture_images()
        return []
    
    def segment_object(self, images, object_query):
        """Segment an object across multiple camera views."""
        if self.scene_analyzer:
            return self.scene_analyzer.segment_object(images, object_query)
        return None, {}
    
    def get_object_pointcloud(self, masks):
        """Get 3D pointcloud of segmented object."""
        if self.object_localizer:
            return self.object_localizer.get_object_pointcloud(masks)
        return None
    
    def locate(self, object_name, force_refresh: bool = False,
               category: str = None, use_labels: bool = None):
        """
        Segment an object, reconstruct its pointcloud and cache the result.

        This is the bridge between the scene analyzer and the localizer: the
        localizer only reads from its cache, so without this step every
        get_object_centroid call returns None.

        For scoop targets (reagent names on paper under white cups), pass the
        label as ``object_name`` (e.g. ``"citric acid"``). That routes through
        multi-instance SAM + VLM label reading unless ``use_labels=False``.

        Args:
            object_name: Semantic name of the object to find
            force_refresh: Re-segment even if the object is already cached
            category: SAM category when resolving labels (default: white paper cup)
            use_labels: Force / disable the labelled-cup path. Default None =
                auto (label path for reagent-like names)

        Returns:
            Dict with points, centroid, dimensions and confidences, or None
        """
        if self.object_localizer is None or self.scene_analyzer is None:
            print("[VisionSystem] Cannot locate: needs both cameras and a SAM checkpoint")
            return None
        
        if not force_refresh and object_name in self.object_localizer._cached_centroids:
            points = self.object_localizer._cached_pointclouds.get(object_name)
            return {
                "points": points,
                "centroid": self.object_localizer._cached_centroids[object_name],
                "dimensions": self.compute_dimensions(points) if points is not None else None,
                "cached": True,
            }

        want_labels = (
            use_labels if use_labels is not None
            else looks_like_label_query(object_name)
        )
        if want_labels:
            located = self.locate_labeled(
                object_name,
                category=category or self.labeled_cup_category,
            )
            if located is not None:
                return located
            print(f"[VisionSystem] Label match failed for {object_name!r}; "
                  f"falling back to direct SAM prompt")

        return self._locate_direct(object_name)

    def locate_labeled(self, label: str, category: str = None):
        """
        Find the cup whose paper label matches ``label``.

        Segments every instance of ``category`` (default white paper cup),
        reads labels with the VLM, then fuses the matching cup in 3D.
        """
        category = category or self.labeled_cup_category
        if self.object_localizer is None or self.scene_analyzer is None:
            return None
        if self.scene_analyzer.grounding is None:
            print("[VisionSystem] locate_labeled needs the grounding service")
            return None

        data = self.object_localizer.capture_pointclouds()
        cam_ids = [c for c in self.object_localizer.camera_ids if c in data["images"]]
        images = {c: data["images"][c] for c in cam_ids}
        if not images:
            print("[VisionSystem] No camera images available")
            return None

        print(f"[VisionSystem] Resolving labelled cup {label!r} "
              f"via category {category!r}")
        instances = self.scene_analyzer.grounding.segment_instances(images, category)
        if not instances:
            print(f"[VisionSystem] No {category!r} instances found")
            return None

        masks, confidences, observed = self.label_resolver.resolve(
            images, instances, label
        )
        if not masks:
            print(f"[VisionSystem] No cup labelled {label!r} in any camera")
            return None

        return self._fuse_masks(
            label, masks, confidences, data,
            extra={"label_matches": observed, "category": category},
        )

    def _locate_direct(self, object_name: str):
        data = self.object_localizer.capture_pointclouds()
        cam_ids = [c for c in self.object_localizer.camera_ids if c in data["images"]]
        images = [data["images"][c] for c in cam_ids]
        
        if not images:
            print("[VisionSystem] No camera images available")
            return None
        
        masks, confidences = self.scene_analyzer.segment_object(
            images, object_name, cameras=cam_ids
        )
        if not masks:
            print(f"[VisionSystem] '{object_name}' not detected in any of cameras {cam_ids}")
            return None

        return self._fuse_masks(object_name, masks, confidences, data)

    def _fuse_masks(self, object_name, masks, confidences, data, extra=None):
        by_camera = self.object_localizer.get_object_points_by_camera(
            masks, depth_images=data["depth_images"], intrinsics=data["intrinsics"]
        )
        if not by_camera:
            print(f"[VisionSystem] No 3D points reconstructed for '{object_name}'")
            return None
        
        agreed, rejected = self._filter_by_consensus(by_camera)
        if not agreed:
            print(f"[VisionSystem] Cameras could not agree on '{object_name}'; "
                  f"refusing to guess a position")
            return None
        if rejected:
            print(f"[VisionSystem] Discarded cameras {rejected} for '{object_name}' "
                  f"(disagreed with the majority)")
        
        points = self.object_localizer._remove_outliers(
            np.vstack([by_camera[c] for c in agreed])
        )
        
        plausible, reason = self._check_plausible(points)
        if not plausible:
            print(f"[VisionSystem] Rejecting '{object_name}': {reason}")
            return None
        
        self.object_localizer.cache_object(object_name, points)
        
        result = {
            "points": points,
            "centroid": self.object_localizer._cached_centroids[object_name],
            "dimensions": self.compute_dimensions(points),
            "confidences": {c: confidences.get(c) for c in agreed},
            "cameras": agreed,
            "rejected_cameras": rejected,
            "cached": False,
        }
        if extra:
            result.update(extra)
        return result
    
    def _filter_by_consensus(self, by_camera: dict):
        """
        Drop cameras whose reconstructed centroid disagrees with the majority.

        A camera that segmented the wrong thing produces a centroid far from
        the others. Fusing it in silently corrupts the result, so it is
        excluded instead.

        Returns:
            (accepted_camera_ids, rejected_camera_ids)
        """
        cam_ids = sorted(by_camera.keys())
        if len(cam_ids) == 1:
            return cam_ids, []
        
        centroids = np.array([by_camera[c].mean(axis=0) for c in cam_ids])
        median = np.median(centroids, axis=0)
        distances = np.linalg.norm(centroids - median, axis=1)
        
        accepted = [c for c, d in zip(cam_ids, distances) if d <= self.consensus_tolerance]
        rejected = [c for c, d in zip(cam_ids, distances) if d > self.consensus_tolerance]
        
        for c, d in zip(cam_ids, distances):
            print(f"    cam {c}: centroid={np.round(by_camera[c].mean(axis=0), 4)} "
                  f"dist_from_median={d:.4f}m "
                  f"{'OK' if d <= self.consensus_tolerance else 'REJECT'}")
        
        # With only two views there is no majority to appeal to, so a
        # disagreement means neither can be trusted.
        if len(cam_ids) == 2 and rejected:
            return [], cam_ids
        
        return accepted, rejected
    
    def _check_plausible(self, points: np.ndarray):
        """
        Sanity-check a reconstructed object against physical expectations.

        Returns:
            (is_plausible, reason)
        """
        if len(points) < self.min_object_points:
            return False, f"only {len(points)} points (need {self.min_object_points})"
        
        extent = points.max(axis=0) - points.min(axis=0)
        if extent.max() > self.max_object_extent:
            return False, (f"extent {np.round(extent, 3)} exceeds "
                           f"{self.max_object_extent}m; mask probably includes the table")
        
        centroid = points.mean(axis=0)
        if centroid[2] < self.min_object_height:
            return False, (f"centroid z={centroid[2]:.3f} is below "
                           f"{self.min_object_height}m; reconstruction is not on the table")
        
        return True, "plausible"
    
    def localize_object(self, object_name):
        """Get 4x4 pose matrix of object."""
        if self.object_localizer is None:
            return None
        if self.locate(object_name) is None:
            return None
        return self.object_localizer.localize_object(object_name)
    
    def get_object_centroid(self, object_name):
        """Get 3D centroid of object."""
        located = self.locate(object_name)
        return located["centroid"] if located else None
    
    def get_object_dimensions(self, object_name):
        """Get bounding box dimensions of object."""
        located = self.locate(object_name)
        return located["dimensions"] if located else None
    
    def compute_grasp_pose(self, object_points, object_name=None, grasp_type="auto",
                           pitch_deg: float = 25.0):
        """Compute optimal grasp pose for object."""
        return self.grasp_analyzer.compute_optimal_grasp(
            object_points, 
            object_type=object_name,
            grasp_preference=grasp_type,
            pitch_deg=pitch_deg,
        )

    def compute_grasp_candidates(self, object_points, object_name=None,
                                 grasp_type="auto", pitch_deg: float = 0.0,
                                 spout_xy=None):
        """
        Candidate grasp poses to try until one is reachable.

        Default path is a single top-down upright grasp. Pass grasp_type="side"
        for tipped / spout-aligned grasps.
        """
        if grasp_type == "side":
            return self.grasp_analyzer.compute_side_grasp_candidates(
                object_points, pitch_deg=pitch_deg, spout_xy=spout_xy
            )
        gtype = "top" if grasp_type in ("auto", "top") else grasp_type
        pose = self.compute_grasp_pose(
            object_points, object_name, gtype, pitch_deg=pitch_deg
        )
        return [pose] if pose is not None else []
    
    def compute_dimensions(self, points):
        """Compute bounding box dimensions from pointcloud."""
        return self.grasp_analyzer.get_principal_dimensions(points)
    
    def clear_cache(self):
        """
        Drop cached segmentations.

        Must be called after the scene changes (e.g. after a pick or place),
        otherwise skills act on stale object positions.
        """
        if self.object_localizer:
            self.object_localizer.clear_cache()
