"""
RoboChem Vision Module

Provides visual perception capabilities:
- Scene analysis using SAM3 for segmentation
- Object localization using multi-camera pointcloud fusion
- Grasp pose computation
- Instruction parsing from images
"""

import numpy as np
from itertools import combinations

from .scene_analyzer import SceneAnalyzer
from .object_localizer import ObjectLocalizer
from .grasp_analyzer import GraspAnalyzer
from .instruction_parser import InstructionParser
from .label_resolver import (LabelResolver, looks_like_label_query,
                             resolve_sam_prompt)


def _label_read_cameras(observed) -> list:
    """Cameras whose text was read, not filled in by projecting another view."""
    readers = []
    for cam, text in (observed or {}).items():
        if text and not str(text).startswith("(projected"):
            readers.append(cam)
    return readers


class VisionSystem:
    """
    Unified vision system that combines all visual perception capabilities.
    """

    #: Channel order of the frames :meth:`capture_scene` returns. The RealSense
    #: cage streams bgr8; the simulated cage renders RGB and says so. Anything
    #: that hands a frame to a VLM or writes it to disk has to know which.
    frame_color_order = "bgr"

    def __init__(
        self,
        cameras: dict = None,
        sam_checkpoint: str = None,
        sam_model_type: str = "vit_h",
        grounding_url: str = None,
        consensus_tolerance: float = 0.05,
        max_object_extent: float = 0.35,
        min_object_height: float = -0.02,
        min_object_points: int = 50,
        labeled_cup_category: str = "white bowl",
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
            labeled_cup_category: SAM prompt used when resolving reagent labels.
                The reagents sit in shallow white bowls, so that is the default;
                a labelled container of any other kind needs the skill's
                ``container_category`` instead, because the label path only
                ever considers instances of this one category. Water lives in
                clear cups, for example, and needs
                ``container_category="clear plastic cup"``.
                Chosen by scripts/sweep_prompts.py against
                scene_captures/bowls_20260924_160459: "white bowl" scored 0.97
                on 4/4 cameras, where the old "white paper cup" managed 0.60
                on paper under scoop-target cups
        """
        self.cameras = cameras or {}
        self.consensus_tolerance = consensus_tolerance
        self.max_object_extent = max_object_extent
        self.min_object_height = min_object_height
        self.min_object_points = min_object_points
        self.labeled_cup_category = labeled_cup_category
        # Drop labelled-cup / multi-instance masks below this SAM score before
        # 3D consensus (cam 2 once matched baking soda at 0.39 on the wrong blob).
        self.min_instance_score = 0.5
        #: Cameras that read the label themselves before the position is
        #: propagated to the rest by projection. >1 so a single misread seed
        #: is caught rather than pushed onto every other camera.
        self.projection_seed_cameras = 2
        #: How far apart two seed cameras may put the same container (metres)
        #: before their agreement is not worth trusting.
        self.projection_seed_tolerance = 0.05
        #: A projection landing this close to an instance still counts as that
        #: instance — masks stop at the visible edge, projections do not.
        self.projection_pixel_slack = 40.0
        
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

    def known_object_names(self):
        """
        Names this perception stack is known to resolve, or ``[]`` if open ended.

        Grounding here is open-vocabulary -- SAM 3 is asked for whatever noun
        phrase a skill passes -- so the real cell has no such inventory and
        returns empty. The simulated cell does know exactly what is on its
        bench, and overrides this, which is what lets the planner be told that
        "measuring scoop" is not a thing it can ask for there but "larger
        spoon" is. Callers must treat an empty list as "no constraint", never
        as "nothing is present".
        """
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

        masks, confidences, observed, how = self._resolve_by_projection(
            label, images, instances, data
        )
        if not masks:
            print(f"[VisionSystem] Projection path found nothing; falling back "
                  f"to reading labels on every camera")
            masks, confidences, observed = self.label_resolver.resolve(
                images, instances, label
            )
            how = "per-camera label reads (fallback)"
        if not masks:
            print(f"[VisionSystem] No cup labelled {label!r} in any camera")
            return None

        extra = {"label_matches": observed, "category": category,
                 "resolved_by": how}
        located = self._fuse_masks(label, masks, confidences, data, extra=extra)
        if located is not None:
            return located

        # A second camera matched by projection can sit on a different cup.
        # Two disagreeing views are both rejected, which throws away the one
        # camera that actually read the label. On 2026-09-25 only cam 3 read
        # "B 10 ML WATER"; cam 4 was 52 mm away and the cup was refused.
        # Trust that single read. Two cameras that both read the label and
        # still disagree are a different failure — neither is trusted.
        readers = _label_read_cameras(observed)
        if len(readers) == 1 and readers[0] in masks:
            cam = readers[0]
            print(f"[VisionSystem] Only cam {cam} read {label!r}; "
                  f"using that camera's location")
            only_extra = dict(extra)
            only_extra["resolved_by"] = (
                f"single camera {cam} (only view that read the label)")
            only_extra["label_matches"] = {cam: observed.get(cam)}
            return self._fuse_masks(
                label, {cam: masks[cam]},
                {cam: confidences.get(cam)}, data, extra=only_extra,
            )
        return None

    def _resolve_by_projection(self, label, images, instances, data):
        """
        Identify the container ONCE, then tell the other cameras where it is.

        Reading the label independently in every camera means every camera can
        disagree, and they did: on 2026-09-21 one camera matched "TONIC Water",
        another returned the same label for two different cups, and three of
        four were rejected for disagreeing in 3D. The 3D check can only discard
        a bad identification after the fact — it cannot help a camera identify
        correctly.

        So: read labels on the ``seed_cameras`` most confident views and
        require them to AGREE (the mitigation — one misread seed would
        otherwise be propagated everywhere). Take that container's 3D centroid,
        project it into every remaining camera, and pick the instance whose
        mask contains the projected pixel. Geometry, not another label read.

        A camera whose projection lands in no instance is skipped, loudly. That
        is the honest outcome: either the container is occluded there, or the
        extrinsics are off — both worth knowing, neither worth guessing past.

        Returns (masks, confidences, observed_labels, description).
        """
        seed_n = max(1, int(self.projection_seed_cameras))
        cam_ids = list(images.keys())

        # Seed on the cameras whose best instance scored highest — the ones
        # most likely to have a clean, readable view.
        def best_score(cam):
            return max((float(i.get("score") or 0.0)
                        for i in instances.get(cam, [])), default=0.0)
        ranked = sorted((c for c in cam_ids if instances.get(c)),
                        key=best_score, reverse=True)
        seeds = ranked[:seed_n]
        if not seeds:
            return {}, {}, {}, "no instances"

        seed_masks, seed_conf, seed_obs = self.label_resolver.resolve(
            {c: images[c] for c in seeds},
            {c: instances[c] for c in seeds},
            label,
        )
        if not seed_masks:
            print(f"[VisionSystem] No seed camera could read {label!r}")
            return {}, {}, {}, "seed read failed"

        # Mitigation: when more than one seed read it, make them agree in 3D
        # before trusting either. A seed that is wrong about WHICH cup would
        # otherwise drag every other camera onto the wrong container.
        seed_points = self.object_localizer.get_object_points_by_camera(
            seed_masks, depth_images=data["depth_images"],
            intrinsics=data["intrinsics"],
        )
        seed_points = {c: p for c, p in (seed_points or {}).items() if len(p)}
        if not seed_points:
            print("[VisionSystem] Seed cameras produced no 3D points")
            return {}, {}, {}, "seed reconstruction failed"

        centroids = {c: p.mean(axis=0) for c, p in seed_points.items()}
        if len(centroids) > 1:
            cams = list(centroids)
            spread = max(
                float(np.linalg.norm(centroids[a] - centroids[b]))
                for i, a in enumerate(cams) for b in cams[i + 1:]
            )
            if spread > self.projection_seed_tolerance:
                print(f"[VisionSystem] Seed cameras {cams} disagree by "
                      f"{spread * 1000:.0f}mm on where {label!r} is (tolerance "
                      f"{self.projection_seed_tolerance * 1000:.0f}mm) — not "
                      f"propagating a position the seeds cannot agree on")
                return {}, {}, {}, "seeds disagreed"
            print(f"[VisionSystem] Seed cameras {cams} agree within "
                  f"{spread * 1000:.0f}mm")

        anchor = np.mean(np.stack(list(centroids.values())), axis=0)
        print(f"[VisionSystem] {label!r} anchored at "
              f"{np.round(anchor, 4)} from {sorted(centroids)}; "
              f"projecting into the other cameras")

        masks = dict(seed_masks)
        confidences = dict(seed_conf)
        observed = dict(seed_obs)

        for cam in cam_ids:
            if cam in masks or not instances.get(cam):
                continue
            uv = self.object_localizer.project_world_point(
                anchor, cam, intrinsics=data["intrinsics"].get(cam))
            if uv is None:
                print(f"[VisionSystem] cam {cam}: {label!r} projects behind "
                      f"the camera; skipping")
                continue
            u, v = int(round(uv[0])), int(round(uv[1]))

            hit = None
            for inst in instances[cam]:
                m = inst["mask"]
                if 0 <= v < m.shape[0] and 0 <= u < m.shape[1] and m[v, u]:
                    hit = inst
                    break
            if hit is None:
                # Nearest instance centre, if it is close enough to be the
                # same object rather than a different cup entirely.
                best, best_d = None, None
                for inst in instances[cam]:
                    ys, xs = np.where(inst["mask"])
                    if not len(xs):
                        continue
                    d = float(np.hypot(xs.mean() - u, ys.mean() - v))
                    if best_d is None or d < best_d:
                        best, best_d = inst, d
                if best is not None and best_d <= self.projection_pixel_slack:
                    hit = best
                    print(f"[VisionSystem] cam {cam}: projection at ({u},{v}) "
                          f"fell just outside a mask; using the instance "
                          f"{best_d:.0f}px away")

            if hit is None:
                print(f"[VisionSystem] cam {cam}: {label!r} projects to "
                      f"({u},{v}), which is inside no {'' if instances[cam] else '(no) '}"
                      f"instance — occluded there, or the extrinsics are off")
                continue

            masks[cam] = hit["mask"]
            confidences[cam] = float(hit.get("score") or 0.0)
            observed[cam] = f"(projected from {sorted(centroids)})"
            print(f"[VisionSystem] cam {cam}: matched by projection at "
                  f"({u},{v}), score {confidences[cam]:.2f}")

        return (masks, confidences, observed,
                f"projection from cameras {sorted(centroids)}")

    def _locate_direct(self, object_name: str):
        # The name SAM is prompted with may differ from the one we track the
        # object by; GroundingClient resolves that at the HTTP boundary so
        # every caller gets it, this one included. See SAM_PROMPT_ALIASES.
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

        # Drop weak SAM instances before consensus (wrong blob, tiny mask, etc.).
        score_rejected = []
        if confidences:
            for cam_id in list(by_camera.keys()):
                score = confidences.get(cam_id)
                if score is not None and float(score) < self.min_instance_score:
                    print(f"[VisionSystem] Discarded cam {cam_id} for "
                          f"{object_name!r}: score {float(score):.2f} < "
                          f"{self.min_instance_score}")
                    del by_camera[cam_id]
                    score_rejected.append(cam_id)
        if not by_camera:
            print(f"[VisionSystem] All cameras below score floor for "
                  f"{object_name!r}")
            return None

        agreed, rejected = self._filter_by_consensus(
            by_camera, confidences=confidences
        )
        rejected = list(score_rejected) + list(rejected)
        if not agreed:
            print(f"[VisionSystem] Cameras could not agree on '{object_name}'; "
                  f"refusing to guess a position")
            return None
        if rejected:
            print(f"[VisionSystem] Discarded cameras {rejected} for '{object_name}' "
                  f"(low score or disagreed with the largest cluster)")

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

    def _filter_by_consensus(self, by_camera: dict, confidences: dict = None):
        """
        Keep the largest set of cameras whose centroids all agree pairwise.

        Median-of-all fails when two wrong views and two right views form
        separate clusters: everyone is far from the global median and the
        object is rejected. Pairwise clustering keeps the biggest agreeing
        group. Ties break on higher mean SAM score, then higher median Z
        (cups sit above the table; floor/label blobs often sit lower).

        Returns:
            (accepted_camera_ids, rejected_camera_ids)
        """
        cam_ids = sorted(by_camera.keys())
        confidences = confidences or {}
        if len(cam_ids) == 1:
            c = cam_ids[0]
            print(f"    cam {c}: centroid={np.round(by_camera[c].mean(axis=0), 4)} "
                  f"(only view)")
            return cam_ids, []

        centroids = {
            c: np.asarray(by_camera[c].mean(axis=0), dtype=float) for c in cam_ids
        }
        tol = float(self.consensus_tolerance)

        def clique_key(subset):
            scores = [float(confidences[c]) for c in subset
                      if confidences.get(c) is not None]
            mean_score = float(np.mean(scores)) if scores else 0.0
            mean_z = float(np.mean([centroids[c][2] for c in subset]))
            pair_ds = [
                np.linalg.norm(centroids[a] - centroids[b])
                for a, b in combinations(subset, 2)
            ] or [0.0]
            # Prefer: larger size, higher score, higher Z, tighter cluster.
            return (len(subset), mean_score, mean_z, -max(pair_ds))

        # Largest pairwise-agreeing cliques; among equal size, clique_key breaks ties.
        candidates = []
        for size in range(len(cam_ids), 0, -1):
            for subset in combinations(cam_ids, size):
                if all(
                    np.linalg.norm(centroids[a] - centroids[b]) <= tol
                    for a, b in combinations(subset, 2)
                ):
                    candidates.append(list(subset))
            if candidates:
                break
        best = max(candidates, key=clique_key) if candidates else []

        # Two cameras that disagree: neither is trustworthy.
        if len(cam_ids) == 2 and len(best) < 2:
            d = np.linalg.norm(centroids[cam_ids[0]] - centroids[cam_ids[1]])
            for c in cam_ids:
                print(f"    cam {c}: centroid={np.round(centroids[c], 4)} "
                      f"pairwise={d:.4f}m REJECT")
            return [], cam_ids

        accepted = best
        rejected = [c for c in cam_ids if c not in accepted]

        for c in cam_ids:
            if c in accepted and len(accepted) == 1:
                print(f"    cam {c}: centroid={np.round(centroids[c], 4)} "
                      f"OK (singleton cluster)")
                continue
            if c in accepted:
                dists = [
                    np.linalg.norm(centroids[c] - centroids[o])
                    for o in accepted if o != c
                ]
                print(f"    cam {c}: centroid={np.round(centroids[c], 4)} "
                      f"max_pair={max(dists):.4f}m OK")
            else:
                refs = accepted or [o for o in cam_ids if o != c]
                d = min(np.linalg.norm(centroids[c] - centroids[o]) for o in refs)
                print(f"    cam {c}: centroid={np.round(centroids[c], 4)} "
                      f"dist_from_cluster={d:.4f}m REJECT")

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
