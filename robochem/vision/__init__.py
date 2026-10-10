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
from .label_resolver import (LabelResolver, labels_match, looks_like_label_query,
                             normalize_label, resolve_sam_prompt)


def triangulate_rays(rays):
    """
    The point nearest every ray, in the least-squares sense.

    ``rays`` is a list of (origin, unit direction). Returns (point, the
    distance of each ray from it).
    """
    A = np.zeros((3, 3))
    b = np.zeros(3)
    projectors = []
    for origin, direction in rays:
        d = np.asarray(direction, dtype=float)
        d = d / np.linalg.norm(d)
        P = np.eye(3) - np.outer(d, d)
        projectors.append((np.asarray(origin, dtype=float), P))
        A += P
        b += P @ np.asarray(origin, dtype=float)
    point = np.linalg.solve(A, b)
    return point, [float(np.linalg.norm(P @ (point - o))) for o, P in projectors]


def cluster_silhouettes(items, plane_z, radius=0.04):
    """
    Group silhouette rays into physical containers.

    ``items`` are (camera, instance index, (origin, direction), score). Each
    ray is cut with the plane at ``plane_z`` (mid-height of the containers)
    and joins the nearest group within ``radius`` that has nothing from its
    camera yet, best scores first. Returns lists of indices into ``items``.
    """
    spots = []
    for _cam, _idx, (origin, direction), _score in items:
        if abs(direction[2]) < 1e-6:
            spots.append(None)
            continue
        spots.append((origin + direction * ((plane_z - origin[2]) / direction[2]))[:2])
    return group_spots(spots, [it[0] for it in items], [it[3] for it in items], radius)


def group_spots(spots, cams, scores, radius):
    """
    Group per-camera (x, y) sightings into physical objects.

    Best scores first, each sighting joins the nearest group within ``radius``
    that has nothing from its camera yet. A None spot is skipped. Returns
    lists of indices.
    """
    groups = []
    for i in sorted(range(len(spots)), key=lambda i: -scores[i]):
        if spots[i] is None:
            continue
        best, best_d = None, None
        for g in groups:
            if any(cams[j] == cams[i] for j in g):
                continue
            d = float(np.linalg.norm(np.mean([spots[j] for j in g], axis=0)
                                     - np.asarray(spots[i])))
            if d <= radius and (best_d is None or d < best_d):
                best, best_d = g, d
        if best is None:
            groups.append([i])
        else:
            best.append(i)
    return groups


def assign_labels(votes):
    """
    Share labels out to containers, one each, to agree with the most reads.

    ``votes[k][j]`` counts the reads of label j on container k. A label may
    go to no container. Returns (best total, every share-out reaching it),
    each a list giving label j's container index or None.
    """
    n_labels = len(votes[0]) if votes else 0
    best = [-1, []]

    def walk(j, used, share, total):
        if j == n_labels:
            if total > best[0]:
                best[0], best[1] = total, [list(share)]
            elif total == best[0]:
                best[1].append(list(share))
            return
        for k in range(len(votes)):
            if k not in used:
                walk(j + 1, used | {k}, share + [k], total + votes[k][j])
        walk(j + 1, used, share + [None], total)

    walk(0, frozenset(), [], 0)
    return best[0], best[1]


def eliminate(votes, shares, want, strong):
    """
    The container label ``want`` must be when no camera read it, or None.

    ``strong`` are the containers two or more cameras placed. It must hold
    exactly as many as there are labels, every other label must sit on one of
    them in every best share-out, read there at least once, and exactly one
    must be left.
    """
    n_labels = len(votes[0]) if votes else 0
    if len(strong) != n_labels:
        return None
    taken = set()
    for j in range(n_labels):
        if j == want:
            continue
        homes = {share[j] for share in shares}
        if len(homes) != 1:
            return None
        k = homes.pop()
        if k is None or k not in strong or votes[k][j] == 0:
            return None
        taken.add(k)
    left = [k for k in strong if k not in taken]
    return left[0] if len(left) == 1 else None


def lookalike(a, b):
    """Two labels one word apart, as "A 10 ML WATER" and "B 10 ML WATER" are."""
    ta, tb = normalize_label(a).split(), normalize_label(b).split()
    return len(ta) == len(tb) and sum(x != y for x, y in zip(ta, tb)) == 1


def misread_twin(votes, names, want, spots, strong, apart, chosen=None):
    """
    A look-alike of ``want`` (one word apart) that makes the vote for
    ``want`` a guess, or None. Two cases:

    - the twin was read on no container, and ``want`` on two or more cups:
      one of those is the twin misread as ``want``. On 2026-10-08 B's cup was
      read "A 10 ML WATER" twice and "B" never; step 23 poured A's indicator
      into B.
    - the cup the vote ``chosen`` for ``want`` is the only one the twin was
      read on: the reads there disagree, and more of them saying ``want``
      (A's own cup read once, B's read "A" 3 times and "B" once) is how B
      gets taken for A.

    ``strong`` are the containers two or more cameras placed; ``spots`` their
    (x, y); two closer than ``apart`` are one cup placed twice. Returns
    (twin's name, the cups in question).
    """
    for j in range(len(names)):
        if j == want or not lookalike(names[j], names[want]):
            continue
        read_on = [k for k in range(len(votes)) if votes[k][j]]
        if not read_on:
            cups = []
            for k in strong:
                if votes[k][want] and all(np.linalg.norm(np.asarray(spots[k])
                                                         - np.asarray(spots[c])) >= apart
                                          for c in cups):
                    cups.append(k)
            if len(cups) >= 2:
                return names[j], cups
        elif chosen is not None and read_on == [chosen]:
            return names[j], [chosen]
    return None


def cylinder_points(center_xy, base_z, radius, height, wall=0.001, n=72, levels=10):
    """
    Points on a thin-walled open cup: wall, rim (outside and inside edge) and floor.

    Shaped for ``BaseSkill.locate_container``: the top and bottom each hold
    well over 3% of the points, so its 97th/3rd percentiles land on the rim
    and the floor, and the rim's inside ring sets the opening.
    """
    cx, cy = float(center_xy[0]), float(center_xy[1])
    ang = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    ring = np.stack([np.cos(ang), np.sin(ang)], axis=1)
    top = base_z + height
    parts = []
    for z in np.linspace(base_z, top, levels):
        parts.append(np.column_stack([cx + radius * ring[:, 0], cy + radius * ring[:, 1],
                                      np.full(n, z)]))
    inner = max(radius - wall, 0.5 * radius)
    parts.append(np.column_stack([cx + inner * ring[:, 0], cy + inner * ring[:, 1],
                                  np.full(n, top)]))
    for r in (0.33 * radius, 0.66 * radius):
        parts.append(np.column_stack([cx + r * ring[:, 0], cy + r * ring[:, 1],
                                      np.full(n, base_z)]))
    parts.append(np.array([[cx, cy, base_z]]))
    return np.vstack(parts)


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
        bench=None,
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
            bench: A :class:`~robochem.vision.bench_manifest.BenchManifest`
                naming what is on the bench and how to find each object, or
                None for open-vocabulary grounding with no inventory.
        """
        self.cameras = cameras or {}
        self.bench = bench
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
        #: Table height in the base frame (m), what a transparent container
        #: stands on when it is rebuilt from its silhouette.
        self.table_z = 0.0
        #: How far one camera's silhouette ray may pass from a transparent
        #: container's triangulated centre before that camera is dropped. Four
        #: clear cups seen by 3-4 cameras each came in at 6-11 mm (2026-10-07).
        self.silhouette_ray_tolerance = 0.025

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
            if self.scene_analyzer else None,
            candidates=bench.label_texts() if bench is not None else None,
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
        phrase a skill passes -- so without a bench manifest the real cell has
        no inventory and returns empty. The simulated cell does know exactly
        what is on its bench, and overrides this, which is what lets the
        planner be told that "measuring scoop" is not a thing it can ask for
        there but "larger spoon" is. A manifest (``bench=``) gives the real
        cell the same. Callers must treat an empty list as "no constraint",
        never as "nothing is present".
        """
        return self.bench.known_object_names() if self.bench else []

    def inventory_notes(self):
        """Label and other names per bench object, as the sim gives them; {} without a manifest."""
        return self.bench.inventory_notes() if self.bench else {}

    def canonical_name(self, name):
        """The bench's name for ``name`` (a label or alias of it), else ``name`` unchanged."""
        return self.bench.canonical(name) if self.bench else name

    def set_held(self, name):
        """What the gripper holds now (None for nothing), so it is not looked for on the bench."""
        self.held_object = (self.bench.canonical(name) if (self.bench and name) else name)

    def trust_measured_tool_offset(self):
        """True when this bench's held-tool offsets come from pick_up's measurement."""
        return bool(self.bench is not None and self.bench.measured_tool_offset)

    def bench_defaults(self, skill, name):
        """Robot-validated parameters for ``skill`` on this bench object; {} if none."""
        return self.bench.defaults(skill, name) if self.bench else {}
    
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

        entry = (self.bench.find(object_name)
                 if self.bench is not None and use_labels is None else None)
        if entry is not None:
            return self._locate_bench_object(object_name, entry, category)

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

    def _locate_bench_object(self, object_name, entry, category=None):
        """
        Find a bench-manifest object the way the manifest says to.

        A labelled container is found by its label among instances of its own
        category; a category a skill passed is ignored, since the manifest is
        what was measured on this bench. If the label cannot be read the
        container is NOT retried as a direct prompt, as unknown names are:
        "clear cup a" prompted directly returns whichever clear cup scores
        highest, which is a wrong cup reported as found.
        """
        if entry.label:
            want = entry.category or category or self.labeled_cup_category
            if category and category != want:
                print(f"[VisionSystem] {object_name!r} is a {want!r} on this bench; "
                      f"not searching {category!r}")
            print(f"[VisionSystem] {object_name!r} is bench object {entry.name!r}: "
                  f"the {want!r} labelled {entry.label!r}")
            located = self.locate_labeled(entry.label, category=want, shape=entry)
            if located is None:
                print(f"[VisionSystem] No {want!r} labelled {entry.label!r} found. "
                      f"Not falling back to a direct prompt, which would return "
                      f"whichever one scores highest")
        else:
            query = entry.prompt or entry.name
            print(f"[VisionSystem] {object_name!r} is bench object {entry.name!r}: "
                  f"SAM prompt {query!r}")
            located = self._locate_direct(query)
        if located is None:
            return None
        located["bench_object"] = entry.name
        if located.get("points") is not None:
            # Skills read the cache back by the name they asked for.
            self.object_localizer.cache_object(object_name, located["points"],
                                               located.get("centroid"))
        return located

    def locate_labeled(self, label: str, category: str = None, shape=None):
        """
        Find the cup whose paper label matches ``label``.

        Segments every instance of ``category`` (default white paper cup),
        reads labels with the VLM, then fuses the matching cup in 3D.

        ``shape`` is the container's bench-manifest entry. With a bench, every
        container of the category is placed first and the labels are then
        voted across cameras (:meth:`_locate_by_vote`); a transparent one is
        placed by its silhouettes, not its depth.
        """
        category = category or self.labeled_cup_category
        if self.object_localizer is None or self.scene_analyzer is None:
            return None
        if self.scene_analyzer.grounding is None:
            print("[VisionSystem] locate_labeled needs the grounding service")
            return None
        if shape is not None and self.bench is not None:
            return self._locate_by_vote(label, category, shape)

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
        seed_cams = sorted(centroids)
        print(f"[VisionSystem] {label!r} anchored at "
              f"{np.round(anchor, 4)} from {seed_cams}; "
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
            observed[cam] = f"(projected from {seed_cams})"
            print(f"[VisionSystem] cam {cam}: matched by projection at "
                  f"({u},{v}), score {confidences[cam]:.2f}")

        return (masks, confidences, observed,
                f"projection from cameras {seed_cams}")

    # ------------------------------------------------- transparent containers
    #
    # A clear cup barely returns depth: the stereo match goes through the wall
    # to the table and paper behind it. Every camera's points therefore sit on
    # the table (z -2..-11 mm for cups 47 mm tall, 2026-10-07) and lean away
    # from that camera, so two cameras on opposite sides of the bench put one
    # cup 80 mm apart and consensus throws it out. The silhouette does not
    # have that problem: the ray through a mask's centre passes through the
    # cup's axis about half-way up, whichever side it is seen from. Measured
    # on four clear cups seen by 3-4 cameras each: depth centroids spread
    # 6 cm, silhouette rays met within 6-11 mm, 22-31 mm above the table.
    #
    # Which cup is which is then settled once for the whole bench rather than
    # per camera: every camera's label reads vote for the cup they were made
    # on, and each label goes to one cup. Per-camera matching put "A" on two
    # different cups (cams 2 and 5) whose rays happened to cross 157 mm above
    # the table.

    def _silhouette_ray(self, cam, mask, data):
        """
        World-frame (origin, unit direction) through ``mask``'s pixel centroid.

        None for a mask cut off by the frame edge: its centroid is not the
        container's. Clear cup A at the bottom edge of cam 2 pulled a
        four-camera fix 46 mm off, with rays still within 20 mm (2026-10-07).
        """
        ys, xs = np.nonzero(mask)
        intr = (data.get("intrinsics") or {}).get(cam)
        if len(xs) < 20 or intr is None:
            return None
        h, w = np.shape(mask)[:2]
        if xs.min() <= 1 or ys.min() <= 1 or xs.max() >= w - 2 or ys.max() >= h - 2:
            return None
        fx, fy = getattr(intr, "fx", None), getattr(intr, "fy", None)
        cx = getattr(intr, "cx", getattr(intr, "ppx", None))
        cy = getattr(intr, "cy", getattr(intr, "ppy", None))
        if None in (fx, fy, cx, cy):
            return None
        extr = self.object_localizer._extrinsics(cam)
        T = extr if isinstance(extr, np.ndarray) else (
            extr.matrix() if hasattr(extr, "matrix") else None)
        if T is None:
            return None
        T = np.asarray(T, dtype=float)
        d = T[:3, :3] @ np.array([(xs.mean() - cx) / fx, (ys.mean() - cy) / fy, 1.0])
        return T[:3, 3].copy(), d / np.linalg.norm(d)

    def _place_rays(self, rays, height, tag=""):
        """
        Where a transparent container stands, from its silhouettes' centre rays.

        ``rays`` maps camera -> (origin, direction). The largest set of two or
        more cameras whose rays all pass within ``silhouette_ray_tolerance`` of
        one point, at a height inside a ``height``-tall container, wins; a
        tighter fit breaks a tie. Rays that cross only in mid-air are two
        different cups. A lone ray is cut with the plane half-way up.

        Returns (point, cameras used, cameras left out, {cam: miss in m}) or None.
        """
        tol = float(self.silhouette_ray_tolerance)
        low, high = self.table_z - 0.01, self.table_z + float(height) + 0.02
        cams = sorted(rays)
        if len(cams) == 1:
            origin, direction = rays[cams[0]]
            if abs(direction[2]) < 1e-6:
                return None
            plane_z = self.table_z + float(height) / 2.0
            point = origin + direction * ((plane_z - origin[2]) / direction[2])
            return point, cams, [], {cams[0]: 0.0}
        for size in range(len(cams), 1, -1):
            best = None
            for subset in combinations(cams, size):
                point, misses = triangulate_rays([rays[c] for c in subset])
                if max(misses) > tol or not (low <= point[2] <= high):
                    continue
                if best is None or max(misses) < best[0]:
                    best = (max(misses), point, list(subset), misses)
            if best is not None:
                _, point, used, misses = best
                left = [c for c in cams if c not in used]
                if left:
                    print(f"[VisionSystem] {tag!r}: cam(s) {left} disagree with "
                          f"cams {used}; left out")
                return point, used, left, dict(zip(used, misses))
        print(f"[VisionSystem] {tag!r}: no two of cams {cams} see one container "
              f"(rays apart, or crossing outside a {float(height) * 1000:.0f}mm cup); "
              f"refusing to guess")
        return None

    def _silhouette_position(self, masks, data, shape, tag=""):
        """:meth:`_place_rays` for one container's per-camera masks."""
        rays = {}
        for cam, mask in masks.items():
            ray = self._silhouette_ray(cam, mask, data)
            if ray is not None:
                rays[cam] = ray
        if not rays:
            print(f"[VisionSystem] {tag!r}: no silhouette rays (no intrinsics?)")
            return None
        return self._place_rays(rays, shape.height, tag=tag)

    def _transparent_result(self, name, fix, shape, confidences, extra=None):
        """
        The located-object dict for a container placed by its silhouettes.

        The skills measure an opening from the points they are given (rim
        circle, top and base heights), and a clear cup's own points are the
        table. So the points handed on are a cylinder of the container's
        radius and height, standing on the table where the silhouettes put it.
        """
        point, cams, left, misses = fix
        if len(cams) > 1:
            how = (f"silhouette triangulation, cams {cams}, rays within "
                   f"{max(misses.values()) * 1000:.0f}mm, centre "
                   f"{(point[2] - self.table_z) * 1000:.0f}mm up")
        else:
            how = (f"silhouette of cam {cams[0]} cut at half height "
                   f"(one camera: expect ~2 cm)")
        points = cylinder_points(point[:2], self.table_z, float(shape.radius),
                                 float(shape.height), float(getattr(shape, "wall", 0.001)))
        self.object_localizer.cache_object(name, points)
        print(f"[VisionSystem] {name!r} (transparent) at "
              f"({point[0]:.4f}, {point[1]:.4f}): {how}")
        result = {
            "points": points,
            "centroid": self.object_localizer._cached_centroids[name],
            "dimensions": self.compute_dimensions(points),
            "confidences": {c: confidences.get(c) for c in cams},
            "cameras": cams,
            "rejected_cameras": left,
            "cached": False,
            "localized_by": how,
            "silhouette_point": [float(v) for v in point],
        }
        if extra:
            result.update(extra)
        return result

    def _locate_transparent(self, name, masks, confidences, data, shape, extra=None):
        """One transparent container from its per-camera masks."""
        fix = self._silhouette_position(masks, data, shape, tag=name)
        if fix is None:
            print(f"[VisionSystem] Could not place transparent {name!r} "
                  f"from its silhouettes")
            return None
        return self._transparent_result(name, fix, shape, confidences, extra)

    def _locate_by_vote(self, label, category, shape):
        """
        Find the bench container labelled ``label`` by placing every container
        of its category first, then letting all cameras' label reads decide
        which is which.

        1. Every instance of ``category`` in every camera is placed: a
           transparent one by its silhouette ray (:meth:`_place_rays`), an
           opaque one by its depth centroid. Instances that land together
           are one physical container.
        2. Every instance's label is read, and each container collects the
           reads of the cameras that placed it.
        3. The bench's labels of this category are shared out, one container
           each, to agree with the most reads. The requested label is found
           only if every best share-out gives it the same container, and at
           least one camera read it there.

        Per-camera matching, which this replaces for bench containers, needed
        two cameras to read a label on the same container. On 2026-10-07 cam 4
        read the baking soda dish as citric acid, cam 2 read the citric acid
        dish right, and the two 155 mm apart refused each other.
        """
        transparent = bool(getattr(shape, "transparent", False))
        held = getattr(self, "held_object", None)
        peers = [o for o in self.bench.objects if o.label and o.category == category
                 and bool(o.transparent) == transparent
                 and (o.name != held or o.name == shape.name)]
        if held and any(o.name == held for o in self.bench.objects
                        if o.category == category and o.name != shape.name):
            print(f"[VisionSystem] '{held}' is in the gripper: left out of the vote")
        names = [o.label.upper() for o in peers]
        want = [i for i, o in enumerate(peers)
                if labels_match(label, o.label) and labels_match(o.label, label)]
        if not want:
            print(f"[VisionSystem] {label!r} is no {category!r} on this bench")
            return None
        want = want[0]

        data = self.object_localizer.capture_pointclouds()
        cam_ids = [c for c in self.object_localizer.camera_ids if c in data["images"]]
        images = {c: data["images"][c] for c in cam_ids}
        if not images:
            print("[VisionSystem] No camera images available")
            return None
        print(f"[VisionSystem] Resolving {label!r}: every {category!r} placed "
              f"{'by silhouette' if transparent else 'by depth'}, then labels voted "
              f"across cameras")
        instances = self.scene_analyzer.grounding.segment_instances(images, category)
        if not instances:
            print(f"[VisionSystem] No {category!r} instances found")
            return None

        placed = []
        if transparent:
            items = []
            for cam in cam_ids:
                for idx, inst in enumerate(instances.get(cam) or []):
                    ray = self._silhouette_ray(cam, inst["mask"], data)
                    if ray is not None:
                        items.append((cam, idx, ray, float(inst.get("score") or 0.0)))
            tallest = max(float(o.height) for o in peers)
            middle = self.table_z + float(np.median([float(o.height) for o in peers])) / 2.0
            for group in cluster_silhouettes(items, middle):
                rays = {items[i][0]: items[i][2] for i in group}
                fix = self._place_rays(rays, tallest, tag=f"{category} group")
                if fix is None:
                    continue
                members = {items[i][0]: items[i][1] for i in group if items[i][0] in fix[1]}
                placed.append((fix[0], fix, members))
        else:
            spots, cams, scores, keys = [], [], [], []
            for cam in cam_ids:
                for idx, inst in enumerate(instances.get(cam) or []):
                    pts = self.object_localizer.get_object_points_by_camera(
                        {cam: inst["mask"]}, depth_images=data["depth_images"],
                        intrinsics=data["intrinsics"]).get(cam)
                    if pts is None or len(pts) < self.min_object_points:
                        continue
                    spots.append(np.asarray(pts, float).mean(axis=0))
                    cams.append(cam)
                    scores.append(float(inst.get("score") or 0.0))
                    keys.append((cam, idx))
            for group in group_spots([p[:2] for p in spots], cams, scores, radius=0.05):
                members = {keys[i][0]: keys[i][1] for i in group}
                point = np.mean([spots[i] for i in group], axis=0)
                placed.append((point, None, members))
        if not placed:
            print(f"[VisionSystem] No {category!r} could be placed")
            return None

        reads = self.label_resolver.read_all(images, instances)

        def read_of(cam, idx):
            labels_here = reads.get(cam) or []
            return labels_here[idx] if idx < len(labels_here) else None

        votes = []
        for _point, _fix, members in placed:
            row = [0] * len(names)
            for cam, idx in members.items():
                read = read_of(cam, idx)
                for j, name in enumerate(names):
                    if read and normalize_label(read) == normalize_label(name):
                        row[j] += 1
            votes.append(row)

        _best, shares = assign_labels(votes)
        for k, (point, _fix, members) in enumerate(placed):
            got = [names[j] for j, kk in enumerate(shares[0]) if kk == k]
            tally = ", ".join(f"{names[j]} x{n}" for j, n in enumerate(votes[k]) if n)
            print(f"[VisionSystem]   {category} at ({point[0]:.3f}, {point[1]:+.3f}) "
                  f"cams {sorted(members)}: reads [{tally or 'none'}] -> "
                  f"{got[0] if got else 'unassigned'}")
        strong = [i for i, (_p, _f, members) in enumerate(placed) if len(members) >= 2]
        twin = misread_twin(votes, names, want, [np.asarray(p)[:2] for p, _f, _m in placed],
                            strong, 2.0 * float(getattr(shape, "radius", None) or 0.03),
                            chosen=shares[0][want])
        if twin is not None:
            other, cups = twin
            spots = ", ".join(f"({placed[c][0][0]:.3f}, {placed[c][0][1]:+.3f})" for c in cups)
            if len(cups) > 1:
                why = (f"{names[want]!r} was read on {len(cups)} cups [{spots}] and {other!r} "
                       f"on none: one of them is {other!r} misread")
            else:
                why = (f"the cup at {spots} would be {names[want]!r}, but it is the only one "
                       f"read as {other!r}: its reads disagree")
            print(f"[VisionSystem] {why}. Refusing to guess; rewrite the {other!r} label "
                  f"(and {names[want]!r}) bigger and bolder")
            return None
        k = shares[0][want]
        if not any(row[want] for row in votes):
            # Nobody read it anywhere. The bench says how many of these there
            # are, though: if every other label is settled on a container two
            # or more cameras placed, and exactly one such container is left,
            # that is this one. On 2026-10-07 two cameras read cup C as "A"
            # and none read "C EMPTY", with A, B and the beaker all read.
            k = eliminate(votes, shares, want,
                          [i for i, (_p, _f, members) in enumerate(placed)
                           if len(members) >= 2])
            if k is None:
                print(f"[VisionSystem] No camera read {label!r} on any container, "
                      f"and it cannot be told by elimination")
                return None
            how = (f"by elimination: the only container {len(placed[k][2])} cameras "
                   f"placed that no other label claims")
        elif len({share[want] for share in shares}) > 1:
            print(f"[VisionSystem] {label!r}: the reads fit more than one container "
                  f"equally well; refusing to guess")
            return None
        elif k is None or votes[k][want] == 0:
            print(f"[VisionSystem] No camera read {label!r} on the container the "
                  f"vote left for it")
            return None
        else:
            how = None

        _point, fix, members = placed[k]
        observed = {cam: read_of(cam, idx) for cam, idx in members.items()}
        confidences = {cam: float(instances[cam][idx].get("score") or 0.0)
                       for cam, idx in members.items()}
        extra = {
            "label_matches": observed,
            "category": category,
            "resolved_by": how or (f"label vote across cameras: {votes[k][want]} of "
                                   f"{len(members)} read {names[want]!r}"),
        }
        if transparent:
            return self._transparent_result(label, fix, shape, confidences, extra)
        masks = {cam: instances[cam][idx]["mask"] for cam, idx in members.items()}
        return self._fuse_masks(label, masks, confidences, data, extra=extra)

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
