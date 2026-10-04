"""
``SimVision`` -- the perception stack, fed by rendered cameras instead of SAM.

This is not a fake that hands skills a centroid. Each of the four simulated
RealSense views is rendered for depth and for a segmentation buffer; the
segmentation supplies the mask SAM 3 would have produced, and everything after
that is the project's own code: ``ObjectLocalizer._depth_to_points`` deprojects
with the same OpenCV intrinsics maths, ``_transform_points`` applies the same
camera-to-world extrinsics, and ``VisionSystem._filter_by_consensus`` /
``_check_plausible`` accept or reject the reconstruction exactly as they do on
hardware.

So a skill running in simulation sees a real multi-view point cloud, with real
self-occlusion from the arm, and fails the same way when a container is hidden
from too many cameras.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import mujoco
import numpy as np

from robochem.vision import VisionSystem

from .bench import Prop
from .scene import SimScene

# The cage streams colour at 848x480 (robomail CameraClass defaults) with depth
# aligned to it, so that is the frame every real mask and depth image arrives in.
# Rendering narrower would give the simulated cameras a 54-degree horizontal field
# where the real ones see 69, and a prop near the edge of a real frame would fall
# outside the simulated one.
DEFAULT_WIDTH = 848
DEFAULT_HEIGHT = 480
MIN_MASK_PIXELS = 40

# Geoms in this group exist for contact only: the collision proxies that give a
# mesh prop a hollow bowl MuJoCo can collide with. They are never drawn, and
# never belong in a mask.
COLLISION_GROUP = 3


class _SimCamera:
    """
    Stands in for a RealSense handle inside ``ObjectLocalizer``.

    The localizer asks its cameras for extrinsics; giving it these means the
    real deprojection path runs untouched instead of being bypassed.
    """

    def __init__(self, extrinsics: np.ndarray):
        self._extrinsics = np.asarray(extrinsics, float)

    def get_cam_extrinsics(self) -> np.ndarray:
        return self._extrinsics


class _Intrinsics:
    """fx/fy/cx/cy in the shape ``_depth_to_points`` reads."""

    def __init__(self, fx, fy, cx, cy):
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy


class SimVision(VisionSystem):
    """VisionSystem backed by rendered MuJoCo cameras."""

    #: MuJoCo renders RGB; the cage streams BGR. Declared so that anything
    #: handing a frame to a VLM converts rather than sending it a picture with
    #: its reds and blues swapped.
    frame_color_order = "rgb"

    def __init__(
        self,
        scene: SimScene,
        width: int = DEFAULT_WIDTH,
        height: int = DEFAULT_HEIGHT,
        noise_m: float = 0.002,
        **kwargs,
    ):
        """
        Args:
            scene: Compiled scene from :func:`robochem.sim.scene.build_scene`
            width/height: Render resolution; 848x480 is the cage stream, which
                with a 42.5-degree fovy reproduces the D435 colour intrinsics
                (fx = fy ~= 617, cx = 424, cy = 240)
            noise_m: Gaussian depth noise, metres. The real cage calibrates to
                a 3-5 mm residual, so a noiseless sim would let geometry
                tolerances pass that hardware would not.
            **kwargs: Forwarded to :class:`VisionSystem` (consensus_tolerance,
                max_object_extent, ...)
        """
        # VisionSystem builds an InstructionParser, whose OpenAI client refuses
        # to construct without a key. Nothing in the simulated perception path
        # calls the VLM, so stand a placeholder up for the constructor only and
        # put the environment back -- a real key, when set, is left untouched.
        had_key = "OPENAI_API_KEY" in os.environ
        if not had_key:
            os.environ["OPENAI_API_KEY"] = "sim-no-vlm"
        try:
            super().__init__(cameras={}, **kwargs)
        finally:
            if not had_key:
                del os.environ["OPENAI_API_KEY"]

        self.scene = scene
        self.width = int(width)
        self.height = int(height)
        self.noise_m = float(noise_m)
        self._rng = np.random.default_rng(0)

        self.camera_ids: List[int] = list(scene.camera_ids)
        self.object_localizer.camera_ids = list(self.camera_ids)
        self.object_localizer.cameras = {
            cam_id: _SimCamera(scene.camera_extrinsics[cam_id])
            for cam_id in self.camera_ids
        }
        self.cameras = self.object_localizer.cameras

        # MuJoCo sizes its offscreen framebuffer from the model, and the default
        # is 640x480 -- one pixel short of the cage frame fails the Renderer.
        vis = scene.model.vis.global_
        vis.offwidth = max(int(vis.offwidth), self.width)
        vis.offheight = max(int(vis.offheight), self.height)

        # Depth and segmentation render WITHOUT multisampling. With it, the
        # resolve averages neighbouring id colours at every edge, and the
        # average spells some other geom's id: a clear cup's mask picked up a
        # pixel 0.37 m away and was rejected as "includes the table"
        # (2026-10-03). RGB keeps its antialiasing on a renderer of its own.
        samples = int(scene.model.vis.quality.offsamples)
        scene.model.vis.quality.offsamples = 0
        try:
            self._renderer = mujoco.Renderer(scene.model, self.height, self.width)
        finally:
            scene.model.vis.quality.offsamples = samples
        self._rgb_renderer = mujoco.Renderer(scene.model, self.height, self.width)

        # Render the depth and segmentation passes with EVERY geom group on.
        #
        # Not for the picture -- collision-only geoms are dropped from the mask
        # a few lines below -- but because MuJoCo's segmentation decoder indexes
        # a table sized by the number of geoms *in the scene* using ids that
        # count every geom in the *model*. Hide a group and those two stop
        # matching, and the render dies with an IndexError as soon as a visible
        # geom's id exceeds the visible count: here, the moment a cup held more
        # than ~90 granules. Keeping every geom in the scene keeps the ids dense.
        self._all_groups = mujoco.MjvOption()
        self._all_groups.geomgroup[:] = 1

        self._intrinsics = self._build_intrinsics()

    # ---------------------------------------------------------------- render

    def _build_intrinsics(self) -> Dict[int, _Intrinsics]:
        model = self.scene.model
        out = {}
        for cam_id in self.camera_ids:
            cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, f"cam{cam_id}")
            fovy = float(model.cam_fovy[cam])
            fy = (self.height / 2) / np.tan(np.deg2rad(fovy) / 2)
            out[cam_id] = _Intrinsics(fx=fy, fy=fy, cx=self.width / 2, cy=self.height / 2)
        return out

    def _render(self, cam_id: int):
        """Return (depth in metres, per-pixel body id) for one camera."""
        name = f"cam{cam_id}"
        self._renderer.disable_segmentation_rendering()
        self._renderer.enable_depth_rendering()
        self._renderer.update_scene(self.scene.data, camera=name,
                                    scene_option=self._all_groups)
        depth = np.array(self._renderer.render(), dtype=float)

        self._renderer.disable_depth_rendering()
        self._renderer.enable_segmentation_rendering()
        self._renderer.update_scene(self.scene.data, camera=name,
                                    scene_option=self._all_groups)
        geom_id = self._segment_geoms()
        self._renderer.disable_segmentation_rendering()

        body = np.full(geom_id.shape, -1, dtype=int)
        valid = geom_id >= 0
        body[valid] = self.scene.model.geom_bodyid[geom_id[valid]]
        # Collision-only geoms carry no appearance; a camera must not see them.
        hidden = np.zeros(geom_id.shape, dtype=bool)
        hidden[valid] = self.scene.model.geom_group[geom_id[valid]] == COLLISION_GROUP
        body[hidden] = -1
        return depth, body

    def _segment_geoms(self) -> np.ndarray:
        """
        Per-pixel model geom id (-1 for none) of the scene just updated.

        MuJoCo's own decode (Renderer.render with segmentation on) builds its
        lookup table from the scene's geom count and indexes it with whatever
        id each pixel's colour spells. Geoms with alpha 0 never enter the
        scene, and a few edge pixels come back as colours that are no id at
        all (seen beside the stirrer holder, ~100 px a frame). With the
        bench's drawn liquids, heaps and loads hidden at alpha 0, such a pixel
        spelled 355 against a 316-entry table and the render raised
        IndexError (2026-10-03); with a larger table it would have been
        mapped, silently, to some unrelated geom. Here an id that names no
        geom in the scene is simply nothing.
        """
        r = self._renderer
        scn = r._scene
        flags = scn.flags.copy()
        scn.flags[mujoco.mjtRndFlag.mjRND_SEGMENT] = True
        scn.flags[mujoco.mjtRndFlag.mjRND_IDCOLOR] = True
        if r._gl_context:
            r._gl_context.make_current()
        out = np.empty((self.height, self.width, 3), dtype=np.uint8)
        try:
            mujoco.mjr_render(r._rect, scn, r._mjr_context)
            mujoco.mjr_readPixels(out, None, r._rect, r._mjr_context)
        finally:
            np.copyto(scn.flags, flags)
        ids = (out[..., 0].astype(np.int64)
               | (out[..., 1].astype(np.int64) << 8)
               | (out[..., 2].astype(np.int64) << 16))
        n = scn.ngeom
        table = np.full(n + 2, -1, dtype=np.int64)
        for i in range(n):
            g = scn.geoms[i]
            if g.segid >= 0 and g.objtype == int(mujoco.mjtObj.mjOBJ_GEOM):
                table[g.segid + 1] = g.objid
        ids[(ids < 0) | (ids > n + 1)] = 0          # 0 is background
        geoms = table[ids]
        # OpenGL reads bottom row first; Renderer.render flips it the same way
        # under EGL, OSMesa and GLFW (not Filament, which has no GL context).
        return np.flipud(geoms) if r._gl_context else geoms

    def render_rgb(self, cam_id: int) -> np.ndarray:
        """One RGB frame from a cage camera, for saving alongside a run."""
        self._rgb_renderer.update_scene(self.scene.data, camera=f"cam{cam_id}")
        return np.array(self._rgb_renderer.render())

    # -------------------------------------------------------------- locating

    def capture_scene(self):
        """RGB from every cage camera, in camera-id order."""
        return [self.render_rgb(cam_id) for cam_id in self.camera_ids]

    def known_object_names(self):
        """
        Every name this bench resolves: each prop's name and its reagent label.

        The real cell cannot answer this -- grounding there is open-vocabulary --
        but the simulated one knows exactly what it was built with, and
        :meth:`Bench.find` will return nothing for a plausible name that is not
        on the list ("measuring scoop" does not reach "larger spoon"). A planner
        told the inventory asks for things that exist; one that is not spends
        its attempts discovering the list one failed skill at a time.
        """
        names = []
        for prop in self.scene.bench.props:
            names.append(prop.name)
            if prop.label:
                names.append(prop.label)
        return sorted(set(names))

    def inventory_notes(self):
        """
        What a person at the bench would know about each prop beyond its name:
        the label it stands on, and the other names it goes by.

        Keyed by prop name, one entry per physical object. The label names in
        :meth:`known_object_names` are the same objects again, which a flat
        list cannot say -- and the scene agent, given "clear cup a" and
        "a 10 ml water" as two names, reported the cup empty (2026-10-02).
        """
        out = {}
        for prop in self.scene.bench.props:
            bits = []
            if prop.label:
                bits.append(f'stands on a handwritten label reading "{prop.label}", '
                            f'and "{prop.label}" names it too')
            if prop.aliases:
                bits.append("also called " + ", ".join(f'"{a}"' for a in prop.aliases))
            out[prop.name] = "; ".join(bits)
        return out

    def _masks_for(self, prop: Prop):
        """Per-camera (mask, depth_mm) for one prop, as SAM would have returned."""
        target = self.scene.prop_bodies[prop.name]
        masks, depths = {}, {}
        for cam_id in self.camera_ids:
            depth_m, body = self._render(cam_id)
            mask = body == target
            if mask.sum() < MIN_MASK_PIXELS:
                continue
            if self.noise_m:
                depth_m = depth_m + self._rng.normal(0, self.noise_m, depth_m.shape)
            masks[cam_id] = mask
            depths[cam_id] = depth_m * 1000.0      # _depth_to_points expects mm
        return masks, depths

    def locate(self, object_name, force_refresh: bool = False,
               category: str = None, use_labels: bool = None):
        """
        Reconstruct a prop from the simulated cage.

        Same contract as :meth:`VisionSystem.locate`: a dict with points,
        centroid, dimensions and the contributing cameras, or None when the
        views cannot agree on anything plausible.
        """
        if not force_refresh and object_name in self.object_localizer._cached_centroids:
            points = self.object_localizer._cached_pointclouds.get(object_name)
            return {
                "points": points,
                "centroid": self.object_localizer._cached_centroids[object_name],
                "dimensions": self.compute_dimensions(points) if points is not None else None,
                "cached": True,
            }

        prop = self.scene.bench.find(object_name)
        if prop is None:
            print(f"[SimVision] Nothing on the bench matches {object_name!r}")
            return None
        if prop.name.lower() != str(object_name).lower():
            print(f"[SimVision] {object_name!r} -> {prop.name!r}"
                  + (f" (label {prop.label!r})" if prop.label else ""))

        masks, depths = self._masks_for(prop)
        if not masks:
            print(f"[SimVision] {prop.name!r} is not visible from any camera")
            return None

        by_camera = self.object_localizer.get_object_points_by_camera(
            masks, depth_images=depths, intrinsics=self._intrinsics
        )
        if not by_camera:
            print(f"[SimVision] No 3D points reconstructed for {prop.name!r}")
            return None

        agreed, rejected = self._filter_by_consensus(by_camera)
        if not agreed:
            print(f"[SimVision] Cameras could not agree on {prop.name!r}")
            return None

        points = self.object_localizer._remove_outliers(
            np.vstack([by_camera[c] for c in agreed])
        )
        plausible, reason = self._check_plausible(points)
        if not plausible:
            print(f"[SimVision] Rejecting {prop.name!r}: {reason}")
            return None

        self.object_localizer.cache_object(object_name, points)
        return {
            "points": points,
            "centroid": self.object_localizer._cached_centroids[object_name],
            "dimensions": self.compute_dimensions(points),
            "cameras": agreed,
            "rejected_cameras": rejected,
            "label": prop.label,
            "prop": prop.name,
            "cached": False,
        }

    def locate_labeled(self, label: str, category: str = None):
        """Reagent labels resolve straight through :meth:`locate` in sim."""
        return self.locate(label)

    def close(self):
        """Release the offscreen render context."""
        for attr in ("_renderer", "_rgb_renderer"):
            renderer = getattr(self, attr, None)
            if renderer is not None:
                renderer.close()
                setattr(self, attr, None)

    def ground_truth(self, object_name: str) -> Optional[np.ndarray]:
        """The prop's true world position -- for checking what perception did."""
        prop = self.scene.bench.find(object_name)
        if prop is None:
            return None
        return self.scene.data.xpos[self.scene.prop_bodies[prop.name]].copy()
