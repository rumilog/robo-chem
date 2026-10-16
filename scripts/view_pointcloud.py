"""
See the cage's point cloud: all four cameras fused in the robot base frame,
cleaned up, with every bench object outlined, coloured and labelled.
Read-only; the robot is never commanded.

What it does:
  1. Captures each camera on its own (streaming all four together has wedged
     this host's USB), at 1280x720 (--size; the D415's best depth). Several
     frames per camera are reduced to a per-pixel median, and pixels whose
     depth flickers between frames (the veils at object edges) are dropped.
  2. Drops depth jumps (flying pixels), moves everything to the base frame
     with the skills' extrinsics, and evens out each camera's exposure on
     patches of table every camera sees.
  3. --mode mesh (default) fuses the four depth images into one surface
     (TSDF, --tsdf-voxel): noise, the cameras' residual mm offsets and flying
     pixels average out, and walls come out as walls. --mode points merges
     the raw points on a voxel grid instead (--voxel). Clear cups cannot be
     measured by depth at all; they are drawn from their model.
  4. Locates every bench object the way the skills do (vision.locate) and
     gives each its own colour. Containers get their real outline from the
     bench manifest (a cylinder of the measured radius and height, centred by
     a rim fit for the dishes, by silhouette triangulation for clear cups,
     which also get a translucent shell because depth passes through them).
     Tools get an oriented box. Each object is labelled.
  5. Writes, under diag_out/pointcloud/:
       pointcloud_<time>.ply         the cleaned, coloured cloud
       pointcloud_<time>.png         perspective view, labels + legend
       pointcloud_<time>_top.png     top-down view, the outlines as circles
       pointcloud_<time>.mp4         turntable, with --video
       pointcloud_<time>_raw.npy     the captures and located objects; re-render
                                     with --from-raw, no cameras needed
     then opens an interactive window with the labels floating over each
     object (drag to rotate, scroll to zoom), unless --no-show.

--per-camera colours points by camera (2 red, 3 green, 4 blue, 5 yellow) and
skips the merge: where the four disagree, calibration is off.

Usage:
    source scripts/env.sh
    python scripts/view_pointcloud.py                        # every bench object
    python scripts/view_pointcloud.py --objects "clear cup a" "larger spoon"
    python scripts/view_pointcloud.py --no-objects           # the cloud only
    python scripts/view_pointcloud.py --video --no-show
    python scripts/view_pointcloud.py --per-camera
    python scripts/view_pointcloud.py --from-raw diag_out/pointcloud/pointcloud_<time>_raw.npy
"""

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from robochem.vision import REAL_TABLE_Z
from robochem.vision.bench_manifest import BENCHES

CAMERA_COLOURS = {2: (1.0, 0.25, 0.25), 3: (0.25, 0.9, 0.3), 4: (0.25, 0.45, 1.0),
                  5: (1.0, 0.85, 0.1)}
# Distinct on a dark background and on the dimmed brown table.
PALETTE = [(0.12, 0.62, 1.00), (1.00, 0.55, 0.05), (0.20, 0.85, 0.30),
           (1.00, 0.25, 0.30), (0.70, 0.40, 1.00), (0.00, 0.90, 0.85),
           (1.00, 0.40, 0.80), (0.85, 0.90, 0.10), (1.00, 0.80, 0.30),
           (0.55, 0.75, 1.00), (0.60, 1.00, 0.70), (1.00, 0.60, 0.55)]
BACKGROUND = (0.07, 0.07, 0.09)
# Looking at the bench from the front-right, above it.
LOOK_AT = np.array([0.50, 0.0, 0.0])


# ------------------------------------------------------------------ capture
def intr_dict(intr):
    cx = getattr(intr, "cx", getattr(intr, "ppx", None))
    cy = getattr(intr, "cy", getattr(intr, "ppy", None))
    return {"fx": float(intr.fx), "fy": float(intr.fy), "cx": float(cx), "cy": float(cy)}


def open_camera(vis, cam_id, size):
    """The camera at ``size`` (the D415's best depth is 1280x720), else its default."""
    try:
        return vis.CameraClass(cam_number=cam_id, W=int(size[0]), H=int(size[1]))
    except Exception as e:
        print(f"cam {cam_id}: {size[0]}x{size[1]} refused ({e}); the default 848x480")
        return vis.CameraClass(cam_number=cam_id)


def capture(cam_ids, frames, warmup, size=(1280, 720)):
    """
    One camera at a time: per-pixel median depth over ``frames`` frames.

    A pixel is kept only where it had depth in most frames and did not
    flicker: on an object's silhouette the depth jumps between the object and
    what is behind it from frame to frame, which is where the veils come from.
    """
    import robomail.vision as vis
    from robomail.vision.cam_utils import get_cam_info
    from robochem.vision.object_localizer import get_live_intrinsics

    out = {}
    for cam_id in cam_ids:
        cam = None
        try:
            print(f"cam {cam_id}: {warmup} warm-up + {frames} frames...")
            cam = open_camera(vis, cam_id, size)
            for _ in range(warmup):
                cam.get_next_frame(get_point_cloud=False, get_verts=False)
            depths, colour = [], None
            for _ in range(frames):
                img, depth, _, _ = cam.get_next_frame(get_point_cloud=False, get_verts=False)
                depths.append(np.asarray(depth, dtype=np.float32))
                colour = np.asarray(img).copy()
            stack = np.stack(depths)
            stack[stack <= 0] = np.nan
            seen = np.sum(np.isfinite(stack), axis=0)
            with np.errstate(all="ignore"):
                med = np.nanmedian(stack, axis=0)
                spread = np.nanstd(stack, axis=0)
            # Parenthesised: & binds tighter than >=, and without them every
            # pixel passed (2026-10-10 19:09: "407040 steady of 407040").
            steady = ((seen >= max(1, int(np.ceil(0.6 * frames))))
                      & (spread <= 3.0 + 0.005 * med))       # mm
            med = np.nan_to_num(med, nan=0.0)
            depth_mm = np.where(steady, med, 0).astype(np.uint16)
            out[cam_id] = {"depth": depth_mm, "colour": colour,
                           "intr": intr_dict(get_live_intrinsics(cam)),
                           "T": np.asarray(get_cam_info(cam_id)[1], dtype=float)}
            print(f"cam {cam_id}: {int(steady.sum())} steady pixels of {steady.size}")
        except Exception as e:
            print(f"cam {cam_id}: capture failed: {type(e).__name__}: {e}")
        finally:
            if cam is not None:
                try:
                    cam.stop_pipeline()
                except Exception:
                    pass
    return out


def locate_objects(names, bench, grounding_url, cam_ids):
    """Every named object as the skills see it: its world points and centroid."""
    from robochem.vision import VisionSystem
    vision = VisionSystem(cameras={}, grounding_url=grounding_url, bench=bench)
    vision.object_localizer.camera_ids = list(cam_ids)
    vision.clear_cache()
    found = []
    for name in names:
        print(f"\nLocating '{name}'...")
        try:
            located = vision.locate(name)
        except Exception as e:
            print(f"'{name}': locate failed: {type(e).__name__}: {e}")
            continue
        if located is None or located.get("points") is None or len(located["points"]) == 0:
            print(f"'{name}': not found")
            continue
        pts = np.asarray(located["points"], dtype=np.float32)
        centroid = np.asarray(located.get("centroid", pts.mean(axis=0)), dtype=float)
        found.append({"name": name, "points": pts, "centroid": centroid})
        print(f"'{name}': {len(pts)} points, centre ({centroid[0]:.3f}, {centroid[1]:+.3f})")
    return found


# ----------------------------------------------------------- cloud building
def clean_depth(rec):
    """Depth in metres with flying pixels (depth edges) zeroed."""
    z = rec["depth"].astype(float) / 1000.0
    # A pixel more than a few mm (growing with range) from any 4-neighbour
    # sits on a depth edge: its depth mixes the two sides. A neighbour with no
    # depth counts as an edge too.
    jump = np.zeros_like(z)
    for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        nb = np.roll(np.roll(z, dy, axis=0), dx, axis=1)
        d = np.abs(z - nb)
        d[nb <= 0] = np.inf
        jump = np.maximum(jump, d)
    jump[[0, -1], :] = np.inf
    jump[:, [0, -1]] = np.inf
    keep = (z > 0.1) & (z < 2.0) & (jump < 0.004 + 0.012 * z)
    return np.where(keep, z, 0.0)


def camera_cloud(rec, stride):
    """World points and RGB (0-1) of one camera, flying pixels dropped."""
    z = clean_depth(rec)
    h, w = z.shape
    keep = z > 0
    v, u = np.mgrid[0:h:stride, 0:w:stride]
    v, u = v.ravel(), u.ravel()
    ok = keep[v, u]
    v, u = v[ok], u[ok]
    zz = z[v, u]
    k = rec["intr"]
    cam = np.column_stack([(u - k["cx"]) * zz / k["fx"], (v - k["cy"]) * zz / k["fy"], zz])
    T = rec["T"]
    world = cam @ T[:3, :3].T + T[:3, 3]
    rgb = rec["colour"][v, u][:, ::-1].astype(float) / 255.0          # BGR -> RGB
    return world, rgb


def balance_exposure(parts, table_z, cell=0.01):
    """
    Scale each camera's colours so the same patch of table looks the same.

    The cameras run their own auto-exposure: the same table read 0.18 in
    cams 2/3 and 0.30 in cams 4/5 (2026-10-10), a patchwork once merged.
    Compared cell by cell (1 cm on the tabletop) over cells every camera
    sees, so white paper filling more of one camera's view than another's
    does not skew it, as a plain median over each view did.
    """
    cells = {}
    for cam_id, (pts, rgb) in parts.items():
        flat = (np.abs(pts[:, 2] - table_z) < 0.004)
        key = np.floor(pts[flat, :2] / cell).astype(np.int64)
        code = key[:, 0] * 100003 + key[:, 1]
        order = np.argsort(code)
        code, col = code[order], rgb[flat][order]
        uniq, first, count = np.unique(code, return_index=True, return_counts=True)
        keep = count >= 3
        cells[cam_id] = {u: np.median(col[f:f + n], axis=0)
                         for u, f, n in zip(uniq[keep], first[keep], count[keep])}
    if len(cells) < 2:
        return {}
    shared = set.intersection(*(set(c) for c in cells.values()))
    if len(shared) < 50:
        print(f"Exposure: only {len(shared)} shared table cells; colours left as captured")
        return {}
    shared = sorted(shared)
    stack = {c: np.array([cells[c][k] for k in shared]) for c in cells}
    mean = np.mean(np.array(list(stack.values())), axis=0)
    gains = {c: np.median(mean / np.maximum(stack[c], 1e-3), axis=0) for c in stack}
    norm = np.exp(np.mean(np.log(np.array(list(gains.values()))), axis=0))
    out = {}
    for cam_id in parts:
        out[cam_id] = np.clip(gains.get(cam_id, norm) / norm, 0.5, 2.0)
        print(f"cam {cam_id}: colour gain {np.round(out[cam_id], 2)} ({len(shared)} shared cells)")
    return out


def build_scene(raw, args, table_z):
    import open3d as o3d
    lo, hi = np.array(args.box[:3]), np.array(args.box[3:])
    parts = {}
    for cam_id, rec in sorted(raw.items()):
        pts, rgb = camera_cloud(rec, max(1, args.stride))
        inside = np.all((pts >= lo) & (pts <= hi), axis=1)
        parts[cam_id] = (pts[inside], rgb[inside])
        print(f"cam {cam_id}: {int(inside.sum())} points on the bench")
    if not parts:
        return None
    if args.per_camera:
        parts = {c: (p, np.tile(CAMERA_COLOURS.get(c, (1, 1, 1)), (len(p), 1)))
                 for c, (p, _) in parts.items()}
    else:
        gains = balance_exposure(parts, table_z)
        parts = {c: (p, np.clip(rgb * gains.get(c, 1.0), 0, 1)) for c, (p, rgb) in parts.items()}

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.vstack([p for p, _ in parts.values()]))
    pcd.colors = o3d.utility.Vector3dVector(np.vstack([c for _, c in parts.values()]))
    n0 = len(pcd.points)
    if not args.per_camera and args.voxel > 0:
        # Where cameras overlap their points merge into one, colours averaged.
        pcd = pcd.voxel_down_sample(args.voxel)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=16, std_ratio=2.0)
    radius = max(4 * args.voxel, 0.006)
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30))
    pcd.orient_normals_towards_camera_location(np.array([0.5, 0.0, 1.2]))
    print(f"Merged: {n0} points -> {len(pcd.points)} after the "
          f"{args.voxel * 1000:.1f} mm grid and outlier removal")
    return pcd


def build_mesh(raw, args, table_z):
    """
    The four cameras fused into one surface (TSDF), coloured, cropped.

    Each depth image is a noisy measurement of the same surfaces; a truncated
    signed distance volume averages them, so noise, residual calibration
    offsets between cameras and flying pixels that only one view sees wash
    out, and walls come out as walls instead of four fuzzy shells of points.
    On the 19:09 capture the dishes came out as solid cylinders and the black
    stirrer holder as one column, where the merged points showed blobs.
    Clear cups cannot appear: depth passes through them. They are drawn from
    their model instead (describe_objects).
    """
    import open3d as o3d
    sample = {c: camera_cloud(rec, 2) for c, rec in raw.items()}
    gains = balance_exposure(sample, table_z)
    vox = float(args.tsdf_voxel)
    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=vox, sdf_trunc=4 * vox,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    for cam_id, rec in sorted(raw.items()):
        depth = clean_depth(rec)
        h, w = depth.shape
        k = rec["intr"]
        intr = o3d.camera.PinholeCameraIntrinsic(w, h, k["fx"], k["fy"], k["cx"], k["cy"])
        rgb = rec["colour"][:, :, ::-1].astype(float) * gains.get(cam_id, 1.0)
        colour = o3d.geometry.Image(np.ascontiguousarray(np.clip(rgb, 0, 255).astype(np.uint8)))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            colour, o3d.geometry.Image((depth * 1000).astype(np.uint16)), depth_scale=1000.0,
            depth_trunc=1.3, convert_rgb_to_intensity=False)
        vol.integrate(rgbd, intr, np.linalg.inv(rec["T"]))
    mesh = vol.extract_triangle_mesh()
    mesh = mesh.crop(o3d.geometry.AxisAlignedBoundingBox(args.box[:3], args.box[3:]))
    tri, size, _ = mesh.cluster_connected_triangles()
    mesh.remove_triangles_by_mask(np.asarray(size)[np.asarray(tri)] < 200)
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    print(f"TSDF at {vox * 1000:.1f} mm: {len(mesh.vertices)} vertices, "
          f"{len(mesh.triangles)} triangles")
    return mesh


# ----------------------------------------------------------------- objects
def object_shape(entry):
    """(radius, height) of a container on this bench, or None for a tool."""
    if entry is None:
        return None
    if entry.radius and entry.height:
        return float(entry.radius), float(entry.height)
    scoop = dict(entry.scoop)
    if scoop.get("container_radius") and scoop.get("container_height"):
        # The dishes record their inside radius; the outside is ~2 mm more.
        return float(scoop["container_radius"]) + 0.002, float(scoop["container_height"])
    return None


def fit_rim_centre(P, guess_xy, radius, top_z):
    """Circle through the rim-top points near ``guess_xy``, or None."""
    near = (np.hypot(P[:, 0] - guess_xy[0], P[:, 1] - guess_xy[1]) < radius + 0.02)
    band = near & (P[:, 2] > top_z - 0.012) & (P[:, 2] < top_z + 0.006)
    xy = P[band, :2]
    if len(xy) < 50:
        return None
    A = np.column_stack([2 * xy[:, 0], 2 * xy[:, 1], np.ones(len(xy))])
    b = (xy ** 2).sum(axis=1)
    rng = np.random.default_rng(0)
    best = None
    for _ in range(300):
        i = rng.choice(len(xy), 3, replace=False)
        try:
            cx, cy, c = np.linalg.solve(A[i], b[i])
        except np.linalg.LinAlgError:
            continue
        r = np.sqrt(max(c + cx * cx + cy * cy, 0.0))
        if not (radius - 0.010 < r < radius + 0.006):
            continue
        inl = np.abs(np.hypot(xy[:, 0] - cx, xy[:, 1] - cy) - r) < 0.003
        if best is None or inl.sum() > best.sum():
            best = inl
    if best is None or best.sum() < 30:
        return None
    cx, cy, _ = np.linalg.lstsq(A[best], b[best], rcond=None)[0]
    if np.hypot(cx - guess_xy[0], cy - guess_xy[1]) > 0.03:
        return None
    return np.array([cx, cy])


def positions(geom):
    """The points of a cloud, or the vertices of a mesh."""
    return np.asarray(geom.vertices if hasattr(geom, "vertices") else geom.points)


def describe_objects(geom, objects, bench, table_z):
    """
    Give each located object its colour, its share of the scene (points or
    mesh vertices), an outline and a label anchor. Returns the owner index of
    every point (-1: none).

    A clear container owns nothing: what the depth shows inside its outline
    is the table and paper seen through it, and colouring that as the cup
    drew blobs where the cup should be (2026-10-10 19:09). It is drawn from
    its model, a translucent shell of the manifest's size.
    """
    import open3d as o3d
    from scipy.spatial import cKDTree

    P = positions(geom)
    owner = np.full(len(P), -1, dtype=int)
    for i, obj in enumerate(objects):
        entry = bench.find(obj["name"]) if bench is not None else None
        obj["colour"] = PALETTE[i % len(PALETTE)]
        obj["title"] = entry.name if entry is not None else obj["name"]
        obj["subtitle"] = entry.label.upper() if entry is not None and entry.label else ""
        obj["transparent"] = bool(entry is not None and entry.transparent)
        shape = object_shape(entry)
        if shape is not None:
            radius, height = shape
            top = table_z + height
            centre = None
            if not obj["transparent"]:
                centre = fit_rim_centre(P, obj["centroid"][:2], radius, top)
            if centre is None:
                centre = np.asarray(obj["centroid"][:2], dtype=float)
            if not obj["transparent"]:
                d = np.hypot(P[:, 0] - centre[0], P[:, 1] - centre[1])
                inside = ((d <= radius + 0.004) & (P[:, 2] > table_z + 0.002)
                          & (P[:, 2] < top + 0.006))
                owner[inside & (owner < 0)] = i
            # Just outside the wall and over the rim, or the points hide it.
            obj["outline"] = cylinder_outline(centre, radius + 0.003, table_z + 0.001, top + 0.002,
                                              verticals=4 if obj["transparent"] else 8)
            obj["shell"] = (centre, radius, table_z, height) if obj["transparent"] else None
            obj["anchor"] = np.array([centre[0], centre[1], top])
            obj["centre_xy"] = centre
            obj["size"] = f"{radius * 2000:.0f} x {height * 1000:.0f} mm"
        else:
            pts = np.asarray(obj["points"], dtype=float)
            lo, hi = pts.min(axis=0) - 0.01, pts.max(axis=0) + 0.01
            box = np.all((P >= lo) & (P <= hi), axis=1)
            idx = np.where(box & (owner < 0))[0]
            if len(idx):
                dist, _ = cKDTree(pts).query(P[idx], distance_upper_bound=0.005)
                owner[idx[np.isfinite(dist)]] = i
            mine = P[owner == i] if (owner == i).sum() >= 50 else pts
            sub = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(mine))
            sub, _ = sub.remove_statistical_outlier(nb_neighbors=12, std_ratio=2.0)
            obb = sub.get_oriented_bounding_box()
            obj["outline"] = o3d.geometry.LineSet.create_from_oriented_bounding_box(obb)
            obj["shell"] = None
            top_z = float(np.asarray(sub.points)[:, 2].max())
            obj["anchor"] = np.array([obb.center[0], obb.center[1], top_z])
            obj["centre_xy"] = np.asarray(obb.center[:2])
            ext = np.sort(np.asarray(obb.extent))[::-1]
            obj["size"] = f"{ext[0] * 1000:.0f} x {ext[1] * 1000:.0f} mm"
        obj["outline"].paint_uniform_color(0.6 * np.array(obj["colour"]) + 0.4)
        shown = "model only" if obj["transparent"] else f"{int((owner == i).sum())} points"
        print(f"  {obj['title']:24s} {shown:>13s}, "
              f"centre ({obj['centre_xy'][0]:.3f}, {obj['centre_xy'][1]:+.3f}), {obj['size']}")
    return owner


def cylinder_outline(centre, radius, z0, z1, segments=72, verticals=8):
    import open3d as o3d
    t = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    ring = np.column_stack([centre[0] + radius * np.cos(t), centre[1] + radius * np.sin(t)])
    pts = np.vstack([np.column_stack([ring, np.full(segments, z0)]),
                     np.column_stack([ring, np.full(segments, z1)])])
    lines = [[i, (i + 1) % segments] for i in range(segments)]
    lines += [[segments + i, segments + (i + 1) % segments] for i in range(segments)]
    step = segments // verticals
    lines += [[i, segments + i] for i in range(0, segments, step)]
    return o3d.geometry.LineSet(o3d.utility.Vector3dVector(pts), o3d.utility.Vector2iVector(lines))


def colour_scene(geom, objects, owner, dim):
    """Objects in their colour over their own shading; everything else dimmed."""
    import open3d as o3d
    mesh = hasattr(geom, "vertex_colors")
    rgb = np.asarray(geom.vertex_colors if mesh else geom.colors).copy()
    lum = rgb @ np.array([0.299, 0.587, 0.114])
    if dim:
        grey = np.repeat(lum[:, None], 3, axis=1)
        rgb = 0.6 * (0.75 * rgb + 0.25 * grey)
    for i, obj in enumerate(objects):
        mine = owner == i
        shade = 0.45 + 0.45 * lum[mine, None]
        rgb[mine] = np.clip(np.array(obj["colour"]) * shade + 0.08, 0, 1)
    if mesh:
        geom.vertex_colors = o3d.utility.Vector3dVector(rgb)
    else:
        geom.colors = o3d.utility.Vector3dVector(rgb)
    return geom


# --------------------------------------------------------------- rendering
def view_eye(azimuth_deg, elevation_deg, distance, look_at=LOOK_AT):
    a, e = np.radians(azimuth_deg), np.radians(elevation_deg)
    return look_at + distance * np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])


def shell_mesh(obj, segments=64):
    """
    A clear container's model: an open cup (wall and floor, no lid) of its
    measured size, where its silhouettes put it. Depth cannot see clear
    plastic, so this is the only way the cup appears in the cloud at all.
    """
    import open3d as o3d
    centre, radius, z0, h = obj["shell"]
    t = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    ring = np.column_stack([centre[0] + radius * np.cos(t), centre[1] + radius * np.sin(t)])
    bottom = np.column_stack([ring, np.full(segments, z0 + 0.0005)])
    top = np.column_stack([ring, np.full(segments, z0 + h)])
    middle = np.array([[centre[0], centre[1], z0 + 0.0005]])
    verts = np.vstack([bottom, top, middle])
    tris = []
    for i in range(segments):
        j = (i + 1) % segments
        tris += [[i, j, segments + j], [i, segments + j, segments + i]]   # wall
        tris += [[2 * segments, j, i]]                                       # floor
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(verts),
                                     o3d.utility.Vector3iVector(np.array(tris)))
    mesh.compute_vertex_normals()
    return mesh


def glass_material(obj):
    from open3d.visualization import rendering
    glass = rendering.MaterialRecord()
    glass.shader = "defaultLitTransparency"
    # Frosted: the object's colour lightened, mostly opaque, so the cup reads
    # as a cup and not as a wire cage (the user, 2026-10-10).
    tint = 0.55 * np.array(obj["colour"]) + 0.45
    glass.base_color = [*tint, 0.7]
    return glass


def make_renderer(pcd, objects, width, height, point_size):
    import open3d as o3d
    from open3d.visualization import rendering
    r = rendering.OffscreenRenderer(width, height)
    r.scene.set_background([*BACKGROUND, 1.0])
    pts = rendering.MaterialRecord()
    pts.shader = "defaultLit"
    pts.point_size = point_size
    r.scene.add_geometry("scene", pcd, pts)
    line = rendering.MaterialRecord()
    line.shader = "unlitLine"
    line.line_width = 5.0
    for i, obj in enumerate(objects):
        r.scene.add_geometry(f"outline{i}", obj["outline"], line)
        if obj.get("shell") is not None:
            r.scene.add_geometry(f"shell{i}", shell_mesh(obj), glass_material(obj))
    r.scene.scene.set_sun_light([-0.35, 0.25, -1.0], [1.0, 1.0, 1.0], 55000)
    r.scene.scene.enable_sun_light(True)
    r.scene.scene.set_indirect_light_intensity(35000)
    return r


def project(renderer, width, height, xyz):
    V = np.asarray(renderer.scene.camera.get_view_matrix())
    Pm = np.asarray(renderer.scene.camera.get_projection_matrix())
    c = Pm @ V @ np.append(np.asarray(xyz, dtype=float), 1.0)
    if c[3] <= 1e-6:
        return None
    n = c[:3] / c[3]
    return np.array([(n[0] + 1) / 2 * width, (1 - n[1]) / 2 * height])


def draw_labels(img, objects, renderer, width, height, scale, avoid=(), on_object=False):
    """
    A tag per object, never overlapping another or the legend. In a
    perspective view it sits above the object on a leader line; seen from
    above (``on_object``) it sits on the object itself, like a map.
    """
    import cv2
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs, th = 0.55 * scale, max(1, int(round(1.4 * scale)))
    pad, lift = int(6 * scale), int(34 * scale)
    placed = list(avoid)
    order = []
    for obj in objects:
        p = project(renderer, width, height, obj["anchor"] + np.array([0, 0, 0.004]))
        if p is not None:
            order.append((p[1], obj, p))
    for _, obj, p in sorted(order, key=lambda t: t[0]):
        text = obj["title"]
        (tw, tht), base = cv2.getTextSize(text, font, fs, th)
        bw, bh = tw + 2 * pad, tht + base + 2 * pad
        def free(x, y):
            return (2 <= x <= width - bw - 2 and 2 <= y <= height - bh - 2 and
                    all(x + bw + 3 < b[0] or b[0] + b[2] + 3 < x or y + bh + 3 < b[1]
                        or b[1] + b[3] + 3 < y for b in placed))
        # Above the object first, then further up, below, and to either side.
        step = bh + 6
        tries = [(0, -lift - bh - k * step) for k in range(6)]
        tries += [(0, lift + k * step) for k in range(4)]
        if on_object:
            tries = [(0, -bh / 2), (0, -bh / 2 + step), (0, -bh / 2 - step)] + tries
        tries += [(sx * (bw / 2 + 20), -bh / 2 + k * step) for k in (0, -1, 1, -2, 2)
                  for sx in (1, -1)]
        x0 = p[0] - bw / 2
        spot = next(((int(x0 + dx), int(p[1] + dy)) for dx, dy in tries
                     if free(int(x0 + dx), int(p[1] + dy))), None)
        if spot is None:
            spot = (int(np.clip(x0, 2, width - bw - 2)),
                    int(np.clip(p[1] - lift - bh, 2, height - bh - 2)))
        x, y = spot
        placed.append((x, y, bw, bh))
        col = tuple(int(255 * c) for c in obj["colour"][::-1])        # BGR
        touching = x <= p[0] <= x + bw and y <= p[1] <= y + bh
        if not touching:
            end = (x + bw // 2, y + bh if y + bh <= p[1] else y)
            cv2.line(img, (int(p[0]), int(p[1])), end, col, th, cv2.LINE_AA)
            cv2.circle(img, (int(p[0]), int(p[1])), int(4 * scale), col, -1, cv2.LINE_AA)
        cv2.rectangle(img, (x, y), (x + bw, y + bh), col, -1, cv2.LINE_AA)
        dark = sum(c * w for c, w in zip(obj["colour"], (0.299, 0.587, 0.114))) > 0.55
        ink = (20, 20, 20) if dark else (255, 255, 255)
        cv2.putText(img, text, (x + pad, y + pad + tht), font, fs, ink, th, cv2.LINE_AA)


def draw_legend(img, objects, title, scale, bottom=False, missing=()):
    import cv2
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs, th = 0.5 * scale, max(1, int(round(1.2 * scale)))
    row = int(24 * scale)
    lines = [(obj["colour"], f"{obj['title']}" + (f"  [{obj['subtitle']}]" if obj["subtitle"] else "")
              + f"  {obj['size']}" + ("  (model)" if obj.get("transparent") else ""))
             for obj in objects]
    if any(obj.get("transparent") for obj in objects):
        lines.append(((0.45, 0.45, 0.45), "model: clear plastic is invisible to depth; "
                                          "its measured size, placed by 4 camera silhouettes"))
    if missing:
        lines.append(((0.45, 0.45, 0.45), "not found: " + ", ".join(missing)))
    w = int(max([cv2.getTextSize(t, font, fs, th)[0][0] for _, t in lines]
                + [cv2.getTextSize(title, font, fs * 1.1, th)[0][0]]) + 50 * scale)
    h = int(row * (len(lines) + 1.6))
    top = img.shape[0] - h - 10 if bottom else 10
    overlay = img.copy()
    box = (10, top, w, h)
    cv2.rectangle(overlay, (10, top), (10 + w, top + h), (25, 25, 30), -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
    cv2.putText(img, title, (22, top + row), font, fs * 1.1, (235, 235, 235), th, cv2.LINE_AA)
    for k, (colour, text) in enumerate(lines):
        y = top + int(row * (k + 2.1))
        col = tuple(int(255 * c) for c in colour[::-1])
        cv2.rectangle(img, (22, y - int(12 * scale)), (22 + int(16 * scale), y + int(2 * scale)),
                      col, -1)
        cv2.putText(img, text, (22 + int(26 * scale), y), font, fs, (230, 230, 230), th,
                    cv2.LINE_AA)
    return box


def render_view(renderer, objects, width, height, eye, fov, title, scale, look_at=LOOK_AT,
                up=(0, 0, 1), map_view=False, missing=()):
    import cv2
    renderer.setup_camera(fov, look_at, eye, list(up))
    img = cv2.cvtColor(np.asarray(renderer.render_to_image()), cv2.COLOR_RGB2BGR)
    if objects:
        legend = draw_legend(img, objects, title, scale, bottom=map_view, missing=missing)
        draw_labels(img, objects, renderer, width, height, scale, avoid=[legend],
                    on_object=map_view)
    return img


def render_outputs(pcd, objects, stamp, out, args, missing=()):
    import cv2
    W, H = args.width, args.height
    scale = W / 1280.0
    r = make_renderer(pcd, objects, W, H, args.point_size)
    title = f"robo-chem bench, cameras {','.join(map(str, args.cameras))}, {stamp}"
    eye = view_eye(-35, 42, 1.05)
    png = out / f"pointcloud_{stamp}.png"
    cv2.imwrite(str(png), render_view(r, objects, W, H, eye, 40, title, scale, missing=missing))
    print(f"Saved {png}")
    # Straight down, a hair off vertical so 'up' stays defined: +x (away from
    # the arm) points up the image, +y (the arm's left) to the left.
    top = out / f"pointcloud_{stamp}_top.png"
    cv2.imwrite(str(top), render_view(r, objects, W, H, LOOK_AT + np.array([0.0005, 0, 1.1]),
                                      40, title + ", from above", scale, up=(1, 0, 0),
                                      map_view=True, missing=missing))
    print(f"Saved {top}")
    if args.video:
        mp4 = out / f"pointcloud_{stamp}.mp4"
        writer = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
        for k in range(args.frames_video):
            az = -35 + 360.0 * k / args.frames_video
            writer.write(render_view(r, objects, W, H, view_eye(az, 42, 1.05), 40, title, scale,
                                     missing=missing))
        writer.release()
        print(f"Saved turntable {mp4}")
    del r


def show_interactive(geom, objects):
    """O3DVisualizer with labels floating over the objects; falls back to plain."""
    import open3d as o3d
    try:
        import open3d.visualization.gui as gui
        from open3d.visualization import rendering
        app = gui.Application.instance
        app.initialize()
        w = o3d.visualization.O3DVisualizer("robo-chem point cloud", 1400, 900)
        w.show_settings = False
        w.set_background([*BACKGROUND, 1.0], None)
        w.add_geometry("scene", geom)
        line = rendering.MaterialRecord()
        line.shader = "unlitLine"
        line.line_width = 3.0
        for i, obj in enumerate(objects):
            w.add_geometry(f"{obj['title']} outline", obj["outline"], line)
            if obj.get("shell") is not None:
                w.add_geometry(f"{obj['title']} shell", shell_mesh(obj), glass_material(obj))
            label = w.add_3d_label(obj["anchor"] + np.array([0, 0, 0.03]), obj["title"])
            try:
                label.color = gui.Color(*obj["colour"])
                label.scale = 1.3
            except Exception:
                pass
        w.point_size = 3
        w.setup_camera(40, LOOK_AT, view_eye(-35, 42, 1.05), [0, 0, 1])
        app.add_window(w)
        print("Interactive view: drag to rotate, scroll to zoom, close the window to quit.")
        app.run()
    except Exception as e:
        print(f"Labelled viewer unavailable ({type(e).__name__}: {e}); plain viewer, no labels")
        geoms = [geom] + [obj["outline"] for obj in objects]
        o3d.visualization.draw_geometries(geoms, window_name="robo-chem point cloud",
                                          width=1400, height=900)


# --------------------------------------------------------------------- main
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--objects", nargs="*", default=None,
                        help="Bench objects to locate (default: every object on the bench)")
    parser.add_argument("--no-objects", action="store_true", help="The cloud only")
    parser.add_argument("--per-camera", action="store_true",
                        help="Colour points by camera, to check calibration")
    parser.add_argument("--cameras", nargs="+", type=int, default=[2, 3, 4, 5])
    parser.add_argument("--box", nargs=6, type=float,
                        default=[0.15, -0.50, -0.03, 0.95, 0.50, 0.35],
                        metavar=("XMIN", "YMIN", "ZMIN", "XMAX", "YMAX", "ZMAX"),
                        help="Crop to this base-frame box, metres")
    parser.add_argument("--frames", type=int, default=8,
                        help="Frames per camera; the per-pixel median is used")
    parser.add_argument("--warmup", type=int, default=15,
                        help="Frames dropped while auto-exposure settles")
    parser.add_argument("--mode", choices=["mesh", "points"], default="mesh",
                        help="mesh: the cameras fused into one surface (TSDF), much "
                             "cleaner shapes; points: the merged raw points")
    parser.add_argument("--tsdf-voxel", type=float, default=0.002,
                        help="TSDF voxel, metres (mesh mode)")
    parser.add_argument("--size", nargs=2, type=int, default=[1280, 720],
                        metavar=("W", "H"),
                        help="Capture resolution; falls back to 848x480 if refused")
    parser.add_argument("--stride", type=int, default=1,
                        help="Use every n-th pixel (1 = all)")
    parser.add_argument("--voxel", type=float, default=0.0015,
                        help="Merge grid, metres (0 = keep every point)")
    parser.add_argument("--no-dim", action="store_true",
                        help="Keep the table and everything unlabelled in full colour")
    parser.add_argument("--point-size", type=float, default=2.5)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1000)
    parser.add_argument("--video", action="store_true", help="Also save a turntable .mp4")
    parser.add_argument("--frames-video", type=int, default=180)
    parser.add_argument("--no-show", action="store_true", help="Do not open the window")
    parser.add_argument("--view-only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--from-raw", default=None, metavar="NPY",
                        help="Re-render a saved capture; no cameras, no grounding")
    parser.add_argument("--out-dir", default=str(ROOT / "diag_out" / "pointcloud"))
    parser.add_argument("--bench", choices=sorted(BENCHES), default="magic_beaker")
    parser.add_argument("--grounding-url",
                        default=os.environ.get("GROUNDING_URL", "http://127.0.0.1:5005"))
    args = parser.parse_args()

    import open3d as o3d
    bench = BENCHES[args.bench]
    table_z = REAL_TABLE_Z
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.from_raw:
        saved = np.load(args.from_raw, allow_pickle=True).item()
        raw, objects = saved["cameras"], saved.get("objects", [])
        missing = list(saved.get("missing", []))
        args.cameras = sorted(raw)
        stamp = Path(args.from_raw).stem.replace("pointcloud_", "").replace("_raw", "")
        print(f"Re-rendering {args.from_raw}: cameras {args.cameras}, {len(objects)} objects")
        if args.no_objects or args.per_camera:
            objects, missing = [], []
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        raw = capture(args.cameras, max(1, args.frames), max(0, args.warmup), args.size)
        if not raw:
            print("No camera delivered a frame.")
            return 1
        objects, missing = [], []
        if not args.no_objects and not args.per_camera:
            names = args.objects if args.objects else [o.name for o in bench.objects]
            objects = locate_objects(names, bench, args.grounding_url, args.cameras)
            found = {o["name"] for o in objects}
            missing = [n for n in names if n not in found]
        raw_path = out / f"pointcloud_{stamp}_raw.npy"
        np.save(raw_path, {"cameras": raw, "objects": objects, "missing": missing},
                allow_pickle=True)
        print(f"Saved {raw_path} (re-render with --from-raw)")

    mesh_mode = args.mode == "mesh" and not args.per_camera
    geom = build_mesh(raw, args, table_z) if mesh_mode else build_scene(raw, args, table_z)
    if geom is None or len(positions(geom)) == 0:
        print("No points.")
        return 1
    if objects:
        print("\nObjects:")
        owner = describe_objects(geom, objects, bench, table_z)
        if not args.per_camera:
            geom = colour_scene(geom, objects, owner, dim=not args.no_dim)

    if args.view_only:
        show_interactive(geom, objects)
        return 0

    ply = out / f"pointcloud_{stamp}.ply"
    if mesh_mode:
        o3d.io.write_triangle_mesh(str(out / f"pointcloud_{stamp}_mesh.ply"), geom)
        # The surface's vertices are an even, clean point cloud of it.
        cloud = o3d.geometry.PointCloud(geom.vertices)
        cloud.colors, cloud.normals = geom.vertex_colors, geom.vertex_normals
        o3d.io.write_point_cloud(str(ply), cloud)
        print(f"Saved {ply} ({len(cloud.points)} points) and the surface as _mesh.ply")
    else:
        o3d.io.write_point_cloud(str(ply), geom)
        print(f"Saved {ply} ({len(geom.points)} points)")
    render_outputs(geom, objects, stamp, out, args, missing=missing)

    if not args.no_show:
        # A fresh process: the offscreen renderer above left Open3D in
        # headless mode, which the window cannot share.
        raw_path = args.from_raw or str(out / f"pointcloud_{stamp}_raw.npy")
        cmd = [sys.executable, str(Path(__file__).resolve()), "--from-raw", raw_path,
               "--view-only", "--voxel", str(args.voxel), "--stride", str(args.stride),
               "--bench", args.bench, "--mode", args.mode, "--tsdf-voxel", str(args.tsdf_voxel)]
        if args.no_dim:
            cmd.append("--no-dim")
        if args.per_camera:
            cmd.append("--per-camera")
        if args.no_objects:
            cmd.append("--no-objects")
        subprocess.run(cmd)
    return 0


if __name__ == "__main__":
    sys.exit(main())
