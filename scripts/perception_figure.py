"""
The perception figure, from one capture: each camera's view with SAM 3's
masks drawn on, the label each camera read on the chosen container, and the
fused cloud with that container and its grasp. Read-only; the robot is never
commanded.

All panels come from the SAME frames. The four cameras are captured once (as
view_pointcloud.py does: per-pixel median depth, 1280x720), and the real
locate (VisionSystem.locate: SAM 3 instances, silhouette placement, label
reads, the vote) runs on those frames instead of capturing its own. The cloud
is fused from the same depth images. The grasp is the one pick_up computes for
this container: the top grasp from the vision, then the bench's pick
parameters (slide-in, forward/lateral offset, z offset, grip width).

Writes diag_out/perception_figure/<time>/:
    cam<N>_masks.png        the camera image, every candidate faint, the chosen
                            one filled in its colour, each tagged with its read
    cam<N>_masks_plain.png  the same without the text tags
    cloud.png, cloud_close.png        fused surface, container, grasp, labels
    cloud_clean.png, cloud_close_clean.png   the same without labels/legend
    votes.txt, votes.json   panel (b): what each camera read on the chosen one
    capture.npy             everything above's inputs; --from-saved re-draws
                            without cameras, grounding or the label model

Usage:
    source scripts/env.sh
    scripts/grounding.sh start
    python scripts/perception_figure.py --object "clear cup a"
    python scripts/perception_figure.py --from-saved diag_out/perception_figure/<time>/capture.npy
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import view_pointcloud as vp  # noqa: E402
from robochem.vision import REAL_TABLE_Z  # noqa: E402
from robochem.vision.bench_manifest import BENCHES  # noqa: E402

CHOSEN = (0.12, 0.62, 1.00)       # the container, in every panel (vp.PALETTE[0])
GRASP = (1.00, 0.55, 0.05)        # the gripper and its slide-in
OTHER = (0.85, 0.85, 0.85)        # every other candidate SAM 3 found
JAW_OPEN = 0.080                  # the Panda's jaws fully open, as they slide in


# ----------------------------------------------------------------- capture
def intrinsics_object(cam_id, rec):
    """The pipeline's intrinsics type, from view_pointcloud's dict."""
    from perception import CameraIntrinsics
    h, w = rec["colour"].shape[:2]
    k = rec["intr"]
    return CameraIntrinsics(frame=str(cam_id), fx=k["fx"], fy=k["fy"], cx=k["cx"],
                            cy=k["cy"], height=h, width=w)


def locate_on(raw, name, bench, grounding_url):
    """
    VisionSystem.locate(name) on the frames in ``raw``, recording SAM 3's
    instances and the label reads it voted with.
    """
    from robochem.vision import VisionSystem
    vision = VisionSystem(cameras={}, grounding_url=grounding_url, bench=bench)
    cams = sorted(raw)
    vision.object_localizer.camera_ids = cams
    vision.clear_cache()
    data = {"pointclouds": {c: None for c in cams},
            "images": {c: raw[c]["colour"] for c in cams},
            "depth_images": {c: raw[c]["depth"] for c in cams},
            "intrinsics": {c: intrinsics_object(c, raw[c]) for c in cams}}

    def capture_pointclouds(warmup=8, cam_ids=None):
        if cam_ids is None:
            vision.object_localizer.last_capture = data
            return data
        return {k: {c: v[c] for c in cam_ids if c in v} for k, v in data.items()}
    vision.object_localizer.capture_pointclouds = capture_pointclouds

    seen = {"instances": [], "reads": None}
    grounding = vision.scene_analyzer.grounding
    segment = grounding.segment_instances

    def segment_instances(images, prompt, *a, **kw):
        out = segment(images, prompt, *a, **kw)
        seen["instances"].append((prompt, out))
        return out
    grounding.segment_instances = segment_instances
    read_all = vision.label_resolver.read_all

    def read_all_recorded(images, instances):
        out = read_all(images, instances)
        seen["reads"] = out
        return out
    vision.label_resolver.read_all = read_all_recorded

    located = vision.locate(name)
    if located is None:
        return None, vision
    category = located.get("category")
    instances = next((out for prompt, out in reversed(seen["instances"])
                      if category is None or prompt == category), {})
    located = dict(located)
    located["instances"] = {c: [{"mask": np.asarray(i["mask"], bool),
                                 "score": float(i.get("score") or 0.0)} for i in insts]
                            for c, insts in (instances or {}).items()}
    located["reads"] = seen["reads"] or {}
    return located, vision


def grasp_of(vision, located, entry):
    """
    The grasp pick_up would command for this container: the vision's top
    grasp, then the bench's pick parameters, as in PickUpSkill.execute.
    """
    pose = vision.compute_grasp_pose(located["points"], entry.name, grasp_type="top",
                                     pitch_deg=0.0)
    if pose is None:
        return None
    pick = dict(entry.pick or ())
    pose = np.asarray(pose, dtype=float).copy()
    slide = float(pick.get("slide_in") or 0.0)
    if slide > 0:
        pose[:3, :3] = np.diag([1.0, -1.0, -1.0])
    pose[2, 3] += float(pick.get("z_offset") or 0.0)
    pose[0, 3] += float(pick.get("forward_offset") or 0.0)
    pose[1, 3] += float(pick.get("lateral_offset") or 0.0)
    return {"pose": pose, "slide_in": slide,
            "grip_width": float(pick.get("grip_width") or 0.06)}


# ------------------------------------------------------------ mask panels
def project_points(rec, world):
    """Pixel coordinates of world points in this camera (T is camera->world)."""
    T, k = rec["T"], rec["intr"]
    cam = (np.asarray(world) - T[:3, 3]) @ T[:3, :3]
    z = cam[:, 2]
    ok = z > 0.05
    u = k["fx"] * cam[:, 0] / np.where(ok, z, 1) + k["cx"]
    v = k["fy"] * cam[:, 1] / np.where(ok, z, 1) + k["cy"]
    return np.column_stack([u, v])[ok]


def fit_mask(mask, shape):
    import cv2
    if mask.shape == shape:
        return mask
    return cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]),
                      interpolation=cv2.INTER_NEAREST).astype(bool)


def chosen_instance(cam, located, raw):
    """Index of the chosen container's mask in this camera, or None."""
    insts = located["instances"].get(cam) or []
    if cam not in (located.get("label_matches") or {}) and cam not in located.get("cameras", []):
        return None
    score = (located.get("confidences") or {}).get(cam)
    if score is not None:
        same = [i for i, inst in enumerate(insts) if abs(inst["score"] - score) < 1e-9]
        if len(same) == 1:
            return same[0]
    # The container's model projected in: the mask it falls in most.
    uv = np.round(project_points(raw[cam], located["points"])).astype(int)
    best, best_hits = None, 0
    for i, inst in enumerate(insts):
        m = fit_mask(inst["mask"], raw[cam]["colour"].shape[:2])
        h, w = m.shape
        ok = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
        hits = int(m[uv[ok, 1], uv[ok, 0]].sum())
        if hits > best_hits:
            best, best_hits = i, hits
    return best if best_hits >= 0.2 * len(uv) else None


def draw_masks(img, insts, reads, chosen, tags, scale):
    import cv2
    out = img.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs, th = 0.5 * scale, max(1, int(round(1.3 * scale)))
    order = [i for i in range(len(insts)) if i != chosen] + ([chosen] if chosen is not None else [])
    for i in order:
        m = fit_mask(insts[i]["mask"], img.shape[:2])
        colour = CHOSEN if i == chosen else OTHER
        bgr = np.array(colour[::-1]) * 255
        alpha = 0.45 if i == chosen else 0.22
        out[m] = (1 - alpha) * out[m] + alpha * bgr
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, contours, -1, tuple(int(c) for c in bgr),
                         max(1, int(round((3 if i == chosen else 1.5) * scale))), cv2.LINE_AA)
    if tags:
        for i in order:
            m = fit_mask(insts[i]["mask"], img.shape[:2])
            ys, xs = np.nonzero(m)
            if not len(xs):
                continue
            read = reads[i] if i < len(reads) else None
            if not read and i != chosen:
                continue
            text = (read or "unread").upper()
            colour = CHOSEN if i == chosen else OTHER
            (tw, tht), base = cv2.getTextSize(text, font, fs, th)
            pad = int(4 * scale)
            x = int(np.clip(xs.mean() - tw / 2 - pad, 2, img.shape[1] - tw - 2 * pad - 2))
            y = int(np.clip(ys.min() - tht - base - 2 * pad - 4, 2, img.shape[0] - tht - 2 * pad))
            bgr = tuple(int(255 * c) for c in colour[::-1])
            cv2.rectangle(out, (x, y), (x + tw + 2 * pad, y + tht + base + 2 * pad), bgr, -1,
                          cv2.LINE_AA)
            ink = (20, 20, 20) if i != chosen else (255, 255, 255)
            cv2.putText(out, text, (x + pad, y + pad + tht), font, fs, ink, th, cv2.LINE_AA)
    return out


# ------------------------------------------------------------- cloud panel
def box_mesh(centre, size, colour):
    import open3d as o3d
    b = o3d.geometry.TriangleMesh.create_box(*size)
    b.translate(np.asarray(centre) - np.asarray(size) / 2)
    b.compute_vertex_normals()
    b.paint_uniform_color(colour)
    return b


def gripper_meshes(grasp):
    """
    Two fingers either side of the grasp point, open as the hand slides in
    (it closes to the grip width only once there), the bar joining them, and
    each finger's slide-in path. Fingers close along base Y, tips at the TCP.
    """
    import open3d as o3d
    p = grasp["pose"][:3, 3]
    half = JAW_OPEN / 2.0
    finger = (0.014, 0.006, 0.040)
    parts = []
    for s in (-1.0, 1.0):
        parts.append(box_mesh(p + [0, s * (half + finger[1] / 2), finger[2] / 2 - 0.004],
                              finger, GRASP))
    parts.append(box_mesh(p + [0, 0, finger[2] + 0.001],
                          (0.014, 2 * half + 2 * finger[1], 0.008), GRASP))
    parts.append(box_mesh(p + [0, 0, finger[2] + 0.017], (0.014, 0.014, 0.025), GRASP))
    if grasp["slide_in"] > 0:
        # Each finger's path: from behind (toward the base), along either
        # side of the container, to where it closes.
        length = grasp["slide_in"] - 0.010
        R = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])  # +Z -> +X
        for s in (-1.0, 1.0):
            arrow = o3d.geometry.TriangleMesh.create_arrow(
                cylinder_radius=0.0022, cone_radius=0.005,
                cylinder_height=length - 0.012, cone_height=0.012)
            arrow.rotate(R, center=(0, 0, 0))
            arrow.translate(p + [-grasp["slide_in"], s * (half + finger[1] / 2),
                                 finger[2] / 2])
            arrow.compute_vertex_normals()
            arrow.paint_uniform_color(GRASP)
            parts.append(arrow)
    return parts


def render_cloud(geom, objects, grasp, out, stamp, args):
    import cv2
    from open3d.visualization import rendering
    W, H = args.width, args.height
    scale = W / 1280.0
    r = vp.make_renderer(geom, objects, W, H, 2.5)
    solid = rendering.MaterialRecord()
    solid.shader = "defaultLit"
    tags = list(objects)
    if grasp is not None:
        for k, m in enumerate(gripper_meshes(grasp)):
            r.scene.add_geometry(f"gripper{k}", m, solid)
        tags.append({"anchor": grasp["pose"][:3, 3] + np.array([0, 0, 0.065]),
                     "title": (f"grasp: slides in {grasp['slide_in'] * 1000:.0f} mm, "
                               f"closes to {grasp['grip_width'] * 1000:.0f} mm"
                               if grasp["slide_in"] > 0 else "grasp"),
                     "colour": GRASP})
    centre = objects[0]["anchor"] if objects else vp.LOOK_AT
    views = {"cloud": (vp.view_eye(-35, 42, 1.05), vp.LOOK_AT),
             # Side-on enough that the slide-in toward +X reads as a motion.
             "cloud_close": (vp.view_eye(-65, 35, 0.42, look_at=centre), centre)}
    title = f"fused cloud, cameras {','.join(map(str, args.cameras))}, {stamp}"
    for name, (eye, look) in views.items():
        r.setup_camera(40, look, eye, [0, 0, 1])
        clean = cv2.cvtColor(np.asarray(r.render_to_image()), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out / f"{name}_clean.png"), clean)
        img = clean.copy()
        legend = vp.draw_legend(img, objects, title, scale) if objects else None
        vp.draw_labels(img, tags, r, W, H, scale, avoid=[legend] if legend else [])
        cv2.imwrite(str(out / f"{name}.png"), img)
        print(f"Saved {out / name}.png and {name}_clean.png")
    del r


# -------------------------------------------------------------------- main
def votes_of(located, entry, cams):
    want = (entry.label or entry.name).upper()
    matches = located.get("label_matches") or {}
    rows = []
    for cam in cams:
        if cam not in matches:
            rows.append({"cam": cam, "read": None, "ok": None, "note": "not placed by this camera"})
            continue
        read = (matches[cam] or "").upper() or None
        rows.append({"cam": cam, "read": read, "ok": bool(read) and read == want})
    return {"wanted": want, "rows": rows, "resolved_by": located.get("resolved_by"),
            "localized_by": located.get("localized_by"),
            "position": [round(float(v), 4) for v in
                         (located.get("silhouette_point") or located["centroid"])]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--object", default="clear cup a", help="Bench object to locate")
    parser.add_argument("--bench", choices=sorted(BENCHES), default="magic_beaker")
    parser.add_argument("--cameras", nargs="+", type=int, default=[2, 3, 4, 5])
    parser.add_argument("--size", nargs=2, type=int, default=[1280, 720], metavar=("W", "H"))
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--box", nargs=6, type=float,
                        default=[0.15, -0.50, -0.03, 0.95, 0.50, 0.35])
    parser.add_argument("--tsdf-voxel", type=float, default=0.002)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1000)
    parser.add_argument("--from-saved", metavar="NPY",
                        help="Re-draw a saved capture.npy; no cameras, grounding or labels")
    parser.add_argument("--from-capture", metavar="NPY",
                        help="Locate on a view_pointcloud.py _raw.npy instead of the cameras")
    parser.add_argument("--out-dir", default=str(ROOT / "diag_out" / "perception_figure"))
    parser.add_argument("--grounding-url",
                        default=os.environ.get("GROUNDING_URL", "http://127.0.0.1:5005"))
    args = parser.parse_args()

    import cv2
    bench = BENCHES[args.bench]
    if args.from_saved:
        saved = np.load(args.from_saved, allow_pickle=True).item()
        raw, located, grasp, name = (saved["raw"], saved["located"], saved["grasp"],
                                     saved["object"])
        out = Path(args.from_saved).parent
        stamp = out.name
    else:
        name = args.object
        if args.from_capture:
            raw = np.load(args.from_capture, allow_pickle=True).item()["cameras"]
            stamp = Path(args.from_capture).stem.replace("pointcloud_", "").replace("_raw", "")
        else:
            stamp = time.strftime("%Y%m%d_%H%M%S")
            raw = vp.capture(args.cameras, max(1, args.frames), max(0, args.warmup), args.size)
        out = Path(args.out_dir) / stamp
        out.mkdir(parents=True, exist_ok=True)
        if not raw:
            print("No camera delivered a frame.")
            return 1
        located, vision = locate_on(raw, name, bench, args.grounding_url)
        if located is None:
            print(f"'{name}' was not located on these frames; nothing drawn. "
                  f"If the 1280x720 frames are the problem, try --size 848 480.")
            np.save(out / "capture.npy", {"raw": raw, "located": None, "grasp": None,
                                          "object": name}, allow_pickle=True)
            return 1
        entry = bench.find(name)
        grasp = grasp_of(vision, located, entry) if entry is not None else None
        keep = {k: v for k, v in located.items() if k != "dimensions"}
        np.save(out / "capture.npy", {"raw": raw, "located": keep, "grasp": grasp,
                                      "object": name}, allow_pickle=True)
        print(f"Saved {out / 'capture.npy'} (re-draw with --from-saved)")

    entry = bench.find(name)
    cams = sorted(raw)
    args.cameras = cams

    # (a) the four views
    for cam in cams:
        img = raw[cam]["colour"]
        insts = located["instances"].get(cam) or []
        reads = (located.get("reads") or {}).get(cam) or []
        k = chosen_instance(cam, located, raw)
        scale = img.shape[1] / 848.0
        for tags, suffix in ((True, ""), (False, "_plain")):
            cv2.imwrite(str(out / f"cam{cam}_masks{suffix}.png"),
                        draw_masks(img, insts, reads, k, tags, scale))
        print(f"cam {cam}: {len(insts)} instance(s), chosen "
              f"{'#' + str(k) if k is not None else 'none'}")

    # (b) the reads and the vote
    v = votes_of(located, entry, cams)
    lines = [f"Wanted: {v['wanted']}"]
    for row in v["rows"]:
        mark = {True: "ok", False: "x", None: "-"}[row["ok"]]
        lines.append(f"cam {row['cam']}  {row['read'] or row.get('note') or 'unread'}  {mark}")
    lines += [f"vote: {v['resolved_by']}", f"placed: {v['localized_by']}",
              f"position: {v['position']}"]
    if grasp is not None:
        g = grasp["pose"][:3, 3]
        lines.append(f"grasp: ({g[0]:.4f}, {g[1]:+.4f}, {g[2]:.4f}), slide-in "
                     f"{grasp['slide_in'] * 1000:.0f} mm, grip {grasp['grip_width'] * 1000:.0f} mm")
    (out / "votes.txt").write_text("\n".join(lines) + "\n")
    (out / "votes.json").write_text(json.dumps(
        {**v, "grasp": None if grasp is None else
         {"xyz": [float(x) for x in grasp["pose"][:3, 3]], "slide_in": grasp["slide_in"],
          "grip_width": grasp["grip_width"]}}, indent=2))
    print("\n".join(lines))

    # (c) the fused cloud, the container and its grasp
    geom = vp.build_mesh(raw, args, REAL_TABLE_Z)
    objects = [{"name": name, "points": np.asarray(located["points"], dtype=float),
                "centroid": np.asarray(located["centroid"], dtype=float)}]
    owner = vp.describe_objects(geom, objects, bench, REAL_TABLE_Z)
    objects[0]["colour"] = CHOSEN
    objects[0]["outline"].paint_uniform_color(0.6 * np.array(CHOSEN) + 0.4)
    geom = vp.colour_scene(geom, objects, owner, dim=True)
    render_cloud(geom, objects, grasp, out, stamp, args)
    print(f"\nEverything in {out}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
