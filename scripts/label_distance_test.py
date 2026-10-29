"""
How far can a container stand from its label and still be found by that label?

Cameras and the grounding service only: this script never imports frankapy
and never connects to the arm. It exits if anything did.

Same bench, manifest and memory-free locate as ``label_reading_benchmark.py``:
the four containers in a square, the label reader choosing among their four
labels, ``VisionSystem.locate`` exactly as a skill calls it, elimination
included. Nothing in the system is switched off.

One container, x (``--object``), is moved by hand to a new spot each trial;
its label paper stays where it was. Each x is tested in two phases:

- ``labelled``: the other three labels stay on the bench. A run can then also
  find x by elimination: no camera read x's label, but the other container
  of its category (the red cabbage dish for the baking soda dish, the oil
  beaker for the water cup and the other way round) was read as its own.
- ``unlabelled``: the other three labels are removed, their containers stay.
  x's partner has no label to be read by, so elimination cannot happen and
  x is found only if a camera reads x's label on it.

Taking the papers away moves the containers, so every phase starts with its
own ground truth, x on its own label:

1. Ground truth: every container placed with its labels not read, named by
   quadrant, located again held to that spot (as in the label benchmark).
   Then x is located by its label ``--checks`` times, on its own label, to
   show the system works when nothing is out of place.
2. ``--trials`` (5) trials. Move x, measure by hand how far it is from where
   it stood on its label, type that distance in cm and press Enter. x is
   then located by its label ``--locates`` times.

The distance is the hand measurement only; the cameras do not measure it
(depth does not see the clear cups). Whether a locate found x needs no
measurement either: only x moves within a phase, and a locate returns one of
the containers of x's category, so a locate that lands within ``--match-tol``
of x's partner's ground-truth spot is "wrong" and any other is x. Keep
x more than ``--match-tol`` from its partner. x is "by label" when at least
one camera read x's label on it, "by elimination" when none did; "not found"
when the locate returns nothing.

All six phases, one at a time (ground truth, checks and trials in one go):

    python scripts/label_distance_test.py run --object "baking soda cup" --phase labelled
    python scripts/label_distance_test.py run --object "baking soda cup" --phase unlabelled
    python scripts/label_distance_test.py run --object "10 ml water cup" --phase labelled
    python scripts/label_distance_test.py run --object "10 ml water cup" --phase unlabelled
    python scripts/label_distance_test.py run --object "beaker" --phase labelled
    python scripts/label_distance_test.py run --object "beaker" --phase unlabelled

The steps also run on their own (``ground-truth``, then ``trials``). Calling
``trials`` again adds spots to the phase, numbered on from its last trial. The table of every phase, also
printed at the end of each run:

    python scripts/label_distance_test.py report

Everything goes to ``--out`` (default experiments/label_distance_<date>):
ground_truth_<x>_<phase>.json, ground_truth_frames_<x>_<phase>/,
checks.jsonl (the on-label locates), trials.jsonl (one line per locate),
trials/<x>_<phase>/NN/ (each trial's frames with x's label spot and each
locate drawn on), report.md and summary.csv.

3D view of each trial, in the trial's folder (and in ground_truth_frames_*
for the on-label checks): the last locate's capture fused into one surface
(view_pointcloud.py's TSDF), x as the skills get it (a clear cup or the
beaker as its model shell, which depth cannot see, placed by silhouettes),
a red dot on every locate's centre (the ``centroid`` the skills are handed)
and a green dot on x's label spot. cloud.png (close-up), cloud_top.png
(the whole bench from above), cloud_mesh.ply, and cloud_raw.npy, which
re-renders or opens in a window without the cameras:

    python scripts/label_distance_test.py view-cloud --object beaker \\
        --npy experiments/label_distance_<date>/trials/vegetable_oil_labelled/01/cloud_raw.npy

Rendered in a separate process after each trial; ``--no-cloud-render``
keeps only cloud_raw.npy (render later with ``render-cloud --npy ...``).
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from label_reading_benchmark import (LABEL_TEST, fresh, ground_truth, make_vision, no_arm,
                                     save_frames, truth_path, xyz)

#: The containers that are moved, one per kind: dish, clear cup, beaker.
OBJECTS = ("baking soda cup", "10 ml water cup", "beaker")
PHASES = ("labelled", "unlabelled")
OUTCOMES = ("by label", "by elimination", "wrong", "not found")
GREEN, RED = (0, 255, 0), (0, 0, 255)


def set_name(obj, phase):
    return f"{LABEL_TEST.find(obj).label.replace(' ', '_')}_{phase}"


def load_truth(args):
    path = truth_path(args)
    if not path.exists():
        sys.exit(f"No {path} yet: run the ground-truth step for this phase first.")
    return json.loads(path.read_text())["objects"]


def recorder(vision):
    """Wrap the label reader so every raw read of the last locate is kept."""
    reads = {}
    real_read_all = vision.label_resolver.read_all

    def recording_read_all(images, instances):
        got = real_read_all(images, instances)
        reads.clear()
        reads.update({str(c): list(v) for c, v in got.items()})
        return got

    vision.label_resolver.read_all = recording_read_all
    return reads


def _same(read, label):
    from robochem.vision.label_resolver import normalize_label
    return normalize_label(read) == normalize_label(label)


def by_label(args, vision, reads, truth):
    """One locate of ``args.object`` by its label. Wrong when it lands on x's partner."""
    label = truth[args.object]["label"]
    fresh(vision)
    reads.clear()
    started = time.time()
    try:
        loc, error = vision.locate(args.object, force_refresh=True), None
    except Exception as exc:
        loc, error = None, f"{type(exc).__name__}: {exc}"
        print(f"[distance] {args.object}: {error}")
    row = {"found": False, "outcome": "not found", "error": error,
           "seconds": round(time.time() - started, 1),
           "all_reads": dict(reads),
           "label_reads_anywhere": sum(1 for v in reads.values()
                                       for r in v if r and _same(r, label))}
    if loc is None:
        if not reads:
            # No camera's labels came back at all: the reader could not be
            # reached, not a container it could not read. On 2026-10-29 every
            # call failed with "Connection error." and three on-label checks of
            # the beaker were logged as not found.
            row["outcome"] = "no reads"
            print("[distance] NO LABEL READS CAME BACK (API or network?): this locate is "
                  "left out of the scores")
        return row, None, None
    p = np.asarray(xyz(loc))
    matches = {str(c): v for c, v in (loc.get("label_matches") or {}).items()}
    on_it = sum(1 for r in matches.values() if r and _same(r, label))
    # A locate returns a container of x's category, so only those can be mistaken for x.
    category = LABEL_TEST.find(args.object).category
    others = {n: t["xyz"] for n, t in truth.items()
              if n != args.object and LABEL_TEST.find(n).category == category}
    nearest = min(others, key=lambda n: np.linalg.norm(p[:2] - np.asarray(others[n])[:2]))
    row.update({"predicted": p.tolist(),
                "cameras": [int(c) for c in loc.get("cameras") or []],
                "label_matches": matches, "label_reads_on_it": on_it,
                "resolved_by": loc.get("resolved_by")})
    if np.linalg.norm(p[:2] - np.asarray(others[nearest])[:2]) <= args.match_tol:
        row["outcome"], row["wrong_is"] = "wrong", nearest
    else:
        row["found"] = True
        row["outcome"] = "by label" if on_it else "by elimination"
    return row, p, loc


# ------------------------------------------------------------------ 3D view
#
# The scene the locate saw, fused in 3D by view_pointcloud.py's TSDF, with x
# drawn as the skills get it (a clear cup is its model shell: depth does not
# see clear plastic) and a dot on each predicted centre and on x's label spot.

CLOUD_BOX = [0.15, -0.50, -0.03, 0.95, 0.50, 0.35]
SPOT_RGB, LOCATE_RGB = (0.2, 0.9, 0.3), (1.0, 0.2, 0.2)


def save_cloud(args, vision, folder, spot, located):
    """Keep the last capture, x's model and its centres, then render them."""
    import subprocess
    from view_pointcloud import intr_dict

    data = vision.object_localizer.last_capture or {}
    cams = {}
    for cam, image in (data.get("images") or {}).items():
        depth = (data.get("depth_images") or {}).get(cam)
        intr = (data.get("intrinsics") or {}).get(cam)
        if depth is None or intr is None:
            continue
        extr = vision.object_localizer._extrinsics(cam)
        T = extr if isinstance(extr, np.ndarray) else extr.matrix()
        cams[int(cam)] = {"depth": np.asarray(depth).astype(np.uint16),
                          "colour": np.asarray(image).copy(), "intr": intr_dict(intr),
                          "T": np.asarray(T, dtype=float)}
    if not cams:
        print("[distance] no capture to build a 3D view from")
        return
    objects, markers = [], [{"name": "label spot", "xyz": np.asarray(spot, float)[:3],
                             "colour": SPOT_RGB}]
    for name, loc in located:
        if not objects:
            objects.append({"name": args.object, "points": np.asarray(loc["points"], np.float32),
                            "centroid": np.asarray(loc["centroid"], float)})
        markers.append({"name": name, "xyz": np.asarray(loc["centroid"], float)[:3],
                        "colour": LOCATE_RGB})
    folder.mkdir(parents=True, exist_ok=True)
    raw = folder / "cloud_raw.npy"
    # Also readable by view_pointcloud.py --from-raw (it ignores the markers).
    np.save(raw, {"cameras": cams, "objects": objects, "markers": markers, "missing": []},
            allow_pickle=True)
    if args.no_cloud_render:
        return
    # A fresh process: Open3D's offscreen renderer does not survive being made
    # again and again in one, and a failed render must not stop the trials.
    try:
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "render-cloud",
                        "--npy", str(raw), "--object", args.object],
                       timeout=300, check=True)
    except Exception as exc:
        print(f"[distance] 3D view not rendered ({type(exc).__name__}: {exc}); the capture "
              f"is kept in {raw}: render-cloud --npy {raw}")


def cloud_scene(saved, obj_name):
    """The fused surface with x coloured and outlined, x's objects and the markers."""
    from types import SimpleNamespace
    import view_pointcloud as vp
    from robochem.vision import REAL_TABLE_Z

    geom = vp.build_mesh(saved["cameras"], SimpleNamespace(tsdf_voxel=0.002, box=CLOUD_BOX),
                         REAL_TABLE_Z)
    objects = list(saved.get("objects") or [])
    if objects:
        owner = vp.describe_objects(geom, objects, LABEL_TEST, REAL_TABLE_Z)
        geom = vp.colour_scene(geom, objects, owner, dim=True)
    markers = list(saved.get("markers") or [])
    for m in markers:
        print(f"  {m['name']:12s} ({m['xyz'][0]:.4f}, {m['xyz'][1]:+.4f}, {m['xyz'][2]:.4f})")
    return geom, objects, markers


def marker_geometry(m):
    """A dot on the centre and a stem from the table up through it, so it shows in a cup."""
    import open3d as o3d
    from robochem.vision import REAL_TABLE_Z
    x, y, z = m["xyz"]
    dot = o3d.geometry.TriangleMesh.create_sphere(radius=0.006)
    dot.translate([x, y, z])
    dot.compute_vertex_normals()
    dot.paint_uniform_color(m["colour"])
    stem = o3d.geometry.LineSet(
        o3d.utility.Vector3dVector([[x, y, REAL_TABLE_Z], [x, y, z + 0.08]]),
        o3d.utility.Vector2iVector([[0, 1]]))
    stem.paint_uniform_color(m["colour"])
    return dot, stem


def marker_text(markers):
    """One tag per spot: locates that landed together share it."""
    tags = []
    for m in markers:
        same = next((t for t in tags if np.linalg.norm(t["xyz"] - m["xyz"]) < 0.002), None)
        if same is not None:
            same["names"].append(m["name"])
        else:
            tags.append({"xyz": m["xyz"], "colour": m["colour"], "names": [m["name"]]})
    for t in tags:
        x, y, z = t["xyz"]
        t["text"] = f"{', '.join(t['names'])} ({x:.3f}, {y:+.3f}, {z:.3f})"
    return tags


def render_cloud(args):
    import cv2
    import open3d as o3d
    from open3d.visualization import rendering
    import view_pointcloud as vp

    path = Path(args.npy)
    saved = np.load(path, allow_pickle=True).item()
    geom, objects, markers = cloud_scene(saved, args.object)
    o3d.io.write_triangle_mesh(str(path.parent / "cloud_mesh.ply"), geom)

    W, H = 1600, 1000
    scale = W / 1280.0
    r = vp.make_renderer(geom, objects, W, H, 2.5)
    unlit, line = rendering.MaterialRecord(), rendering.MaterialRecord()
    unlit.shader = "defaultUnlit"
    line.shader, line.line_width = "unlitLine", 4.0
    for i, m in enumerate(markers):
        dot, stem = marker_geometry(m)
        r.scene.add_geometry(f"dot{i}", dot, unlit)
        r.scene.add_geometry(f"stem{i}", stem, line)
    tags = marker_text(markers)
    found = [m["xyz"] for m in markers if m["name"] != "label spot"]
    centre = np.asarray(found[0] if found else markers[0]["xyz"], float)
    title = f"{args.object}: {path.parent.name}"
    views = [("cloud.png", vp.view_eye(-35, 42, 0.55, look_at=centre), centre, (0, 0, 1), False),
             ("cloud_top.png", vp.LOOK_AT + np.array([0.0005, 0, 1.1]), vp.LOOK_AT, (1, 0, 0),
              True)]
    font = cv2.FONT_HERSHEY_SIMPLEX
    for name, eye, look, up, top in views:
        img = vp.render_view(r, objects, W, H, eye, 40, title, scale, look_at=look, up=up,
                             map_view=top)
        # Drawn over the render, so a dot inside a frosted cup still shows.
        for k, t in enumerate(tags):
            p = vp.project(r, W, H, t["xyz"])
            if p is None:
                continue
            col = tuple(int(255 * c) for c in t["colour"][::-1])
            u, v = int(p[0]), int(p[1])
            cv2.circle(img, (u, v), int(7 * scale), (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(img, (u, v), int(5 * scale), col, -1, cv2.LINE_AA)
            ty = v + int((22 + 24 * k) * scale)
            cv2.putText(img, t["text"], (u + int(10 * scale), ty), font, 0.55 * scale,
                        (0, 0, 0), int(4 * scale), cv2.LINE_AA)
            cv2.putText(img, t["text"], (u + int(10 * scale), ty), font, 0.55 * scale, col,
                        max(1, int(1.4 * scale)), cv2.LINE_AA)
        cv2.imwrite(str(path.parent / name), img)
        print(f"Saved {path.parent / name}")
    del r
    return 0


def view_cloud(args):
    """The same scene in a window: drag to rotate, scroll to zoom."""
    import open3d as o3d
    import open3d.visualization.gui as gui
    from open3d.visualization import rendering
    import view_pointcloud as vp

    saved = np.load(Path(args.npy), allow_pickle=True).item()
    geom, objects, markers = cloud_scene(saved, args.object)
    app = gui.Application.instance
    app.initialize()
    w = o3d.visualization.O3DVisualizer(f"{args.object}: {Path(args.npy).parent.name}",
                                        1400, 900)
    w.show_settings = False
    w.set_background([*vp.BACKGROUND, 1.0], None)
    w.add_geometry("scene", geom)
    unlit, line = rendering.MaterialRecord(), rendering.MaterialRecord()
    unlit.shader = "defaultUnlit"
    line.shader, line.line_width = "unlitLine", 3.0
    for i, obj in enumerate(objects):
        w.add_geometry(f"{obj['title']} outline", obj["outline"], line)
        if obj.get("shell") is not None:
            w.add_geometry(f"{obj['title']} shell", vp.shell_mesh(obj), vp.glass_material(obj))
    for i, m in enumerate(markers):
        dot, stem = marker_geometry(m)
        w.add_geometry(f"{m['name']} dot {i}", dot, unlit)
        w.add_geometry(f"{m['name']} stem {i}", stem, line)
    for t in marker_text(markers):
        label = w.add_3d_label(t["xyz"] + np.array([0, 0, 0.09]), t["text"])
        try:
            label.color = gui.Color(*t["colour"])
        except Exception:
            pass
    found = [m["xyz"] for m in markers if m["name"] != "label spot"]
    centre = np.asarray(found[0] if found else markers[0]["xyz"], float)
    w.point_size = 3
    w.setup_camera(40, centre, vp.view_eye(-35, 42, 0.55, look_at=centre), [0, 0, 1])
    app.add_window(w)
    print("Drag to rotate, scroll to zoom, close the window to quit.")
    app.run()
    return 0


# --------------------------------------------------------------- ground truth

def take_truth(args, vision=None):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    vision = vision or make_vision(args)
    truth = ground_truth(args, vision)
    reads = recorder(vision)
    spot = truth[args.object]["xyz"]

    print(f"\n########## {args.object} on its own label ({args.phase}), located by label "
          f"{args.checks} times ##########", flush=True)
    good, unread, located = 0, 0, []
    with (out / "checks.jsonl").open("a") as f:
        for i in range(1, args.checks + 1):
            print(f"\n--- check {i}/{args.checks} ---", flush=True)
            row, p, loc = by_label(args, vision, reads, truth)
            if loc is not None:
                located.append((f"check {i}", loc))
            row.update({"set": args.set, "phase": args.phase, "check": i,
                        "object": args.object, "truth": spot,
                        "time": datetime.now().isoformat(timespec="seconds")})
            if p is not None:
                row["gap_mm"] = round(float(np.linalg.norm(p[:2] - np.asarray(spot)[:2]))
                                      * 1000, 1)
            good += row["outcome"] == "by label"
            unread += row["outcome"] == "no reads"
            print(f"[distance] check {i}: {row['outcome']}"
                  + (f", {row['gap_mm']} mm from the ground truth, "
                     f"{row['label_reads_on_it']} camera(s) read the label"
                     if p is not None else ""))
            f.write(json.dumps(row) + "\n")
    no_arm()
    save_cloud(args, vision, out / f"ground_truth_frames_{args.set}", spot, located)
    print(f"\n[distance] {good}/{args.checks} on-label locates found {args.object} by its "
          f"label.")
    if unread:
        print(f"[distance] {unread} check(s) got no label reads at all: the reader could not be "
              f"reached. Check the network and run ground-truth again.")
    elif good < args.checks:
        print("[distance] The system misses this container even on its own label: fix "
              "that before reading anything into the trials.")
    return vision


# --------------------------------------------------------------------- trials

def ask_distance(prompt):
    """The hand-measured distance in cm, or None to stop."""
    while True:
        answer = input(prompt).strip()
        if answer.lower() == "q":
            return None
        try:
            return float(answer)
        except ValueError:
            print(f"[distance] {answer!r} is not a number: type the distance in cm")


def trials(args, vision=None):
    out = Path(args.out)
    truth = load_truth(args)
    spot = np.asarray(truth[args.object]["xyz"])
    vision = vision or make_vision(args)
    reads = recorder(vision)
    log = out / "trials.jsonl"
    label = truth[args.object]["label"]

    print(f"\n{args.object} ({label!r}, {args.phase}): leave its label paper and every "
          f"other container where they are; keep it more than "
          f"{args.match_tol * 100:.0f} cm from the other {LABEL_TEST.find(args.object).category}.")
    if args.first is None:
        # Carry on from this phase's last trial: a second `trials` call started
        # at 1 again and overwrote trial 1's frames (2026-10-29).
        done = [r["trial"] for r in read_jsonl(log) if r.get("set") == args.set]
        args.first = max(done, default=0) + 1
    for t in range(args.first, args.first + args.trials):
        dist = ask_distance(f"\n>>> {args.set} trial {t}: move {args.object} to a new spot, "
                            f"type how far it moved from its spot on the label in cm, and "
                            f"press Enter (q to stop): ")
        if dist is None:
            break
        base = {"set": args.set, "phase": args.phase, "trial": t, "object": args.object,
                "label": label, "truth": spot.tolist(), "distance_cm": dist}
        marks, located = [("label", spot, GREEN)], []
        for j in range(1, args.locates + 1):
            print(f"\n========== {args.set} trial {t}, locate {j}/{args.locates}: {label} "
                  f"({dist:g} cm off its label) ==========", flush=True)
            row, p, loc = by_label(args, vision, reads, truth)
            row.update({**base, "locate": j,
                        "time": datetime.now().isoformat(timespec="seconds")})
            if p is not None:
                marks.append((f"locate {j}", p, RED))
                located.append((f"locate {j}", loc))
            print(f"[distance] trial {t}, locate {j}: {row['outcome'].upper()}"
                  + (f" (that is {row['wrong_is']})" if row["outcome"] == "wrong" else "")
                  + f"; {row.get('label_reads_on_it', 0)} camera(s) read {label!r} on it, "
                    f"{row['label_reads_anywhere']} anywhere")
            with log.open("a") as f:
                f.write(json.dumps(row) + "\n")
        folder = out / "trials" / args.set / f"{t:02d}"
        save_frames(vision, folder, marks)
        save_cloud(args, vision, folder, spot, located)
    no_arm()
    return report(args)


def run(args):
    others = "REMOVED (their containers stay)" if args.phase == "unlabelled" else "on the bench"
    input(f"\n>>> {args.set}: put {args.object} on its own label; the other three labels "
          f"{others}. Press Enter for the ground truth: ")
    vision = take_truth(args)
    return trials(args, vision)


# --------------------------------------------------------------------- report

def read_jsonl(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def no_reads(r):
    """A locate the label reader never answered (older logs: not found with no reads)."""
    return r.get("outcome") in ("not found", "no reads") and not r.get("all_reads")


def limit_line(spots, ok):
    """Farthest spot found on every locate, nearest spot missed at least once."""
    good = [d for d, n in spots if all(o in ok for o in n)]
    bad = [d for d, n in spots if not all(o in ok for o in n)]
    return (f"{max(good):g}" if good else "none", f"{min(bad):g}" if bad else "none")


def report(args):
    out = Path(args.out)
    rows = read_jsonl(out / "trials.jsonl")
    if not rows:
        sys.exit(f"No {out / 'trials.jsonl'} yet.")
    checks = read_jsonl(out / "checks.jsonl")
    order = [set_name(o, p) for o in OBJECTS for p in PHASES]
    sets = [s for s in order if any(r["set"] == s for r in rows)]
    sets += sorted({r["set"] for r in rows} - set(order))

    lines = [f"# Label distance test ({out.name})", "",
             "x is moved off its label; distance is measured by hand from where x stood on "
             "its label. `labelled`: the other three labels on the bench, so elimination is "
             "possible; `unlabelled`: they are removed, so only reading x's label finds x. "
             f"Wrong: the locate landed within {args.match_tol * 1000:.0f} mm of the "
             "partner's ground-truth spot (the other container of x's kind).", "",
             "## Summary", "",
             "| x | phase | on own label | by label | by elimination | wrong | not found | "
             "farthest always by label cm | nearest missed by label cm | farthest always "
             "found cm | nearest not found cm |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    detail = []
    csv = ["set,object,phase,trial,distance_cm,locates,"
           + ",".join(o.replace(" ", "_") for o in OUTCOMES)]
    skipped = sum(no_reads(r) for r in rows + checks)
    for s in sets:
        mine = [r for r in rows if r["set"] == s and not no_reads(r)]
        if not mine:
            continue
        obj, phase = mine[0]["object"], mine[0].get("phase", "")
        # Keyed by distance too: before trials were numbered on from the last
        # one, two spots could share a number.
        by_trial = {}
        for r in mine:
            by_trial.setdefault((r["trial"], r["distance_cm"]), []).append(r)
        detail += [f"## {obj}, {phase}", "",
                   "| trial | distance cm | by label | by elimination | wrong | not found "
                   "| label reads on x per locate |",
                   "|---|---|---|---|---|---|---|"]
        spots, totals = [], {o: 0 for o in OUTCOMES}
        for (t, _d), rs in sorted(by_trial.items(), key=lambda kv: kv[0][1]):
            n = {o: sum(r["outcome"] == o for r in rs) for o in OUTCOMES}
            for o in OUTCOMES:
                totals[o] += n[o]
            detail.append(
                f"| {t} | {rs[0]['distance_cm']:g} | "
                + " | ".join(f"{n[o]}/{len(rs)}" for o in OUTCOMES)
                + " | " + ", ".join(str(r.get("label_reads_on_it", "-")) for r in rs) + " |")
            csv.append(",".join(str(v) for v in [
                s, obj, phase, t, rs[0]["distance_cm"], len(rs), *(n[o] for o in OUTCOMES)]))
            spots.append((rs[0]["distance_cm"], [r["outcome"] for r in rs]))
        detail.append("")
        n_loc = sum(totals.values())
        mine_checks = [c for c in checks if c["set"] == s and not no_reads(c)]
        own = (f"{sum(c['outcome'] == 'by label' for c in mine_checks)}/{len(mine_checks)}"
               if mine_checks else "-")
        far_l, near_l = limit_line(spots, {"by label"})
        far_f, near_f = limit_line(spots, {"by label", "by elimination"})
        lines.append(f"| {obj} | {phase} | {own} | "
                     + " | ".join(f"{totals[o]}/{n_loc}" for o in OUTCOMES)
                     + f" | {far_l} | {near_l} | {far_f} | {near_f} |")
    lines += ["", "\"Always\": every locate at that spot. \"On own label\": the ground-truth "
              "checks, x on its label, found by label."]
    if skipped:
        lines.append(f"Left out: {skipped} locate(s) where no label read came back at all "
                     "(reader unreachable), not counted as not found.")
    lines += [""] + detail
    text = "\n".join(lines) + "\n"
    (out / "report.md").write_text(text)
    (out / "summary.csv").write_text("\n".join(csv) + "\n")
    print("\n" + text)
    print(f"Wrote {out / 'report.md'} and {out / 'summary.csv'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["run", "ground-truth", "trials", "report",
                                         "render-cloud", "view-cloud"])
    parser.add_argument("--object", default="baking soda cup", choices=OBJECTS,
                        help="x: the container moved off its label")
    parser.add_argument("--phase", default="labelled", choices=PHASES,
                        help="labelled: the other three labels on the bench; unlabelled: "
                             "removed, their containers left in place")
    parser.add_argument("--trials", type=int, default=5, help="Spots to try")
    parser.add_argument("--first", type=int, default=None,
                        help="Number of the first trial (default: one after this phase's "
                             "last, so a second trials call adds spots)")
    parser.add_argument("--locates", type=int, default=3,
                        help="Locates by label at each spot (the reads vary)")
    parser.add_argument("--checks", type=int, default=3,
                        help="Locates by label on its own label, after the ground truth")
    parser.add_argument("--repeats", type=int, default=3,
                        help="Ground-truth locates per container, averaged")
    parser.add_argument("--match-tol", type=float, default=0.03,
                        help="A locate this close (m) to x's partner's ground-truth "
                             "spot is the partner")
    parser.add_argument("--out", default=str(ROOT / "experiments" /
                                             f"label_distance_{datetime.now():%Y%m%d}"))
    parser.add_argument("--cameras", nargs="+", type=int, default=[2, 3, 4, 5])
    parser.add_argument("--grounding-url",
                        default=os.environ.get("GROUNDING_URL", "http://127.0.0.1:5005"))
    parser.add_argument("--no-cloud-render", action="store_true",
                        help="Keep each trial's cloud_raw.npy but do not render the 3D view")
    parser.add_argument("--npy", default=None,
                        help="render-cloud / view-cloud: a trial's cloud_raw.npy")
    args = parser.parse_args()
    args.set = set_name(args.object, args.phase)
    if args.step == "ground-truth":
        take_truth(args)
        return 0
    if args.step in ("render-cloud", "view-cloud") and not args.npy:
        parser.error(f"{args.step} needs --npy <a trial's cloud_raw.npy>")
    return {"run": run, "trials": trials, "report": report, "render-cloud": render_cloud,
            "view-cloud": view_cloud}[args.step](args)


if __name__ == "__main__":
    sys.exit(main())
