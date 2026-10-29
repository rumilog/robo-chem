"""
How well does the cell read container labels? Ground truth vs repeated scans.

Cameras and the grounding service only: this script never imports frankapy
and never connects to the arm. It exits if anything did.

Each labelled container is found exactly as a skill finds its target on the
real cell (``VisionSystem.locate`` -> ``_locate_by_vote``): every container of
the category is placed, every instance's label is read by the strong model
from the bench's closed label set, and the reads are voted across cameras.
The only differences from a run are that nothing moves and that the vote is
given no memory: kept scans, last-seen positions (the tie-breaker) and
arm-placed positions are cleared before every locate, so a tie or a misread
is not rescued by where the container was found last time.

The bench (2026-10-29): four containers in a square. Between sets the labels
are swapped and the square may be moved, but each container keeps its corner,
so every set takes its own ground truth.

    red cabbage powder (dish)    top left      baking soda (dish)     top right
    10 ml water (clear cup)      bottom left   vegetable oil (beaker) bottom right

"Top" is away from the arm (+x), "left" is the arm's left (+y), as seen
standing behind the arm.

1. Ground truth per set, labels NOT read. Every container is placed with its
   label reads switched off, each is named by its quadrant, and then located
   again through the same code path, held to that spot
   (``VisionSystem.tracked``) instead of being picked by label. Repeated
   ``--repeats`` times and averaged; the spread is the localisation noise
   floor. ``scan`` does this first when the set has no ground truth yet
   (``--redo-truth`` to retake it); on its own:

       python scripts/label_reading_benchmark.py ground-truth --set rumi

2. One set of scans per label writer. A scan locates the four containers in
   order, one full capture each; ``--scans`` 10 of them per set.

       python scripts/label_reading_benchmark.py scan --set printed
       # swap the labels (and move the square if needed), then
       python scripts/label_reading_benchmark.py scan --set rumi
       python scripts/label_reading_benchmark.py scan --set bozhang
       python scripts/label_reading_benchmark.py scan --set aliyah

3. Table of every set (also printed at the end of each scan run):

       python scripts/label_reading_benchmark.py report

Everything goes to ``--out`` (default experiments/label_benchmark_<date>):
ground_truth_<set>.json, ground_truth_frames_<set>/ (the ground truth drawn
on the frames, check it), scans.jsonl (one line per locate), report.md,
summary.csv, and failures/ (the frames of every locate that was wrong or
found nothing).
"""

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from robochem.vision import VisionSystem
from robochem.vision.bench_manifest import JELLYFISH, BenchManifest

#: In scan order, with the quadrant each one stands in (row, column).
LAYOUT = [
    ("red cabbage powder cup", "top", "left"),
    ("baking soda cup", "top", "right"),
    ("10 ml water cup", "bottom", "left"),
    ("beaker", "bottom", "right"),
]

#: The jellyfish bench's dishes and oil beaker, and its clear cup on a
#: "10 ML WATER" label instead of "20 ML WATER". The label reader chooses
#: among exactly these four labels, as it would on a bench of four.
LABEL_TEST = BenchManifest("label_test", [
    JELLYFISH.find("red cabbage powder cup"),
    JELLYFISH.find("baking soda cup"),
    replace(JELLYFISH.find("other clear cup"), name="10 ml water cup",
            label="10 ml water", aliases=("water cup",), note="a clear plastic cup"),
    JELLYFISH.find("beaker"),
])

SETS = ("printed", "rumi", "bozhang", "aliyah")


def no_arm():
    if any(m == "frankapy" or m.startswith("frankapy.") for m in sys.modules):
        sys.exit("frankapy was imported: this script must not touch the arm. Stopping.")


def make_vision(args):
    from robochem.agents import config as agent_config
    agent_config.load_env()
    vision = VisionSystem(cameras={}, grounding_url=args.grounding_url, bench=LABEL_TEST)
    vision.object_localizer.camera_ids = list(args.cameras)
    vision.reuse_scans = False
    no_arm()
    return vision


def fresh(vision):
    """Forget everything a previous locate left behind."""
    vision.clear_cache()
    vision.scans.clear()
    vision.last_seen.clear()
    vision.tracked.clear()
    vision.seen_containers.clear()


def xyz(loc):
    return [float(v) for v in np.asarray(loc["centroid"], float)[:3]]


def save_frames(vision, folder, marks=()):
    """The last capture's colour frames, with ``marks`` [(name, xyz, bgr)] drawn on."""
    import cv2
    data = vision.object_localizer.last_capture or {}
    folder.mkdir(parents=True, exist_ok=True)
    for cam, image in (data.get("images") or {}).items():
        out = image.copy()
        intr = (data.get("intrinsics") or {}).get(cam)
        for name, point, colour in marks:
            try:
                uv = vision.object_localizer.project_world_point(point, cam, intrinsics=intr)
            except Exception:
                uv = None
            if uv is None:
                continue
            u, v = int(round(uv[0])), int(round(uv[1]))
            cv2.circle(out, (u, v), 6, colour, -1)
            cv2.putText(out, name, (u + 8, v - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)
        cv2.imwrite(str(folder / f"cam{cam}.png"), out)


# --------------------------------------------------------------- ground truth

def place_without_labels(vision):
    """Every container of each category on the bench, placed with the label reads off."""
    real_read_all = vision.label_resolver.read_all
    vision.label_resolver.read_all = lambda images, instances: {}
    try:
        spots = {}
        for category in sorted({LABEL_TEST.find(n).category for n, _r, _c in LAYOUT}):
            name = next(n for n, _r, _c in LAYOUT if LABEL_TEST.find(n).category == category)
            print(f"\n--- placing every {category!r} (labels not read) ---", flush=True)
            fresh(vision)
            vision.locate(name, force_refresh=True)  # fails by design: nothing is read
            spots[category] = [np.asarray(p, float) for p in
                               vision.seen_containers.get(category, [])]
        return spots
    finally:
        vision.label_resolver.read_all = real_read_all


def assign_quadrants(spots):
    """Name each placed container by the quadrant it stands in, or raise."""
    every = [p for ps in spots.values() for p in ps]
    if len(every) != len(LAYOUT):
        raise RuntimeError(f"expected {len(LAYOUT)} containers placed by 2+ cameras, got "
                           f"{len(every)}: {spots}")
    cx, cy = np.mean(every, axis=0)
    named = {}
    for name, row, col in LAYOUT:
        category = LABEL_TEST.find(name).category
        fits = [p for p in spots.get(category, [])
                if (p[0] > cx) == (row == "top") and (p[1] > cy) == (col == "left")]
        if len(fits) != 1:
            raise RuntimeError(f"{name}: {len(fits)} {category!r} in the {row} {col} "
                               f"quadrant (centre {cx:.3f}, {cy:+.3f}): {spots}")
        named[name] = fits[0]
    return named


def truth_path(args):
    return Path(args.out) / f"ground_truth_{args.set}.json"


def ground_truth(args, vision=None):
    """Take this set's ground truth, write it, and return it by bench name."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    vision = vision or make_vision(args)
    print(f"\n########## ground truth for set {args.set!r} (labels not read) ##########",
          flush=True)
    named = assign_quadrants(place_without_labels(vision))
    for name, p in named.items():
        print(f"[truth] {name:24s} placed at ({p[0]:.3f}, {p[1]:+.3f})")

    repeats = {name: [] for name in named}
    for r in range(args.repeats):
        for name, spot in named.items():
            print(f"\n--- ground truth {r + 1}/{args.repeats}: {name} (held to its "
                  f"quadrant, labels not read) ---", flush=True)
            fresh(vision)
            vision.tracked[vision._scan_key(name)] = np.asarray(spot, float)[:2]
            loc = vision.locate(name, force_refresh=True)
            if loc is None:
                raise RuntimeError(f"ground truth: {name} not found at its quadrant spot")
            repeats[name].append(xyz(loc))
    no_arm()

    truth = {}
    for name, row, col in LAYOUT:
        pts = np.asarray(repeats[name])
        mean = pts.mean(axis=0)
        spread = float(np.max(np.linalg.norm(pts[:, :2] - mean[:2], axis=1))) * 1000
        truth[name] = {"label": LABEL_TEST.find(name).label, "quadrant": f"{row} {col}",
                       "xyz": [float(v) for v in mean], "repeats": pts.tolist(),
                       "max_xy_spread_mm": round(spread, 1)}
        print(f"[truth] {name:24s} ({mean[0]:.4f}, {mean[1]:+.4f}, {mean[2]:.4f})  "
              f"{row} {col}, {len(pts)} repeats within {spread:.1f} mm")
    record = {"set": args.set, "time": datetime.now().isoformat(timespec="seconds"),
              "cameras": list(args.cameras), "objects": truth}
    truth_path(args).write_text(json.dumps(record, indent=2))

    colours = [(0, 0, 255), (255, 0, 255), (255, 128, 0), (0, 200, 255)]
    marks = [(name, truth[name]["xyz"], colours[i]) for i, (name, _r, _c) in enumerate(LAYOUT)]
    frames = out / f"ground_truth_frames_{args.set}"
    save_frames(vision, frames, marks)
    print(f"\nWrote {truth_path(args)}. Check the names on {frames}/cam*.png.")
    return truth


# ---------------------------------------------------------------------- scans

def scan(args):
    out = Path(args.out)
    vision = make_vision(args)
    if args.redo_truth or not truth_path(args).exists():
        truth = ground_truth(args, vision)
    else:
        truth = json.loads(truth_path(args).read_text())["objects"]
        print(f"[bench] using the existing {truth_path(args)}")

    # Every label read of every instance in every camera, not only the ones
    # that ended up on the chosen container: what a hand was read as.
    reads = {}
    real_read_all = vision.label_resolver.read_all

    def recording_read_all(images, instances):
        got = real_read_all(images, instances)
        reads.clear()
        reads.update({int(c): list(v) for c, v in got.items()})
        return got

    vision.label_resolver.read_all = recording_read_all

    log = out / "scans.jsonl"
    for i in range(args.first, args.first + args.scans):
        for name, _row, _col in LAYOUT:
            label = truth[name]["label"]
            print(f"\n========== {args.set} scan {i}/{args.first + args.scans - 1}: "
                  f"{label} ==========", flush=True)
            fresh(vision)
            reads.clear()
            started = time.time()
            try:
                loc = vision.locate(name, force_refresh=True)
                error = None
            except Exception as exc:
                loc, error = None, f"{type(exc).__name__}: {exc}"
                print(f"[bench] {name}: {error}")
            row = {"set": args.set, "scan": i, "object": name, "label": label,
                   "time": datetime.now().isoformat(timespec="seconds"),
                   "seconds": round(time.time() - started, 1),
                   "truth": truth[name]["xyz"], "found": loc is not None, "error": error,
                   "all_reads": {str(c): v for c, v in reads.items()}}
            if loc is not None:
                p = np.asarray(xyz(loc))
                d = p - np.asarray(truth[name]["xyz"])
                nearest = min(truth, key=lambda n: np.linalg.norm(
                    p[:2] - np.asarray(truth[n]["xyz"])[:2]))
                row.update({
                    "predicted": p.tolist(), "offset": d.tolist(),
                    "offset_xy_mm": round(float(np.linalg.norm(d[:2])) * 1000, 1),
                    "offset_mm": round(float(np.linalg.norm(d)) * 1000, 1),
                    "nearest_truth": nearest, "correct": nearest == name,
                    "cameras": [int(c) for c in loc.get("cameras") or []],
                    "resolved_by": loc.get("resolved_by"),
                    "label_matches": {str(c): v for c, v in
                                      (loc.get("label_matches") or {}).items()},
                })
                verdict = "correct" if row["correct"] else f"WRONG (that is {nearest})"
                print(f"[bench] {label}: ({p[0]:.4f}, {p[1]:+.4f}, {p[2]:.4f}), offset "
                      f"({d[0] * 1000:+.1f}, {d[1] * 1000:+.1f}, {d[2] * 1000:+.1f}) mm, "
                      f"|xy| {row['offset_xy_mm']} mm, {verdict}; {row['resolved_by']}")
            else:
                row["correct"] = False
                print(f"[bench] {label}: NOT FOUND")
            if not row["correct"]:
                save_frames(vision, out / "failures" / f"{args.set}_{i:02d}_{name.replace(' ', '_')}",
                            [(name, truth[name]["xyz"], (0, 255, 0))]
                            + ([("predicted", row["predicted"], (0, 0, 255))]
                               if loc is not None else []))
            with log.open("a") as f:
                f.write(json.dumps(row) + "\n")
    no_arm()
    return report(args)


# --------------------------------------------------------------------- report

def stats(rows):
    found = [r for r in rows if r["found"]]
    out = {"n": len(rows), "found": len(found), "correct": sum(r["correct"] for r in rows)}
    if found:
        d = np.asarray([r["offset"] for r in found]) * 1000
        out["mean_offset_mm"] = d.mean(axis=0).round(1).tolist()
        out["mean_xy_mm"] = round(float(np.mean([r["offset_xy_mm"] for r in found])), 1)
        out["max_xy_mm"] = round(float(np.max([r["offset_xy_mm"] for r in found])), 1)
        good = [r for r in found if r["correct"]]
        out["mean_xy_correct_mm"] = (round(float(np.mean([r["offset_xy_mm"] for r in good])), 1)
                                     if good else None)
    return out


def fmt(s):
    if not s["found"]:
        return f"| {s['correct']}/{s['n']} | 0/{s['n']} | - | - | - | - |"
    dx, dy, dz = s["mean_offset_mm"]
    return (f"| {s['correct']}/{s['n']} | {s['found']}/{s['n']} | ({dx:+.1f}, {dy:+.1f}, "
            f"{dz:+.1f}) | {s['mean_xy_mm']} | {s['mean_xy_correct_mm'] or '-'} | "
            f"{s['max_xy_mm']} |")


def report(args):
    out = Path(args.out)
    log = out / "scans.jsonl"
    if not log.exists():
        sys.exit(f"No {log} yet.")
    rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
    sets = [s for s in SETS if any(r["set"] == s for r in rows)]
    sets += sorted({r["set"] for r in rows} - set(sets))

    head = ("| | correct | found | mean offset (dx, dy, dz) mm | mean xy mm | "
            "mean xy, correct only | max xy mm |\n|---|---|---|---|---|---|---|")
    lines = [f"# Label reading benchmark ({out.name})", "",
             "Offsets are predicted minus ground truth, base frame (x away from the arm, "
             "y to its left). Correct: the predicted spot is nearer this container's "
             "ground truth than any other's. Means are over the scans that found something.",
             ""]
    csv = ["set,object,n,correct,found,mean_dx_mm,mean_dy_mm,mean_dz_mm,mean_xy_mm,"
           "mean_xy_correct_mm,max_xy_mm"]
    for s in sets:
        mine = [r for r in rows if r["set"] == s]
        lines += [f"## {s}", "", head]
        for name, _r, _c in LAYOUT:
            st = stats([r for r in mine if r["object"] == name])
            if st["n"]:
                lines.append(f"| {name} " + fmt(st))
                csv.append(csv_row(s, name, st))
        st = stats(mine)
        lines += [f"| **all** " + fmt(st), ""]
        csv.append(csv_row(s, "all", st))
    lines += ["## Every set", "", head]
    for s in sets:
        lines.append(f"| {s} " + fmt(stats([r for r in rows if r["set"] == s])))
    st = stats(rows)
    lines.append("| **total** " + fmt(st))
    csv.append(csv_row("total", "all", st))
    text = "\n".join(lines) + "\n"
    (out / "report.md").write_text(text)
    (out / "summary.csv").write_text("\n".join(csv) + "\n")
    print("\n" + text)
    print(f"Wrote {out / 'report.md'} and {out / 'summary.csv'}")
    return 0


def csv_row(s, name, st):
    d = st.get("mean_offset_mm") or ["", "", ""]
    return ",".join(str(v) for v in [s, name, st["n"], st["correct"], st["found"], *d,
                                      st.get("mean_xy_mm", ""),
                                      st.get("mean_xy_correct_mm") or "",
                                      st.get("max_xy_mm", "")])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["ground-truth", "scan", "report"])
    parser.add_argument("--set", default="printed",
                        help=f"Whose labels are on the bench: {', '.join(SETS)} (or any name)")
    parser.add_argument("--scans", type=int, default=10)
    parser.add_argument("--first", type=int, default=1,
                        help="Number of the first scan (to continue a stopped set)")
    parser.add_argument("--repeats", type=int, default=3,
                        help="Ground-truth locates per container, averaged")
    parser.add_argument("--redo-truth", action="store_true",
                        help="Retake this set's ground truth before scanning")
    parser.add_argument("--out", default=str(ROOT / "experiments" /
                                             f"label_benchmark_{datetime.now():%Y%m%d}"))
    parser.add_argument("--cameras", nargs="+", type=int, default=[2, 3, 4, 5])
    parser.add_argument("--grounding-url",
                        default=os.environ.get("GROUNDING_URL", "http://127.0.0.1:5005"))
    args = parser.parse_args()
    if args.step == "ground-truth":
        ground_truth(args)
        return 0
    return {"scan": scan, "report": report}[args.step](args)


if __name__ == "__main__":
    sys.exit(main())
