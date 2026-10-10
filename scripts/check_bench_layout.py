"""
Can the cell find everything on the bench, and is the layout safe for the arm?

Locates every object of a bench manifest on the real cell -- cameras and the
grounding service only, the arm is never connected -- and then checks the
layout as found against the sim's layout rules (``robochem/sim/bench.py
layout_violations``): the reach limits, body gaps, holder clearance and dump
sweep corridors that were measured off the arm in sim.

The bench is not expected to match the sim's default positions. It is laid out
by hand and moved between runs, the way the sim's random layouts move it, so
what matters is that each object is found and that no measured rule is broken
where things actually stand.

The coordinates are the arm's base frame: x away from the arm, y to the arm's
left (as seen standing behind the arm, looking where it looks).

Usage:
    source scripts/env.sh
    python scripts/check_bench_layout.py                      # all objects
    python scripts/check_bench_layout.py "clear cup a" stirrer

Each object costs a four-camera capture and, for a labelled container, one
label read per container per camera, so a full check takes several minutes.
"""

import argparse
import importlib.util
import os
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from robochem.vision import VisionSystem
from robochem.vision.bench_manifest import BENCHES


def sim_bench_module():
    """``robochem/sim/bench.py``, loaded by path.

    Importing the ``robochem.sim`` package imports mujoco, which the robot venv
    (Python 3.8) does not have and cannot install. bench.py itself needs only
    numpy.
    """
    spec = importlib.util.spec_from_file_location(
        "_sim_bench", ROOT / "robochem" / "sim" / "bench.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module      # dataclasses look their module up here
    spec.loader.exec_module(module)
    return module


def rule_violations(found: dict, bench=None) -> list:
    """The sim's layout rules applied to where things were found.

    ``found`` maps bench names to (x, y). The sim's props supply each object's
    body (radius, height, label paper), except where the bench manifest knows
    the real one's size: the sim's beaker is 95 mm tall and 70 mm across, the
    real bench's a narrow clear cup. Objects not found are left out rather
    than checked at a position they may not have.
    """
    sim = sim_bench_module()
    where = dict(found)
    if "stirrer" in where:
        # The holder is not located: the stirrer stands in it.
        where.setdefault("stirrer holder", where["stirrer"])
    props = []
    for p in sim.default_bench():
        if p.name not in where:
            continue
        size = {}
        real = bench.find(p.name) if bench is not None else None
        if real is not None and real.radius and real.height:
            size = {"radius": float(real.radius), "height": float(real.height)}
        props.append(replace(p, pos=tuple(where[p.name]), **size))
    return sim.layout_violations(sim.Bench(props=props))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", help="Objects to check (default: all)")
    parser.add_argument("--bench", choices=sorted(BENCHES), default="magic_beaker")
    parser.add_argument("--cameras", nargs="+", type=int, default=[2, 3, 4, 5])
    parser.add_argument("--grounding-url",
                        default=os.environ.get("GROUNDING_URL", "http://127.0.0.1:5005"))
    args = parser.parse_args()

    bench = BENCHES[args.bench]
    # The holder is not looked for: put-back goes to where the stirrer was
    # picked, and the holder's position is the stirrer's.
    objects = [o for o in bench.objects if o.label or o.prompt]
    if args.names:
        wanted = {bench.canonical(n) for n in args.names}
        objects = [o for o in objects if o.name in wanted]

    vision = VisionSystem(cameras={}, grounding_url=args.grounding_url, bench=bench)
    vision.object_localizer.camera_ids = list(args.cameras)

    rows = []
    for obj in objects:
        print(f"\n========== {obj.name} ==========", flush=True)
        vision.clear_cache()
        try:
            loc = vision.locate(obj.name, force_refresh=True)
        except Exception as exc:
            print(f"[check] {obj.name}: {type(exc).__name__}: {exc}")
            loc = None
        rows.append((obj, loc))

    print("\n" + "=" * 78)
    print(f"Bench '{bench.name}' (x away from the arm, y to its left)")
    print("=" * 78)
    found = {}
    for obj, loc in rows:
        if loc is None:
            print(f"  NOT FOUND  {obj.name}")
            continue
        c = np.asarray(loc["centroid"], dtype=float)
        found[obj.name] = (float(c[0]), float(c[1]))
        cams = loc.get("cameras") or []
        how = loc.get("resolved_by") or "SAM prompt"
        if loc.get("localized_by"):
            how += f"; placed by {loc['localized_by']}"
        weak = "  <- one camera only: trust it less" if len(cams) < 2 else ""
        print(f"  found      {obj.name:24s} ({c[0] * 100:4.0f}, {c[1] * 100:+4.0f}) cm "
              f"[{len(cams)} cams: {how}]{weak}")

    violations = rule_violations(found, bench) if found else []
    missing = [o.name for o, loc in rows if loc is None]
    print()
    if violations:
        print(f"Layout rules measured in sim: {len(violations)} broken where things stand now")
        for v in violations:
            print(f"  - {v}")
    else:
        print("Layout rules measured in sim: none broken by what was found")
    if missing:
        print(f"Not checked against the rules (not found): {', '.join(missing)}")
    print()
    return 0 if not (violations or missing) else 1


if __name__ == "__main__":
    sys.exit(main())
