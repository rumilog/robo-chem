"""
The booklet's Magic Beaker, scripted -- no planner -- on the simulated cell.

25 skill calls: the big scoop takes citric acid into clear cup A and baking
soda into B, the small scoop takes red cabbage powder into the beaker (which
holds its 50 ml of water from the start), the stirrer mixes A, B and the
beaker, the beaker pours into A (80 deg) and B (90), and A and B are poured
into C. Every tool and cup goes back where it came from.

After each skill it prints what robochem.sim.lab says moved, and at the end
every contact the arm -- or what it holds -- made with anything else, per step,
plus any prop that ended more than 3 mm from where it started. The only
contacts a clean run has are the grasps, the cups set down, the stirrer in its
holder, and the stirrer meeting the floor of A and B while ``to_floor`` feels
for it.

    perception_env/bin/python scripts/sim_magic_beaker.py
    perception_env/bin/python scripts/sim_magic_beaker.py --viewer
    perception_env/bin/python scripts/sim_magic_beaker.py --seed 4242 --steps 1-10
"""

import argparse
import collections
import json
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robochem.sim import build_cell                     # noqa: E402
from robochem.sim import sim_arm as SA                  # noqa: E402

SPOON = {"z_offset": 0.0, "grasp_force": 1.0}
# Picked as a planner picks them, on pick_up's defaults. The bench-validated
# beaker values (z_offset 0.02) put the TCP above a 47 mm cup's rim, so the
# pads held its top 2 mm.
CUP = {}
CHAIN = [
    ("pick_up", {"object_name": "larger spoon", **SPOON}),
    ("scoop", {"powder_source": "citric acid"}),
    ("dump", {"target_container": "clear cup a"}),
    ("scoop", {"powder_source": "baking soda"}),
    ("dump", {"target_container": "clear cup b"}),
    ("place", {"target_location": "larger spoon"}),
    ("pick_up", {"object_name": "smaller spoon", **SPOON}),
    ("scoop", {"powder_source": "red cabbage powder"}),
    ("dump", {"target_container": "plastic beaker"}),
    ("place", {"target_location": "smaller spoon"}),
    ("pick_up", {"object_name": "stirring rod"}),
    ("stir", {"target_container": "clear cup a", "revolutions": 2, "to_floor": True}),
    ("stir", {"target_container": "clear cup b", "revolutions": 2, "to_floor": True}),
    ("stir", {"target_container": "plastic beaker", "revolutions": 2, "to_floor": True}),
    ("place", {"target_location": "stirring rod"}),
    ("pick_up", {"object_name": "plastic beaker", **CUP}),
    ("pour", {"target_container": "clear cup a", "pour_angle": 80}),
    ("pour", {"target_container": "clear cup b", "pour_angle": 90}),
    ("place", {"target_location": "plastic beaker"}),
    ("pick_up", {"object_name": "clear cup a", **CUP}),
    ("pour", {"target_container": "clear cup c"}),
    ("place", {"target_location": "clear cup a"}),
    ("pick_up", {"object_name": "clear cup b", **CUP}),
    ("pour", {"target_container": "clear cup c"}),
    ("place", {"target_location": "clear cup b"}),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", default=f"1-{len(CHAIN)}", help="e.g. 1-13")
    ap.add_argument("--seed", type=int, default=None,
                    help="random layout seed (default: the fixed bench)")
    ap.add_argument("--viewer", action="store_true", help="watch it, in real time")
    ap.add_argument("--out", default=None, help="save the four camera frames at the end (png)")
    ap.add_argument("--set", action="append", default=[], metavar="N:JSON",
                    help="override step N's params, e.g. '20:{\"pour_angle\": 70}'")
    args = ap.parse_args()
    lo, hi = (int(x) for x in args.steps.split("-"))
    over = {int(k): json.loads(v) for k, v in (s.split(":", 1) for s in args.set)}

    # Counted on the CLASS, before the cell exists, so the lab's own wrapper
    # (an instance attribute) calls through it. Wrapping the instance after
    # the lab has started silently drops one or the other: an earlier version
    # of this audit reported "0 contacts" for runs that had thousands.
    current = ["-"]
    hits = collections.Counter()
    original = SA.SimFrankaArm._step

    def audited(self, n=1):
        original(self, n)
        m, d = self.model, self.data
        if not hasattr(self, "_robot_ids"):
            hand = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "hand")
            self._robot_ids = {b for b in range(m.nbody)
                               if m.body_rootid[b] == m.body_rootid[hand]}
        mine = self._robot_ids | {self.scene.prop_bodies[k] for k in self._attached}
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if (b1 in mine) != (b2 in mine):
                a, b = (b1, b2) if b1 in mine else (b2, b1)
                hits[(current[0], mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(a)),
                      "table" if b == 0 else
                      mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(b)))] += 1

    SA.SimFrankaArm._step = audited

    cell = build_cell(viewer=args.viewer, realtime=args.viewer, verbose=False,
                      layout_seed=args.seed, workspace_min=[0.25, -0.40, -0.13])
    d = cell.scene.data
    start = {p.name: d.xpos[cell.scene.prop_bodies[p.name]].copy()
             for p in cell.scene.bench.props}
    t0 = time.time()
    ok_all = True
    try:
        for i, (name, params) in enumerate(CHAIN, 1):
            if not lo <= i <= hi:
                continue
            params = {**params, **over.get(i, {})}
            current[0] = f"{i}:{name}"
            ok, res = cell.skills.execute(name, params)
            extra = {k: res.get(k) for k in ("tip_achieved", "tip_after", "error")
                     if isinstance(res, dict) and res.get(k) is not None}
            print(f"### step {i} {name} {params} -> {'ok' if ok else 'FAILED'} {extra}",
                  flush=True)
            if not ok:
                ok_all = False
                break

        print(f"\n=== {time.time() - t0:.0f} s; contacts of the arm or what it holds, per step ===")
        for (step, a, b), n in sorted(hits.items(), key=lambda kv: int(kv[0][0].split(":")[0])):
            print(f"  {step:14s} {a:28s} x {b:28s} {n:6d}")
        print("=== props moved more than 3 mm ===")
        for prop, p0 in start.items():
            moved = float(np.linalg.norm(d.xpos[cell.scene.prop_bodies[prop]] - p0)) * 1000
            if moved > 3:
                print(f"  {prop:24s} {moved:6.1f} mm")
        print("=== contents ===\n" + cell.lab.summary())
        if args.out:
            from PIL import Image
            f = cell.vision.capture_scene()
            Image.fromarray(np.vstack([np.hstack(f[:2]), np.hstack(f[2:])])).save(args.out)
        if args.viewer:
            cell.arm.hold(1e9)
    finally:
        cell.close()
    print("CHAIN", "OK" if ok_all else "FAILED")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
