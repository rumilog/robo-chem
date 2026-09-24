"""
A geometric estimate of what a scoop collects from a powder bed -- no particles.

Granules were the physical alternative (``build_scene(granules=True)``), and
they were both slow -- a bed of a hundred 8 mm spheres is a thousand contacts
every step, a finer one ten times that -- and a poor stand-in for powder: an
8 mm sphere does not even fit under the scoop's 7.5 mm-deep mouth. So the bed
is a drawn fill with no contacts, and this module tallies what the stroke
takes from it.

The model is kinematic. The powder is at rest; it can only enter the bowl
through the MOUTH, and only where the mouth is advancing INTO the powder. Each
patch of the opening that lies inside the bed (under the surface, above the
floor, inside the dish) adds

    dV = dA * max(0, dp . n)

per step, where ``n`` is the mouth's outward normal and ``dp`` the patch's
displacement over that step: the volume the patch sweeps through, if it is
sweeping outward. The total is capped at the bowl's struck capacity.

What that rewards is what a person scooping does. Plunging a tilted bowl
straight down moves its forward-facing mouth sideways to itself: little. A
steep bowl driven forward presents its whole mouth to the powder: a lot. A
level bowl moved forward faces up and collects nothing -- it only bulldozes.
It deliberately ignores powder flowing into a submerged void from above and
the heap that can ride above the rim, so it errs low.

Out of the bed the same tally tracks what LEAVES the bowl. Powder is not a
liquid: its free surface can stand at up to the angle of repose, so it only
starts to run once the bowl tips past that angle, and it runs over the LOWEST
point of the rim. The bowl keeps what fits below that point with the surface
leaning back toward the mouth by the repose angle; the rest falls straight
down from there, and lands in whichever container's opening lies under that
point -- or on the table. Cohesion, which is what shaking is for, is ignored,
so this errs toward a clean dump.

Usage (as film_sim_skill does)::

    tally = ScoopTally(cell, spoon_prop, bed_from_prop(scene, cup_prop))
    tally.start()                 # wraps the arm's physics step
    cell.skills.execute("scoop", params)
    tally.stop()
    print(tally.report())
    tally.start()                 # keep the load, now track where it goes
    cell.skills.execute("dump", {"target_container": "white paper cup"})
    tally.stop()
    print(tally.report())
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import mujoco
import numpy as np


@dataclass
class PowderBed:
    """A flat-topped powder bed filling a round dish."""

    centre: np.ndarray        # dish centre, world xy
    inner_radius: float       # inside radius of the dish, metres
    floor_z: float            # inside floor, world z
    surface_z: float          # powder surface, world z

    def contains(self, points: np.ndarray) -> np.ndarray:
        """Which of ``points`` (N x 3, world) lie inside the powder."""
        radial = np.linalg.norm(points[:, :2] - self.centre[None, :], axis=1)
        return ((points[:, 2] < self.surface_z) & (points[:, 2] > self.floor_z)
                & (radial < self.inner_radius))


def bed_from_prop(scene, prop) -> PowderBed:
    """The bed a container prop holds, from its model geometry and powder_level."""
    body = scene.prop_bodies[prop.name]
    pos = scene.data.xpos[body]
    floor = pos[2] - prop.height / 2 + 2 * prop.wall
    return PowderBed(centre=np.asarray(pos[:2], float).copy(),
                     inner_radius=prop.radius - prop.wall,
                     floor_z=float(floor),
                     surface_z=float(floor + prop.powder_level))


@dataclass
class Opening:
    """The mouth of a container that falling powder can land in."""

    name: str
    centre: np.ndarray        # world xy
    inner_radius: float
    floor_z: float            # inside floor, world z
    heap_geom: int = -1       # the drawn pile of what landed, -1 for none
    heap_body: int = -1


def openings_from_bench(scene) -> List[Opening]:
    """Every container on the bench, as somewhere spilled powder can land."""
    out = []
    for prop in scene.bench.props:
        if prop.kind != "container" or prop.mesh:
            continue
        body = scene.prop_bodies[prop.name]
        pos = scene.data.xpos[body]
        out.append(Opening(
            name=prop.name,
            centre=np.asarray(pos[:2], float).copy(),
            inner_radius=prop.radius - prop.wall,
            floor_z=float(pos[2] - prop.height / 2 + 2 * prop.wall),
            heap_geom=mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_GEOM,
                                        f"{prop.body}_received"),
            heap_body=mujoco.mj_name2id(scene.model, mujoco.mjtObj.mjOBJ_BODY,
                                        f"{prop.body}_received"),
        ))
    return out


class ScoopTally:
    """
    Integrates the powder a held scoop's mouth sweeps into its bowl.

    Hooks the arm's physics step, like film_sim_skill's ContactWatch, so it
    sees the motion as it actually ran rather than as it was commanded.
    """

    def __init__(self, cell, spoon_prop, bed: PowderBed, grid=(10, 6),
                 bulk_density: float = 0.9, repose_deg: float = 35.0,
                 receivers: Optional[List[Opening]] = None):
        """
        Args:
            cell: A SimCell (build_cell)
            spoon_prop: The scoop's Prop; needs bowl_size / bowl_offset / wall
            bed: The powder bed being scooped
            grid: Patches along x (length) and y (width) of the mouth
            bulk_density: g/ml, to express the volume as a mass. ~0.9 suits
                citric acid and baking soda powders.
            repose_deg: The powder's angle of repose. Out of the bed, a bowl
                tipped further than this starts to lose its load.
            receivers: Where falling powder can land. Default: every container
                on the bench, the source included.
        """
        self.cell = cell
        self.model, self.data = cell.scene.model, cell.scene.data
        self.bed = bed
        self.body = cell.scene.prop_bodies[spoon_prop.name]
        self.bulk_density = float(bulk_density)
        self.repose_deg = float(repose_deg)

        bowl = np.asarray(spoon_prop.bowl_size, float)
        centre = np.asarray(spoon_prop.bowl_offset, float)
        wall = spoon_prop.wall / 2           # panel half-thickness, as in scene.py
        self.half_len = bowl[0] - 2 * wall   # inside half-extents of the cavity
        self.half_wid = bowl[1] - 2 * wall
        self.floor_z = centre[2] - bowl[2] + 2 * wall   # inside of the bowl floor
        mouth_z = centre[2] + bowl[2]                   # top of the walls
        self.depth = mouth_z - self.floor_z
        self.capacity = 4 * self.half_len * self.half_wid * self.depth
        self.centre = centre

        nx, ny = grid
        xs = centre[0] + (np.arange(nx) + 0.5) / nx * 2 * self.half_len - self.half_len
        ys = (np.arange(ny) + 0.5) / ny * 2 * self.half_wid - self.half_wid
        self.patches = np.array([[x, y, mouth_z] for x in xs for y in ys])   # body frame
        self.patch_area = 4 * self.half_len * self.half_wid / (nx * ny)
        self.normal = np.array([0.0, 0.0, 1.0])     # mouth faces body +z

        # The cavity as a lattice of equal cells, for how much of it lies below
        # the rim at a given tilt, and the four corners of the rim.
        cx = centre[0] + (np.arange(12) + 0.5) / 12 * 2 * self.half_len - self.half_len
        cy = (np.arange(8) + 0.5) / 8 * 2 * self.half_wid - self.half_wid
        cz = self.floor_z + (np.arange(6) + 0.5) / 6 * self.depth
        self.cells = np.array([[x, y, z] for x in cx for y in cy for z in cz])
        self.cell_volume = self.capacity / len(self.cells)
        self.corners = np.array([[centre[0] + sx * self.half_len, sy * self.half_wid, mouth_z]
                                 for sx in (-1, 1) for sy in (-1, 1)])
        self.receivers = (openings_from_bench(cell.scene) if receivers is None
                          else list(receivers))
        self.landed = {r.name: 0.0 for r in self.receivers}   # m^3 per container
        self._heap_xy = {r.name: np.zeros(2) for r in self.receivers}
        self.lost = 0.0               # m^3 that fell outside every container
        self.peak = 0.0               # most the bowl held at once
        self.pours = []               # runs of falling powder, see _log_pour

        self.load_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM,
                                           f"{spoon_prop.body}_load")
        self.collected = 0.0          # m^3 in the bowl
        self.swept = 0.0              # m^3 swept into the mouth, before the cap
        self.submerged_steps = 0
        self.max_tilt_after = 0.0     # deg from level once the mouth has left the bed
        self._left_bed = False
        self._prev = None
        self._original = None

    # ------------------------------------------------------------ hooking
    def start(self):
        arm = self.cell.arm
        self._original = arm._step

        def stepped(n=1):
            self._original(n)
            self._update()

        arm._step = stepped
        self._prev = self._patch_world()
        self._show()

    def stop(self):
        if self._original is not None:
            self.cell.arm._step = self._original
            self._original = None

    # --------------------------------------------------------------- core
    def _pose(self):
        return (self.data.xpos[self.body].copy(),
                self.data.xmat[self.body].reshape(3, 3).copy())

    def _patch_world(self):
        pos, mat = self._pose()
        return pos[None, :] + self.patches @ mat.T

    def _update(self):
        pts = self._patch_world()
        _, mat = self._pose()
        n = mat @ self.normal
        mid = 0.5 * (pts + self._prev)
        inside = self.bed.contains(mid)
        if inside.any():
            advance = np.clip((pts - self._prev) @ n, 0.0, None)
            dv = float((advance * inside).sum() * self.patch_area)
            self.swept += dv
            self.collected = min(self.capacity, self.collected + dv)
            self.submerged_steps += 1
            if self._left_bed:
                self._left_bed = False       # went back in
        elif self.collected > 0:
            self._left_bed = True
            tilt = float(np.degrees(np.arccos(np.clip(n[2], -1.0, 1.0))))
            self.max_tilt_after = max(self.max_tilt_after, tilt)
            keep = self._holdable(mat, n)
            if self.collected > keep + 1e-12:
                self._land(self.collected - keep)
                self.collected = keep
        self.peak = max(self.peak, self.collected)
        self._prev = pts
        self._show()

    def _holdable(self, mat, n) -> float:
        """The most the bowl can keep at this orientation, m^3."""
        theta = float(np.arccos(np.clip(n[2], -1.0, 1.0)))
        phi = np.radians(self.repose_deg)
        if theta <= phi:
            return self.capacity
        if theta - phi >= np.pi / 2:
            return 0.0
        # The free surface leans from level toward the mouth by the angle of
        # repose -- the steepest slope the powder stands at -- and passes
        # through the lowest corner of the rim. Below it stays; above it runs.
        up = np.array([0.0, 0.0, 1.0])
        lean = (np.sin(theta - phi) * up + np.sin(phi) * n) / np.sin(theta)
        lean_body = mat.T @ lean
        lip = float((self.corners @ lean_body).min())
        return float((self.cells @ lean_body <= lip).sum() * self.cell_volume)

    def _land(self, dv: float):
        """Drop ``dv`` straight down from the lowest point of the rim."""
        pos, mat = self._pose()
        rim = pos[None, :] + self.corners @ mat.T
        low = rim[:, 2] <= rim[:, 2].min() + 1e-3     # an edge, or a corner
        xy = rim[low, :2].mean(axis=0)
        for r in self.receivers:
            if np.linalg.norm(xy - r.centre) < r.inner_radius:
                # The pile's centre is the volume-weighted landing point.
                total = self.landed[r.name] + dv
                self._heap_xy[r.name] = (self._heap_xy[r.name] * self.landed[r.name]
                                         + (xy - r.centre) * dv) / total
                self.landed[r.name] = total
                self._show_heap(r)
                self._log_pour(r.name, xy, dv, mat)
                return
        self.lost += dv
        self._log_pour(None, xy, dv, mat)

    def _log_pour(self, into, xy, dv, mat):
        """One run per stretch of powder landing in the same place."""
        t = float(self.data.time)
        tilt = float(np.degrees(np.arccos(np.clip((mat @ self.normal)[2], -1.0, 1.0))))
        last = self.pours[-1] if self.pours else None
        if last is not None and last["into"] == into:
            last["ml"] += dv * 1e6
            last["t1"] = t
            last["tilt_deg"][1] = tilt
            last["x_range"] = [min(last["x_range"][0], xy[0]), max(last["x_range"][1], xy[0])]
            last["y_range"] = [min(last["y_range"][0], xy[1]), max(last["y_range"][1], xy[1])]
        else:
            self.pours.append({"into": into, "ml": dv * 1e6, "t0": t, "t1": t,
                               "x_range": [float(xy[0])] * 2,
                               "y_range": [float(xy[1])] * 2,
                               "tilt_deg": [tilt, tilt]})

    def _show_heap(self, r: Opening, radius: float = 0.012):
        """Draw what landed in ``r`` as a flat pile on its floor."""
        if r.heap_geom < 0:
            return
        m = self.model
        rad = min(radius, 0.8 * r.inner_radius)
        h = max(self.landed[r.name] / (np.pi * rad ** 2), 1e-4)
        off = self._heap_xy[r.name]
        room = r.inner_radius - rad        # keep the pile inside the walls
        if np.linalg.norm(off) > room:
            off = off / np.linalg.norm(off) * max(room, 0.0)
        m.geom_size[r.heap_geom] = [rad, h / 2, 0]
        m.geom_pos[r.heap_geom] = [off[0], off[1], h / 2]
        m.geom_rbound[r.heap_geom] = float(np.hypot(rad, h / 2))
        m.geom_rgba[r.heap_geom, 3] = 1.0

    def _show(self):
        """Draw the load as a level fill in the bowl, height = volume / area."""
        if self.load_geom < 0:
            return
        m = self.model
        h = max(self.collected / (4 * self.half_len * self.half_wid), 1e-4)
        m.geom_size[self.load_geom] = [self.half_len, self.half_wid, h / 2]
        m.geom_pos[self.load_geom] = [self.centre[0], 0.0, self.floor_z + h / 2]
        m.geom_rbound[self.load_geom] = float(np.linalg.norm(m.geom_size[self.load_geom]))
        m.geom_rgba[self.load_geom, 3] = 1.0 if self.collected > 1e-9 else 0.0

    # ------------------------------------------------------------ results
    def result(self) -> dict:
        ml = self.collected * 1e6
        return {
            "collected_ml": ml,                 # in the bowl now
            "collected_g": ml * self.bulk_density,
            "fill_fraction": self.collected / self.capacity if self.capacity else 0.0,
            "peak_ml": self.peak * 1e6,         # the most it held at once
            "capacity_ml": self.capacity * 1e6,
            "swept_ml": self.swept * 1e6,
            "submerged_steps": self.submerged_steps,
            "max_tilt_after_exit_deg": self.max_tilt_after,
            "spill_risk": self.max_tilt_after > self.repose_deg,
            "landed_ml": {k: v * 1e6 for k, v in self.landed.items() if v > 0},
            "lost_ml": self.lost * 1e6,         # fell outside every container
            "pours": self.pours,
            "bulk_density_g_per_ml": self.bulk_density,
        }

    def report_dump(self, target: str) -> str:
        """Where the load went, for a skill that empties the bowl into ``target``."""
        r = self.result()
        into = r["landed_ml"].get(target, 0.0)
        others = {k: v for k, v in r["landed_ml"].items() if k != target}
        line = (f"powder (geometric estimate): {into:.2f} ml of the "
                f"{r['peak_ml']:.2f} ml scooped landed in the {target!r}")
        for name, v in others.items():
            line += f"; {v:.2f} ml fell into the {name!r}"
        if r["lost_ml"] > 1e-3:
            line += f"; {r['lost_ml']:.2f} ml fell outside every container"
        line += f"; {r['collected_ml']:.2f} ml is still in the bowl"
        for run in r["pours"]:
            if run["ml"] < 0.005:
                continue
            (x0, x1), (y0, y1) = run["x_range"], run["y_range"]
            line += (f"\n    t={run['t0']:.1f}..{run['t1']:.1f}s  {run['ml']:.2f} ml -> "
                     f"{run['into'] or 'OUTSIDE every container'}, falling at "
                     f"x {x0:.3f}..{x1:.3f} y {y0:.3f}..{y1:.3f}, while the bowl "
                     f"tipped {run['tilt_deg'][0]:.0f}->{run['tilt_deg'][1]:.0f} deg")
        return line

    def report(self) -> str:
        r = self.result()
        line = (f"powder (geometric estimate): {r['collected_ml']:.2f} ml "
                f"(~{r['collected_g']:.2f} g) = {100 * r['fill_fraction']:.0f}% of "
                f"the bowl's {r['capacity_ml']:.2f} ml; the mouth swept "
                f"{r['swept_ml']:.2f} ml of the bed")
        if r["spill_risk"]:
            line += (f"; tipped {r['max_tilt_after_exit_deg']:.0f} deg after "
                     f"leaving the bed, past a {self.repose_deg:.0f} deg angle "
                     f"of repose, so expect spill")
        return line
