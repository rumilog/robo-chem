"""
What the containers on the simulated bench hold, and what happens to it.

MuJoCo models the arm, the props and their contacts. It has no powder, no
liquid and no chemistry, and the kit's experiments are nothing but those. This
module keeps the books instead -- kinematically, from the poses the physics
produced -- and draws the result into the scene, so the cameras, the viewer
and the planner's scene agent all see it:

  powder     what each scoop's mouth sweeps out of a dish
             (:class:`robochem.sim.powder.ScoopTally`), and where it falls
             when the bowl tips: into whichever container's opening is under
             the rim, or onto the table
  liquid     how much each container holds. Tipped past the point where its
             level surface reaches the lowest point of the rim, the excess
             runs over that point and falls straight down -- again into
             whatever opening is beneath, or onto the table
  chemistry  powders dissolve in water, much faster while a stirrer moves in
             it; citric acid and baking soda neutralise each other and fizz;
             red cabbage indicator takes the colour of the solution's pH

Every constant below is a modelling choice, stated where it is made. None of
it was measured on the kit. It exists so a plan can be run end to end and its
result looked at, not to predict the real experiment's numbers.

    cell = build_cell()            # a Lab is built and hooked by default
    cell.skills.execute("pick_up", {"object_name": "larger spoon"})
    ...
    print(cell.lab.summary())
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import mujoco
import numpy as np

from .powder import ScoopTally, beds_from_bench, openings_from_bench


# ------------------------------------------------------------------ reagents

@dataclass(frozen=True)
class Reagent:
    """A powder from the kit."""

    name: str
    kind: str                       # acid | base | indicator
    bulk_density: float             # g per ml of loose powder
    rgba: Tuple[float, float, float, float]
    molar_mass: float = 0.0         # g/mol
    equivalents: float = 0.0        # acidic protons or base equivalents per mole
    # g that dissolve per ml of WARM water (the booklet's "warm water").
    # Citric acid ~1.5 g/ml; sodium bicarbonate 0.096 at 20 C and ~0.13 at
    # 40-45 C, so a full big scoop of it is about what 10 ml can take.
    solubility: float = 1e9


REAGENTS: Dict[str, Reagent] = {
    "citric acid": Reagent("citric acid", "acid", 0.9, (0.98, 0.98, 0.94, 1.0),
                           molar_mass=192.12, equivalents=3.0, solubility=1.4),
    "baking soda": Reagent("baking soda", "base", 1.0, (1.0, 1.0, 1.0, 1.0),
                           molar_mass=84.01, equivalents=1.0, solubility=0.13),
    "red cabbage powder": Reagent("red cabbage powder", "indicator", 0.45,
                                  (0.58, 0.24, 0.50, 1.0)),
}


def reagent(name: Optional[str]) -> Optional[Reagent]:
    """The kit reagent a label or bed names, matched loosely."""
    if not name:
        return None
    key = " ".join(str(name).lower().split())
    if key in REAGENTS:
        return REAGENTS[key]
    for r in REAGENTS.values():
        if r.name in key or key in r.name:
            return r
    if "bicarbonate" in key or "soda" in key:
        return REAGENTS["baking soda"]
    if "cabbage" in key or "indicator" in key:
        return REAGENTS["red cabbage powder"]
    return None


# Rates. Unstirred, a spoonful at the bottom of a cup takes a minute or so;
# stirred, a few seconds. The reaction itself is fast once both are in water.
TAU_DISSOLVE_STILL = 45.0          # s
TAU_DISSOLVE_STIRRED = 2.0         # s
TAU_REACT = 1.5                    # s
TAU_FOAM = 6.0                     # s, how long the fizz's foam stands
FOAM_ML_PER_EQ = 1500.0            # foam drawn per equivalent neutralised
STIR_SPEED = 0.01                  # m/s of the rod tip that counts as stirring
POUR_RATE_ML_S = 80.0              # how fast a lip runs once over-full


# ------------------------------------------------------------------ colour

#: Red cabbage (anthocyanin) indicator: pH -> colour of a strong solution.
CABBAGE_PALETTE = [
    (1.0, (0.86, 0.06, 0.16)),     # red
    (3.0, (0.86, 0.18, 0.40)),     # pink-red
    (5.0, (0.62, 0.18, 0.56)),     # magenta
    (6.5, (0.44, 0.17, 0.60)),     # purple
    (7.5, (0.32, 0.22, 0.68)),     # violet
    (8.5, (0.16, 0.30, 0.78)),     # blue
    (10.0, (0.10, 0.55, 0.58)),    # blue-green
    (12.0, (0.38, 0.70, 0.20)),    # green
    (13.5, (0.88, 0.84, 0.20)),    # yellow
]
# Clear water, tinted enough that the cameras and the scene agent can see a
# few millimetres of it in a clear cup.
WATER_RGBA = (0.62, 0.80, 0.98, 0.50)
#: g of indicator per ml at which the colour is ~63% of full strength.
INDICATOR_STRENGTH = 0.0015


def cabbage_rgb(ph: float) -> np.ndarray:
    pts = CABBAGE_PALETTE
    if ph <= pts[0][0]:
        return np.asarray(pts[0][1], float)
    for (p0, c0), (p1, c1) in zip(pts, pts[1:]):
        if ph <= p1:
            t = (ph - p0) / (p1 - p0)
            return (1 - t) * np.asarray(c0, float) + t * np.asarray(c1, float)
    return np.asarray(pts[-1][1], float)


def colour_word(ph: Optional[float]) -> str:
    if ph is None:
        return ""
    for limit, word in ((2.5, "red"), (4.5, "pink"), (6.0, "magenta"), (7.2, "purple"),
                        (8.0, "violet"), (9.5, "blue"), (11.0, "blue-green"),
                        (13.0, "green")):
        if ph < limit:
            return word
    return "yellow"


# ---------------------------------------------------------------- contents

@dataclass
class Contents:
    """What one container holds."""

    water_ml: float = 0.0
    solid_g: Dict[str, float] = field(default_factory=dict)       # undissolved
    dissolved_g: Dict[str, float] = field(default_factory=dict)
    salt_eq: float = 0.0           # acid neutralised so far (as sodium citrate)
    foam_ml: float = 0.0
    fizzed_eq: float = 0.0         # all the CO2 this container has given off
    stirred_s: float = 0.0
    spilled_in_ml: float = 0.0     # liquid that has arrived from elsewhere

    def copy(self) -> "Contents":
        return Contents(self.water_ml, dict(self.solid_g), dict(self.dissolved_g),
                        self.salt_eq, self.foam_ml, self.fizzed_eq, self.stirred_s,
                        self.spilled_in_ml)

    # -- chemistry ----------------------------------------------------------
    def _eq(self, kind: str, solid: bool = False) -> float:
        out = 0.0
        for name, g in (self.solid_g if solid else self.dissolved_g).items():
            r = REAGENTS.get(name)
            if r is not None and r.kind == kind and r.molar_mass:
                out += g / r.molar_mass * r.equivalents
        return out

    def ph(self) -> Optional[float]:
        """
        A rough pH from what is dissolved. None for a dry container.

        Citric acid is triprotic (pKa 3.13, 4.76, 6.40); between none and all
        of it neutralised, the pH is interpolated through those pKas by the
        fraction neutralised -- a titration curve, flattened. Unneutralised,
        it is the weak-acid formula on the first pKa. Excess bicarbonate holds
        a solution near 8.3. Sodium citrate alone is mildly basic.
        """
        if self.water_ml <= 1e-6:
            return None
        # Whatever acid and base are both still dissolved are as good as
        # reacted -- the reaction is fast, and only its bookkeeping lags.
        acid_all = self._eq("acid")
        base_all = self._eq("base")
        both = min(acid_all, base_all)
        acid, base = acid_all - both, base_all - both
        salt = self.salt_eq + both
        litres = self.water_ml / 1000.0
        if acid > 1e-7:
            total = acid + salt
            f = salt / total
            molar = acid / 3.0 / litres
            weak = 0.5 * (3.13 - math.log10(max(molar, 1e-6)))
            weak = min(max(weak, 1.2), 3.13)
            knots = [(0.0, weak), (1 / 6, 3.13), (0.5, 4.76), (5 / 6, 6.40), (1.0, 7.0)]
            for (f0, p0), (f1, p1) in zip(knots, knots[1:]):
                if f <= f1:
                    return p0 + (p1 - p0) * (f - f0) / (f1 - f0)
            return 7.0
        if base > 1e-7:
            return 8.3
        if salt > 1e-7:
            return 7.6
        return 7.0

    def indicator_g(self) -> float:
        return sum(g for n, g in self.dissolved_g.items()
                   if REAGENTS.get(n) and REAGENTS[n].kind == "indicator")

    def liquid_rgba(self) -> Tuple[float, float, float, float]:
        if self.water_ml <= 1e-6:
            return (*WATER_RGBA[:3], 0.0)
        ph = self.ph()
        conc = self.indicator_g() / self.water_ml
        s = 1.0 - math.exp(-conc / INDICATOR_STRENGTH)
        rgb = (1 - s) * np.asarray(WATER_RGBA[:3]) + s * cabbage_rgb(ph if ph is not None else 7.0)
        # Undissolved powder stirred up clouds it a little.
        cloud = min(0.35, 0.08 * sum(self.solid_g.values()) / max(self.water_ml / 10.0, 1e-3))
        rgb = (1 - cloud) * rgb + cloud * np.array([0.95, 0.95, 0.93])
        alpha = WATER_RGBA[3] + 0.6 * s + cloud
        return (float(rgb[0]), float(rgb[1]), float(rgb[2]), float(min(alpha, 0.95)))

    def describe(self) -> str:
        bits = []
        if self.water_ml > 0.05:
            bits.append(f"{self.water_ml:.1f} ml of liquid")
        for name, g in sorted(self.dissolved_g.items()):
            if g > 0.005:
                bits.append(f"{g:.2f} g {name} dissolved")
        for name, g in sorted(self.solid_g.items()):
            if g > 0.005:
                bits.append(f"{g:.2f} g {name} undissolved")
        if self.salt_eq > 1e-5:
            bits.append(f"{self.salt_eq * 1000:.1f} meq neutralised")
        ph = self.ph() if self.water_ml > 0.05 else None
        if ph is not None:
            word = colour_word(ph) if self.indicator_g() > 1e-4 else "colourless"
            bits.append(f"pH ~{ph:.1f} ({word})")
        if self.foam_ml > 0.5:
            bits.append(f"fizzing ({self.foam_ml:.0f} ml of foam)")
        return ", ".join(bits) or "empty"


# ------------------------------------------------------------- liquid shape

def _holdable_table(inner_radius: float, depth: float, n_r: int = 14,
                    n_phi: int = 28, n_z: int = 24) -> np.ndarray:
    """
    ml a straight open cylinder keeps at each whole degree of tilt, 0..180.

    The level surface of what stays passes through the lowest point of the
    rim; tilted by theta, that point sits ``depth*cos - R*sin`` up the world
    vertical, measured from the floor's centre. A lattice of equal-volume
    cells gives the volume under it -- a closed form exists, but this is
    computed once per container and is obviously right.
    """
    rr = inner_radius * np.sqrt((np.arange(n_r) + 0.5) / n_r)     # equal-area rings
    ph = 2 * np.pi * (np.arange(n_phi) + 0.5) / n_phi
    zz = depth * (np.arange(n_z) + 0.5) / n_z
    R, P, Z = np.meshgrid(rr, ph, zz, indexing="ij")
    x, z = (R * np.cos(P)).ravel(), Z.ravel()
    full = math.pi * inner_radius ** 2 * depth * 1e6
    out = np.empty(181)
    for deg in range(181):
        t = math.radians(deg)
        up = (math.sin(t), math.cos(t))           # world up in the cup's x-z plane
        lip = depth * up[1] - inner_radius * up[0]
        out[deg] = float((x * up[0] + z * up[1] <= lip).mean()) * full
    return out


@dataclass
class _Vessel:
    """Per-container geometry and draw handles."""

    name: str
    body: int
    inner_radius: float
    floor_z: float                 # inside floor, body frame
    rim_z: float                   # top of the wall, body frame
    holdable: np.ndarray           # ml by whole degree of tilt
    liquid_geom: int
    foam_geom: int
    heap_geom: int
    bed: bool                      # a dish with a drawn powder bed
    capacity_ml: float = 0.0


class Lab:
    """
    Keeps the books on every container's contents and draws them.

    Hooks the arm's physics step, like ScoopTally and film_sim_skill's
    ContactWatch, so it follows the motion as it actually ran.
    """

    EVERY = 5                      # physics steps per update (10 ms)

    def __init__(self, cell, verbose: bool = True):
        self.cell = cell
        self.scene = cell.scene
        self.model, self.data = cell.scene.model, cell.scene.data
        self.verbose = verbose
        self.vessels: Dict[str, _Vessel] = {}
        self.contents: Dict[str, Contents] = {}
        self.spilled_ml = 0.0                 # liquid on the table
        self.spilled_powder_g: Dict[str, float] = {}
        self.events: List[dict] = []
        self._tick = 0
        self._last_t = float(self.data.time)
        self._original = None
        self._stir_prev: Dict[str, np.ndarray] = {}

        for prop in self.scene.bench.props:
            if prop.kind != "container" or prop.mesh:
                continue

            def gid(suffix):
                return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM,
                                         f"{prop.body}_{suffix}")
            floor = -prop.height / 2 + prop.floor_height
            depth = prop.height - prop.floor_height
            self.vessels[prop.name] = _Vessel(
                name=prop.name, body=self.scene.prop_bodies[prop.name],
                inner_radius=prop.inner_radius, floor_z=floor, rim_z=prop.height / 2,
                holdable=_holdable_table(prop.inner_radius, depth),
                liquid_geom=gid("liquid"), foam_geom=gid("foam"), heap_geom=gid("received"),
                bed=prop.powder_level > 0,
                capacity_ml=math.pi * prop.inner_radius ** 2 * depth * 1e6,
            )

        self.tallies: Dict[str, ScoopTally] = {}
        colors = {k: r.rgba for k, r in REAGENTS.items()}
        for prop in self.scene.bench.props:
            if prop.mesh and prop.bowl_size is not None:
                self.tallies[prop.name] = ScoopTally(
                    cell, prop, beds_from_bench(self.scene),
                    bulk_density=0.9,
                    on_land=self._powder_landed, colors=colors,
                    receivers=openings_from_bench(self.scene))
                self.tallies[prop.name].on_take = self._powder_taken
        self.stirrers = [p for p in self.scene.bench.props if p.kind == "stirrer"]
        self.reset()

    # ------------------------------------------------------------ lifecycle
    def initial_contents(self, prop) -> Contents:
        c = Contents(water_ml=float(prop.liquid_ml))
        r = reagent(prop.reagent)
        if prop.powder_level > 0 and r is not None:
            v = self.vessels[prop.name]
            bed_ml = math.pi * v.inner_radius ** 2 * prop.powder_level * 1e6
            c.solid_g[r.name] = bed_ml * r.bulk_density
        return c

    def reset(self):
        """Every container back to what it held at the start."""
        self.contents = {}
        for prop in self.scene.bench.props:
            if prop.name in self.vessels:
                self.contents[prop.name] = self.initial_contents(prop)
        self.spilled_ml = 0.0
        self.spilled_powder_g = {}
        self.events = []
        self._snap_mark = 0
        for t in self.tallies.values():
            t.clear()
            t.landed = {k: 0.0 for k in t.landed}
            t.lost = 0.0
            t.pours = []
            t.peak = 0.0
        self._stir_prev = {}
        self._last_t = float(self.data.time)
        self._draw_all()

    def start(self):
        """Hook the arm's physics step."""
        arm = self.cell.arm
        if self._original is not None:
            return
        self._original = arm._step

        def stepped(n=1):
            for _ in range(n):
                self._original(1)
                self._tick += 1
                if self._tick % self.EVERY == 0:
                    self.update()

        arm._step = stepped
        for t in self.tallies.values():
            t._prev = t._patch_world()

    def stop(self):
        if self._original is not None:
            self.cell.arm._step = self._original
            self._original = None

    # --------------------------------------------------------------- update
    def update(self):
        now = float(self.data.time)
        dt = max(now - self._last_t, 0.0)
        self._last_t = now
        for t in self.tallies.values():
            t.update()
        stirred = self._stirred(dt)
        for name, v in self.vessels.items():
            self._pour_over(v, dt)
        for name, c in self.contents.items():
            if name in stirred:
                c.stirred_s += dt
            self._dissolve(name, c, dt, name in stirred)
            self._react(name, c, dt)
        self._draw_all()

    def _pose(self, v: _Vessel):
        return self.data.xpos[v.body], self.data.xmat[v.body].reshape(3, 3)

    def _surface_world_z(self, v: _Vessel, c: Contents) -> float:
        pos, mat = self._pose(v)
        h = c.water_ml * 1e-6 / (math.pi * v.inner_radius ** 2)
        return float(pos[2] + mat[2, 2] * (v.floor_z + h))

    def _stirred(self, dt: float) -> set:
        """Containers a stirrer's rod tip is moving in, below the surface."""
        out = set()
        for prop in self.stirrers:
            bid = self.scene.prop_bodies[prop.name]
            tip = (self.data.xpos[bid] + self.data.xmat[bid].reshape(3, 3)
                   @ np.array([0.0, 0.0, -prop.tool_length]))
            prev = self._stir_prev.get(prop.name)
            self._stir_prev[prop.name] = tip.copy()
            if prev is None or dt <= 0:
                continue
            speed = float(np.linalg.norm(tip - prev)) / dt
            if speed < STIR_SPEED:
                continue
            for name, v in self.vessels.items():
                c = self.contents[name]
                if c.water_ml <= 0.05:
                    continue
                pos, _ = self._pose(v)
                if (np.linalg.norm(tip[:2] - pos[:2]) < v.inner_radius
                        and tip[2] < self._surface_world_z(v, c) + 0.002):
                    out.add(name)
        return out

    # -- liquid ---------------------------------------------------------------
    def _pour_over(self, v: _Vessel, dt: float):
        c = self.contents[v.name]
        if c.water_ml <= 1e-4:
            return
        pos, mat = self._pose(v)
        cos_t = float(np.clip(mat[2, 2], -1.0, 1.0))
        tilt = math.degrees(math.acos(cos_t))
        keep = float(np.interp(tilt, np.arange(181), v.holdable))
        excess = c.water_ml - keep
        if excess <= 1e-4:
            return
        out_ml = min(excess, POUR_RATE_ML_S * max(dt, 1e-3))
        # The lowest point of the rim, which is where it runs over.
        up = mat[2, :]                                   # world up in the body frame
        side = np.array([up[0], up[1], 0.0])
        n = float(np.linalg.norm(side))
        rim_local = np.array([0.0, 0.0, v.rim_z])
        if n > 1e-9:
            rim_local[:2] = -side[:2] / n * v.inner_radius
        lip = pos + mat @ rim_local
        into = self._opening_under(lip, exclude=v.name)
        moved = self._take_liquid(c, out_ml)
        if into is None:
            self.spilled_ml += out_ml
            self._event(("spill", v.name), "{ml:.1f} ml ran out of the " + v.name
                        + f" onto the table near ({lip[0]:.2f}, {lip[1]:.2f})", ml=out_ml)
        else:
            dst = self.contents[into]
            dst.water_ml += out_ml
            dst.spilled_in_ml += out_ml
            for k, g in moved["dissolved"].items():
                dst.dissolved_g[k] = dst.dissolved_g.get(k, 0.0) + g
            dst.salt_eq += moved["salt"]
            self._event(("pour", v.name, into),
                        "{ml:.1f} ml poured from the " + v.name + " into the " + into,
                        ml=out_ml)

    @staticmethod
    def _take_liquid(c: Contents, ml: float) -> dict:
        """Remove ``ml`` of solution, carrying its dissolved share with it."""
        f = min(1.0, ml / c.water_ml) if c.water_ml > 0 else 0.0
        moved = {"dissolved": {k: g * f for k, g in c.dissolved_g.items()},
                 "salt": c.salt_eq * f}
        c.water_ml -= ml
        for k in c.dissolved_g:
            c.dissolved_g[k] *= (1 - f)
        c.salt_eq *= (1 - f)
        if c.water_ml < 1e-4:
            c.water_ml = 0.0
        return moved

    def _opening_under(self, point: np.ndarray, exclude: str = "") -> Optional[str]:
        """The container whose opening is straight below ``point``, if any."""
        best, best_z = None, -1e9
        for name, v in self.vessels.items():
            if name == exclude:
                continue
            pos, mat = self._pose(v)
            if mat[2, 2] < 0.8:                       # tipped itself: not an opening
                continue
            rim_z = float(pos[2] + mat[2, 2] * v.rim_z)
            if rim_z > point[2] + 1e-3:
                continue
            if np.linalg.norm(point[:2] - pos[:2]) < v.inner_radius and rim_z > best_z:
                best, best_z = name, rim_z
        return best

    # -- powder ---------------------------------------------------------------
    def _powder_taken(self, bed_name: str, reagent_name: str, m3: float):
        c = self.contents.get(bed_name)
        r = REAGENTS.get(reagent_name)
        if c is None or r is None:
            return
        c.solid_g[r.name] = max(0.0, c.solid_g.get(r.name, 0.0) - m3 * 1e6 * r.bulk_density)

    def _powder_landed(self, into: Optional[str], parts: Dict[str, float]):
        grams = {}
        for k, m3 in parts.items():
            r = REAGENTS.get(k)
            grams[k] = m3 * 1e6 * (r.bulk_density if r else 0.9)
        if into is None or into not in self.contents:
            for k, g in grams.items():
                self.spilled_powder_g[k] = self.spilled_powder_g.get(k, 0.0) + g
            self._event(("powder", None), "{g} fell onto the table", grams=grams)
            return
        c = self.contents[into]
        for k, g in grams.items():
            c.solid_g[k] = c.solid_g.get(k, 0.0) + g
        self._event(("powder", into), "{g} landed in the " + into, grams=grams)

    def _dissolve(self, name: str, c: Contents, dt: float, stirred: bool):
        if c.water_ml <= 1e-4 or dt <= 0 or self.vessels[name].bed:
            return
        tau = TAU_DISSOLVE_STIRRED if stirred else TAU_DISSOLVE_STILL
        frac = 1.0 - math.exp(-dt / tau)
        for k in list(c.solid_g):
            g = c.solid_g[k]
            if g <= 0:
                continue
            r = REAGENTS.get(k)
            room = (r.solubility * c.water_ml if r else 1e9) - c.dissolved_g.get(k, 0.0)
            move = min(g * frac, max(room, 0.0))
            if move > 0:
                c.solid_g[k] = g - move
                c.dissolved_g[k] = c.dissolved_g.get(k, 0.0) + move

    def _react(self, name: str, c: Contents, dt: float):
        if dt > 0:
            c.foam_ml *= math.exp(-dt / TAU_FOAM)
        if c.water_ml <= 1e-4 or dt <= 0:
            return
        acid = c._eq("acid")
        # Undissolved bicarbonate fizzes on contact with an acid solution too.
        base = c._eq("base") + c._eq("base", solid=True)
        if acid <= 1e-9 or base <= 1e-9:
            return
        r_eq = min(acid, base) * (1.0 - math.exp(-dt / TAU_REACT))
        if r_eq <= 1e-12:
            return
        acid_r, base_r = REAGENTS["citric acid"], REAGENTS["baking soda"]
        c.dissolved_g["citric acid"] = max(
            0.0, c.dissolved_g.get("citric acid", 0.0) - r_eq / acid_r.equivalents * acid_r.molar_mass)
        need = r_eq * base_r.molar_mass
        for pool in (c.dissolved_g, c.solid_g):
            have = pool.get("baking soda", 0.0)
            take = min(have, need)
            pool["baking soda"] = have - take
            need -= take
        c.salt_eq += r_eq
        c.fizzed_eq += r_eq
        c.foam_ml += r_eq * FOAM_ML_PER_EQ
        self._event(("fizz", name), "the " + name + " fizzed ({meq:.1f} meq neutralised)",
                    meq=r_eq * 1000)

    # ----------------------------------------------------------------- draw
    def _draw_all(self):
        m = self.model
        for name, v in self.vessels.items():
            c = self.contents[name]
            area = math.pi * v.inner_radius ** 2
            room = v.rim_z - v.floor_z
            h = min(c.water_ml * 1e-6 / area, room)
            if v.liquid_geom >= 0:
                rgba = c.liquid_rgba()
                hh = max(h, 1e-4)
                m.geom_size[v.liquid_geom, 1] = hh / 2
                m.geom_pos[v.liquid_geom, 2] = hh / 2
                m.geom_rbound[v.liquid_geom] = float(np.hypot(v.inner_radius, hh / 2))
                m.geom_rgba[v.liquid_geom] = rgba if h > 1e-4 else (*rgba[:3], 0.0)
            if v.foam_geom >= 0:
                f = min(c.foam_ml * 1e-6 / area, max(room - h, 0.0))
                ff = max(f, 1e-4)
                m.geom_size[v.foam_geom, 1] = ff / 2
                m.geom_pos[v.foam_geom, 2] = h + ff / 2
                m.geom_rbound[v.foam_geom] = float(np.hypot(v.inner_radius, ff / 2))
                tint = c.liquid_rgba()
                m.geom_rgba[v.foam_geom] = (0.5 + 0.5 * tint[0], 0.5 + 0.5 * tint[1],
                                            0.5 + 0.5 * tint[2],
                                            0.85 if f > 2e-4 else 0.0)
            if v.heap_geom >= 0 and not v.bed:
                self._draw_heap(v, c)

    def _draw_heap(self, v: _Vessel, c: Contents, radius: float = 0.012):
        m = self.model
        ml = sum(g / (REAGENTS[k].bulk_density if k in REAGENTS else 0.9)
                 for k, g in c.solid_g.items())
        if ml <= 1e-3:
            m.geom_rgba[v.heap_geom, 3] = 0.0
            return
        rad = min(max(radius, 0.6 * v.inner_radius), 0.85 * v.inner_radius)
        h = max(ml * 1e-6 / (math.pi * rad ** 2), 1e-4)
        m.geom_size[v.heap_geom] = [rad, h / 2, 0]
        m.geom_pos[v.heap_geom] = [0.0, 0.0, h / 2]
        m.geom_rbound[v.heap_geom] = float(np.hypot(rad, h / 2))
        total = sum(c.solid_g.values())
        rgb = sum(np.asarray(REAGENTS[k].rgba[:3] if k in REAGENTS else (0.97, 0.96, 0.9))
                  * g for k, g in c.solid_g.items()) / max(total, 1e-12)
        m.geom_rgba[v.heap_geom] = (*[0.85 * x for x in rgb], 1.0)

    # ------------------------------------------------------------ reporting
    def _event(self, key, fmt: str, ml: float = 0.0, meq: float = 0.0,
               grams: Optional[Dict[str, float]] = None):
        """
        Note what happened, merged with the last note if it is more of the
        same -- a pour is hundreds of updates, and is one event.
        """
        last = self.events[-1] if self.events else None
        # Never into an event from before the latest snapshot: that one has
        # already been reported, and what is added to it now would not be.
        if len(self.events) <= getattr(self, "_snap_mark", 0):
            last = None
        if last is None or last["key"] != key:
            last = {"key": key, "fmt": fmt, "ml": 0.0, "meq": 0.0, "grams": {}}
            self.events.append(last)
        last["ml"] += ml
        last["meq"] += meq
        for k, g in (grams or {}).items():
            last["grams"][k] = last["grams"].get(k, 0.0) + g

    @staticmethod
    def _event_text(e: dict) -> str:
        g = ", ".join(f"{v:.2f} g of {k}" for k, v in e["grams"].items())
        return e["fmt"].format(ml=e["ml"], meq=e["meq"], g=g)

    def snapshot(self) -> dict:
        self._snap_mark = len(self.events)
        return {"contents": {k: c.copy() for k, c in self.contents.items()},
                "events": len(self.events),
                "loads": {n: {k: v * 1e6 for k, v in t.load.items()} for n, t in self.tallies.items()}}

    def changes_since(self, snap: dict) -> List[str]:
        """What happened since ``snap``, then each container that changed."""
        before = snap["contents"]
        merged: Dict[tuple, dict] = {}
        for e in self.events[snap["events"]:]:
            m = merged.setdefault(e["key"], {"key": e["key"], "fmt": e["fmt"], "ml": 0.0,
                                             "meq": 0.0, "grams": {}})
            m["ml"] += e["ml"]
            m["meq"] += e["meq"]
            for k, g in e["grams"].items():
                m["grams"][k] = m["grams"].get(k, 0.0) + g
        out = [self._event_text(e) for e in merged.values()
               if e["ml"] > 0.05 or e["meq"] > 0.05 or any(v > 0.005 for v in e["grams"].values())]
        touched = set()
        for key in merged:
            touched.update(k for k in key[1:] if isinstance(k, str))
        for name, c in self.contents.items():
            b = before.get(name)
            if b is None:
                continue
            if self.vessels[name].bed:
                for k, g in b.solid_g.items():
                    if g - c.solid_g.get(k, 0.0) > 0.005:
                        out.append(f"{name}: {g - c.solid_g.get(k, 0.0):.2f} g of {k} taken out")
                continue
            # A container this skill put something in or took something out
            # of, or whose indicator changed colour or that finished
            # dissolving -- not one that merely went on dissolving meanwhile.
            colour = lambda x: colour_word(x.ph()) if x.indicator_g() > 1e-4 and x.water_ml > 0.05 else ""
            undissolved = lambda x: sum(x.solid_g.values()) > 0.005
            if (name in touched or colour(c) != colour(b)
                    or (undissolved(b) and not undissolved(c) and c.water_ml > 0.05)
                    or c.stirred_s > b.stirred_s):
                out.append(f"{name}: {c.describe()}")
        held = set(getattr(self.cell.arm, "_attached", {}))
        for name, t in self.tallies.items():
            load = {k: v * 1e6 for k, v in t.load.items() if v > 1e-9}
            was = snap.get("loads", {}).get(name, {})
            changed = any(abs(load.get(k, 0) - was.get(k, 0)) > 0.005 for k in set(load) | set(was))
            if load and (changed or name in held):
                out.append(f"{name} is carrying " + ", ".join(
                    f"{ml:.2f} ml ({ml * (REAGENTS[k].bulk_density if k in REAGENTS else 0.9):.2f} g) {k}"
                    for k, ml in load.items()))
        return out

    def summary(self) -> str:
        lines = []
        for name, c in self.contents.items():
            if self.vessels[name].bed:
                continue
            if c.water_ml > 0.05 or any(g > 0.005 for g in c.solid_g.values()):
                lines.append(f"  {name}: {c.describe()}")
        if self.spilled_ml > 0.05:
            lines.append(f"  on the table: {self.spilled_ml:.1f} ml of liquid")
        for k, g in self.spilled_powder_g.items():
            if g > 0.005:
                lines.append(f"  on the table: {g:.2f} g of {k}")
        return "\n".join(lines) or "  (every container empty)"

    def state(self) -> dict:
        """JSON-able contents, for a run's record."""
        out = {
            name: {"liquid_ml": round(c.water_ml, 2),
                   "dissolved_g": {k: round(g, 3) for k, g in c.dissolved_g.items() if g > 1e-4},
                   "solid_g": {k: round(g, 3) for k, g in c.solid_g.items() if g > 1e-4},
                   "neutralised_meq": round(c.salt_eq * 1000, 2),
                   "ph": None if c.ph() is None else round(c.ph(), 2),
                   "colour": colour_word(c.ph()) if c.indicator_g() > 1e-4 else None}
            for name, c in self.contents.items() if not self.vessels[name].bed
        }
        out["_table"] = {"liquid_ml": round(self.spilled_ml, 2),
                         "powder_g": {k: round(g, 3) for k, g in self.spilled_powder_g.items()}}
        return out
