"""
The bench: what sits on the table in front of the simulated arm.

Props are described semantically (a cup here, a spoon there) and compiled into
MJCF by :mod:`robochem.sim.scene`. Names are the ones the skills already pass to
``VisionSystem.locate`` -- "plastic beaker", "clear cup a", "larger spoon" --
so a simulated run uses exactly the same parameters as a real one.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import List, Optional, Tuple

import numpy as np

# The table top. The Panda is bolted to it, so its base plane and the table
# surface are both z=0 in world coordinates, matching the real cage where
# object centroids land just above zero.
TABLE_Z = 0.0


@dataclass
class Prop:
    """One object on the bench."""

    name: str                       # the query the skills use
    kind: str = "container"         # container | rod | plate
    pos: Tuple[float, float] = (0.5, 0.0)
    radius: float = 0.035
    height: float = 0.09
    wall: float = 0.0015
    rgba: Tuple[float, float, float, float] = (0.9, 0.9, 0.9, 1.0)
    mass: float = 0.02
    label: Optional[str] = None     # reagent written on the paper under a cup
    aliases: Tuple[str, ...] = ()   # other names a skill might ask for it by
    fill: int = 0                   # granules to drop in at reset
    fill_rgba: Tuple[float, float, float, float] = (0.95, 0.95, 0.9, 1.0)
    grain_radius: float = 0.0035   # coarse grains cost fewer bodies for a given bed
    # Height of a powder bed above the container's INSIDE floor, metres. Not
    # simulated as particles: drawn as a solid-looking fill (no contacts) and
    # used by robochem.sim.powder to estimate what a scoop collects. Granules
    # (``fill``) are the physical alternative and only appear with granules=True.
    powder_level: float = 0.0
    # Which reagent the bed is (robochem.sim.lab.REAGENTS); None means the
    # label's. What a scoop takes out of this dish carries this name.
    powder: Optional[str] = None
    # Water in the container at reset, millilitres. Drawn and poured by
    # robochem.sim.lab, never simulated as fluid.
    liquid_ml: float = 0.0
    # The paper under a labelled container, half-extents in metres.
    label_half: Tuple[float, float] = (0.055, 0.04)
    # A container's base, bottom to inside floor, metres. None keeps the old
    # model, where the base is as thick as the wall panels (2 * wall).
    floor_thickness: Optional[float] = None
    static: bool = False            # True = welded to the table, never moves

    # A prop whose real shape matters to perception carries its CAD instead of
    # being approximated. The mesh is what the cameras see; MuJoCo would only
    # collide with its convex hull, which fills an open bowl solid, so contact
    # still comes from the primitives below.
    mesh: Optional[str] = None      # STL path, relative to the repo root
    mesh_scale: float = 1.0         # 0.001 for a CAD file authored in mm
    mesh_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    # wxyz. MuJoCo re-frames an imported mesh onto its principal axes of
    # inertia, which is not the pose the part is used in, so this puts it back.
    mesh_quat: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)

    # An open box the granules can actually sit in, built from five thin
    # panels. Half-extents and centre, in the body frame.
    bowl_size: Optional[Tuple[float, float, float]] = None
    bowl_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # A stirrer: a cube to grasp, on a rod that goes into the liquid. The body
    # origin is the cube's centre, so the rod hangs from -cube_half down to
    # -(cube_half + rod_length) and the tool reaches that far past the TCP.
    cube_half: Optional[float] = None
    rod_radius: float = 0.005
    rod_length: float = 0.035

    # A stirrer holder: a block with a bore the stirrer's rod drops into, so
    # the tool is always presented upright. Heights are above the body origin,
    # which sits on the table.
    # Name of the holder this prop starts seated in. Its rest height then
    # comes from the holder's top face rather than the table, so the two stay
    # in agreement if either part is re-measured.
    seated_in: Optional[str] = None

    bore_radius: Optional[float] = None
    bore_bottom: float = 0.010      # the rod cannot fall past this
    funnel_bottom: float = 0.066    # where the lead-in chamfer starts
    funnel_radius: float = 0.0145   # bore radius at the very top

    @property
    def body(self) -> str:
        """MuJoCo body name -- the semantic name is not XML-safe."""
        return "prop_" + "".join(c if c.isalnum() else "_" for c in self.name).lower()

    # A container's wall is a ring of panels 2 * wall thick, CENTRED on
    # ``radius`` (scene._add_container), so radius is the wall's mid-line:
    @property
    def inner_radius(self) -> float:
        """Inside radius of a container: the panels' inner face."""
        return self.radius - self.wall

    @property
    def outer_radius(self) -> float:
        """Outside radius of a container: the panels' outer face."""
        return self.radius + self.wall

    @property
    def floor_height(self) -> float:
        """Height of a container's inside floor above the table."""
        return self.floor_thickness if self.floor_thickness is not None else 2 * self.wall

    @property
    def reagent(self) -> Optional[str]:
        """The powder this container's bed is made of, if it has one."""
        if self.powder_level <= 0:
            return None
        return self.powder or self.label

    @property
    def grasp_offset(self) -> Tuple[float, float, float]:
        """
        Where the gripper is expected to close, relative to the body origin.

        Containers are grasped near the rim, a rod (spoon) across its handle.
        Used only to decide whether a closing gripper has caught this prop.
        """
        if self.kind in ("rod", "stirrer"):
            # Both put their body origin where the jaws close: the middle of
            # the spoon's handle, the centre of the stirrer's cube.
            return (0.0, 0.0, 0.0)
        return (0.0, 0.0, self.height * 0.75)

    @property
    def grasp_width(self) -> float:
        """Jaw opening that corresponds to holding this prop."""
        if self.kind == "stirrer":
            return 2 * (self.cube_half or 0.01)
        if self.kind == "rod":
            return 2 * self.wall
        return 2 * self.radius

    @property
    def tool_length(self) -> float:
        """
        How far the working tip reaches past the grasp point.

        For a stirrer this is what ``stir`` needs as ``tool_length``: the TCP
        sits at the cube's centre, the rod tip is this far below it, and the
        immersion depth is measured from the tip, not the wrist.
        """
        if self.kind == "stirrer":
            return (self.cube_half or 0.01) + self.rod_length
        return 0.0


def _powder_dish(name: str, pos, label: str, fill_rgba, **kw) -> Prop:
    """
    One of the bench's powder dishes, measured 2026-09-25: 91mm across the
    inside, 102mm across the outside, 29mm tall, an 8mm base. radius is the
    wall's mid-line (see Prop.inner_radius), so 45.5mm inside + 51.0mm outside
    -> radius 48.25, wall 2.75. The scoop's stroke is sized to the inside
    radius and its depth to the inside floor, so these are the numbers that
    matter.
    """
    args = dict(
        name=name, pos=pos, label=label,
        radius=0.04825, height=0.029, wall=0.00275, floor_thickness=0.008,
        rgba=(0.97, 0.97, 0.97, 1.0), mass=0.01,
        fill=90,           # granules, only with granules=True
        grain_radius=0.003,
        fill_rgba=fill_rgba,
        # 18mm of powder on the 8mm base: surface 3mm under the rim. Not
        # measured -- set it to the real fill. It decides whether a scoop
        # collects anything: the stroke keeps the tip 3mm off the floor and
        # the mouth sits ~10mm above the tip, so in this 21mm-deep dish 14mm
        # of powder gave 4% of a bowl, 16mm 41%, 18mm 100% (sim).
        powder_level=0.018,
    )
    args.update(kw)
    return Prop(**args)


def _clear_cup(name: str, pos, label: Optional[str] = None, liquid_ml: float = 0.0,
               **kw) -> Prop:
    """
    A thin-walled clear plastic cup, as the "A 10 ML WATER" / "B 10 ML WATER"
    cups on the real bench: 47.3 mm tall, 55.5 mm across the outside
    (measured by the user, 2026-10-03). The wall (1 mm) and the base (3 mm,
    standing for the raised foot) are NOT measured. A straight cylinder of the
    rim's diameter; the real cup tapers, so 10 ml stands 4.4 mm deep here and
    a little more in the real one. At a 1.5 mm base the stirrer's
    force-guarded descent went through the floor onto the table.
    """
    args = dict(
        name=name, pos=pos, label=label, liquid_ml=liquid_ml,
        # radius is the wall's mid-line: 27.75 mm outside - 0.5 mm half-wall.
        radius=0.02725, height=0.0473, wall=0.0005, floor_thickness=0.003,
        rgba=(0.86, 0.92, 0.97, 0.28), mass=0.006,
    )
    args.update(kw)
    return Prop(**args)


def default_bench() -> List[Prop]:
    """
    What the booklet's Magic Beaker (pages 12-13) needs and nothing else, laid
    out the way the real bench is: the three reagent dishes, three clear cups
    -- A and B with their 10 ml of water, C empty for the final mix -- and the
    beaker holding the indicator's 50 ml of water, all on handwritten labels,
    plus both scoops and the stirrer.

    The positions were annealed under :func:`layout_violations` -- whose
    sweeps were measured off the arm in sim -- then moved to the middle of
    the spot each may occupy, so a random layout has room to jitter, and
    checked by running the whole experiment; see progress.md (2026-10-03).
    """
    return [
        Prop(
            name="plastic beaker",
            pos=(0.395, 0.031),
            radius=0.035,
            height=0.095,
            wall=0.002,
            rgba=(0.75, 0.85, 0.95, 0.55),
            mass=0.03,
            fill=30,
            fill_rgba=(0.35, 0.65, 0.95, 1.0),
            # Water comes pre-measured, as on the real bench: the pour skill
            # tips until the level reaches the rim and cannot meter, so
            # "fill with 50 mL" is met by the beaker already holding it.
            label="50 ml water",
            liquid_ml=50.0,
        ),
        _powder_dish("citric acid cup", (0.42, -0.294), "citric acid",
                     (0.98, 0.98, 0.94, 1.0)),
        _powder_dish("baking soda cup", (0.449, 0.256), "baking soda",
                     (1.0, 1.0, 1.0, 1.0)),
        # The kit's indicator: a dark purple-magenta powder.
        _powder_dish("red cabbage powder cup", (0.48, -0.152), "red cabbage powder",
                     (0.72, 0.30, 0.62, 1.0)),
        _clear_cup("clear cup a", (0.349, -0.144), label="a 10 ml water", liquid_ml=10.0,
                   aliases=("cup a",)),
        _clear_cup("clear cup b", (0.313, 0.196), label="b 10 ml water", liquid_ml=10.0,
                   aliases=("cup b",)),
        _clear_cup("clear cup c", (0.301, -0.241),
                   aliases=("cup c", "empty clear cup", "empty cup")),
        # The lab's printed scoop, straight off spoon.stl: a 30mm flat handle,
        # a crank down, and a 27x20mm open bowl. Every number below is measured
        # from that file (authored in mm, hence mesh_scale), with the body
        # origin put at the middle of the handle -- where the jaws close.
        # It is the booklet's "big scoop".
        Prop(
            name="larger spoon",
            kind="rod",
            pos=(0.62, -0.03),
            mesh="spoon.stl",
            mesh_scale=0.001,
            mesh_pos=(-0.015, 0.0, 0.0035),
            radius=0.01025,    # bowl half-width
            height=0.030,      # handle length
            wall=0.004,        # handle half-width -> 8mm jaw width
            bowl_size=(0.01375, 0.01025, 0.00575),
            bowl_offset=(0.0408, 0.0, -0.0223),
            rgba=(0.88, 0.88, 0.90, 1.0),
            mass=0.01,
            aliases=("big scoop", "big spoon", "large spoon"),
        ),
        # The booklet's "small scoop". There is no printed one yet, so this is
        # PLACEHOLDER geometry: spoon.stl at 75%, every length scaled with it.
        # Replace it with the real part's CAD once it exists, and the
        # "smaller spoon" entry in skills/tool_geometry.py with it.
        Prop(
            name="smaller spoon",
            kind="rod",
            pos=(0.62, -0.12),
            mesh="spoon.stl",
            mesh_scale=0.00075,
            mesh_pos=(-0.01125, 0.0, 0.002625),
            radius=0.0076875,
            height=0.0225,
            wall=0.003,
            bowl_size=(0.0103125, 0.0076875, 0.0043125),
            bowl_offset=(0.0306, 0.0, -0.016725),
            rgba=(0.88, 0.88, 0.90, 1.0),
            mass=0.006,
            aliases=("small scoop", "small spoon", "smaller scoop"),
        ),
        # The holder the stirrer lives in: a 50mm block, 85mm tall, bored
        # 12mm for the rod with a chamfer at the top to catch it on the way in.
        # Welded to the bench -- it is a fixture, and a holder that skitters
        # when the rod goes in is not presenting anything upright.
        Prop(
            name="stirrer holder",
            kind="holder",
            pos=(0.62, 0.14),
            mesh="stirrer holder.stl",
            mesh_scale=0.001,
            mesh_quat=(0.70710678, 0.70710678, 0.0, 0.0),
            mesh_pos=(0.0, 0.0, 0.105),
            radius=0.025,           # outer half-width
            height=0.085,
            bore_radius=0.0075,     # 7.5mm against a 5mm rod: 2.5mm of slack
            bore_bottom=0.010,
            funnel_bottom=0.066,
            funnel_radius=0.0145,
            aliases=("holder", "stirrer stand", "stir holder", "tool holder"),
            rgba=(0.55, 0.57, 0.62, 1.0),
            static=True,
        ),
        # The printed stirrer, straight off "stirrer v1.stl": a 10mm rod 66mm
        # long under a 30mm grasp head, 96mm overall. It starts seated in the
        # holder -- head resting on the holder's top face, rod hanging in the
        # bore -- which is the pose the gripper takes it from.
        #
        # The two STLs are authored in one assembly frame, so both take the
        # same +90deg-about-X and their z coordinates line up directly: the
        # head's underside lands exactly on the holder's top face.
        Prop(
            name="stirrer",
            kind="stirrer",
            pos=(0.62, 0.14),       # same axis as the holder: it sits in it
            mesh="stirrer v1.stl",
            mesh_scale=0.001,
            mesh_quat=(0.70710678, 0.70710678, 0.0, 0.0),
            mesh_pos=(0.0, 0.0, 0.005),     # CoM -> head centre
            seated_in="stirrer holder",
            cube_half=0.015,
            rod_radius=0.005,
            rod_length=0.066,       # head underside to rod tip
            aliases=("stirring rod", "stir rod", "stirring stick",
                     "stir stick", "stirrer rod"),
            rgba=(0.30, 0.55, 0.85, 1.0),
            mass=0.015,
        ),
    ]


#: How far a randomised layout may move each container from its usual spot.
LAYOUT_JITTER = 0.06
#: Distance from the robot base a randomised container must stay within.
#: The outer bound is the dump's: pointing straight ahead the arm tips a bowl
#: less the further out the cup is (71 deg at 0.54 m, 86 at 0.50), and dump
#: refuses below 60 deg, which is about 0.56 m.
LAYOUT_REACH = (0.35, 0.56)
#: A tall container closer than this is too close for dump: straight ahead,
#: 10 mm over a 90 mm rim, the arm is folded tight. The paper cup at 0.358 m
#: failed after a scoop on the far side of the bench (the bowl stopped 22 mm
#: short); at 0.38 m it tipped to 79 degrees (sim, 2026-10-02).
LAYOUT_REACH_TALL = 0.38
#: How far forward of the base (world x) a container must stand. Closer in,
#: the arm folds too tight to hold the bowl still while dump tips it: over a
#: clear cup at x 0.231-0.257 the dump failed or the bowl drifted 16-18 mm;
#: from x 0.276 out it held within 0.4 mm, left or right, at any angle tried
#: (sim, 2026-10-03).
LAYOUT_MIN_X = 0.28
#: Clear space kept between two containers' bodies, so the jaws and a carried
#: tool fit between them. Labels are paper: they only must not overlap.
LAYOUT_GAP = 0.03
#: A container kept at least this far from the stirrer holder: pointing
#: straight ahead, dump's wrist reaches ~15 cm past its target, and at 16 cm
#: from the beaker it swept into the seated stirrer (2026-09-25).
LAYOUT_HOLDER_CLEARANCE = 0.12
#: How close a TALL container may stand to a dish being scooped from, centre
#: to centre. The hand is ~200 mm across its jaws while it scoops, and the
#: wrist reaches further toward the arm's own base. Measured in sim
#: (2026-10-02), centre distances: 144 mm hit (the beaker, in the old default
#: layout, scooping baking soda), 145 mm hit (paper cup), 161 mm hit (link7
#: on the beaker); the old default paper cup at 172 mm was clear.
LAYOUT_SCOOP_CLEARANCE = 0.18
#: What the arm sweeps while dump tips straight ahead over a target, measured
#: in sim (2026-10-02) from the links' collision geometry, in the target's
#: radial frame: (height above the target's rim the obstacle stands, metres)
#: -> (ahead from, ahead to, half-width). Nothing comes lower than ~26 mm
#: above the rim (the fingertips); a prop up to 15 mm above it keeps 10 mm.
#: Over a 52 mm cup, the parts below 0.10 m spanned -0.05..+0.17 ahead and
#: -0.06..+0.12 aside; below 0.13 m, -0.09..+0.27 and -0.13..+0.15 (link6 and
#: link7 reach out past the target). The obstacle's own radius is added.
#: Pour keeps every link above 0.13 m and needs no corridor of its own.
DUMP_SWEEP = ((0.015, None), (0.05, (-0.06, 0.18, 0.12)), (1.0, (-0.10, 0.28, 0.15)))
#: Taller than this counts as tall: the 29 mm dishes and the 52 mm clear cups
#: stay below the hand (see progress.md, 2026-10-02).
_TALL = 0.06


def _footprint(prop: Prop) -> float:
    """Radius of the circle a prop's BODY needs on the bench (labels apart)."""
    if prop.kind == "container":
        return prop.outer_radius
    if prop.kind == "rod" and prop.bowl_size is not None:
        # Handle middle to the far end of the bowl: a 70 mm scoop needs 55 mm.
        return float(prop.bowl_offset[0] + prop.bowl_size[0])
    return max(prop.radius, 0.02) * 1.45     # a square block's half-diagonal


def _standing_height(prop: Prop, all_props=()) -> float:
    """How far a prop stands above the table. A scoop's ``height`` is its
    handle's length, and it lies flat; a holder carries its stirrer's head."""
    if prop.kind == "rod":
        return 2 * prop.wall
    if prop.kind == "holder":
        for o in all_props:
            if o.seated_in == prop.name:
                return prop.height + 2 * (o.cube_half or 0.0)
    return prop.height


def _dump_box(target_height: float, obstacle_height: float):
    """The (ahead_from, ahead_to, half_width) the dumping arm sweeps at the
    obstacle's height, or None when it passes over it."""
    dz = obstacle_height - target_height
    for limit, box in DUMP_SWEEP:
        if dz <= limit:
            return box
    return DUMP_SWEEP[-1][1]


def _conflicts(p: Prop, xy, others, all_props) -> List[str]:
    """
    Why ``p`` cannot stand at ``xy`` among ``others``, empty if it can.

    The rules every layout keeps -- the default bench as well as every
    randomised one: within reach, bodies LAYOUT_GAP apart, labels not
    overlapping, clear of the stirrer holder, tall things out of the scooping
    hand's sweep, and nothing tall in the corridor a dump or pour reaches
    into beyond its target.
    """
    out = []
    r = float(np.hypot(*xy))
    if p.kind == "container" and not p.static and not (LAYOUT_REACH[0] <= r <= LAYOUT_REACH[1]):
        out.append(f"{p.name}: {r:.3f} m from the base, outside {LAYOUT_REACH}")
    if p.kind == "container" and not p.static and xy[0] < LAYOUT_MIN_X:
        out.append(f"{p.name}: x {xy[0]:.3f}, too close in for dump to hold "
                   f"the bowl still (needs {LAYOUT_MIN_X})")
    if (p.kind == "container" and not p.static and p.height > _TALL
            and r < LAYOUT_REACH_TALL):
        out.append(f"{p.name}: {r:.3f} m from the base, too close for a dump "
                   f"into something {p.height * 1000:.0f} mm tall")
    for other in others:
        if other.seated_in:            # sits in its holder: the holder's footprint
            continue
        if p.kind != "container" and other.kind != "container":
            continue                   # the tool row is laid out by hand
        d = float(np.hypot(xy[0] - other.pos[0], xy[1] - other.pos[1]))
        need = _footprint(p) + _footprint(other) + LAYOUT_GAP
        if d < need:
            out.append(f"{p.name}-{other.name}: {d:.3f} apart, bodies need {need:.3f}")
        if p.label and other.label:
            ha, hb = p.label_half, other.label_half
            if (abs(xy[0] - other.pos[0]) < ha[0] + hb[0]
                    and abs(xy[1] - other.pos[1]) < ha[1] + hb[1]):
                out.append(f"{p.name}-{other.name}: labels overlap")
        for a, b in ((p, other), (other, p)):
            if a.kind == "holder" and b.kind == "container" and d < LAYOUT_HOLDER_CLEARANCE:
                out.append(f"{b.name}: {d:.3f} from the {a.name}")
        # The scooping hand's sweep, whichever of the pair is the dish.
        # Measured (2026-10-02): below 0.10 m the arm spans -0.08..+0.14 ahead
        # of the dish and +-0.13 aside, so a 0.09 m cup needs ~0.18 m.
        for tall, dish in ((other, p), (p, other)):
            if (dish.powder_level > 0 and _standing_height(tall, all_props) > _TALL
                    and d < LAYOUT_SCOOP_CLEARANCE):
                out.append(f"{tall.name} {d:.3f} from the {dish.name} it would be "
                           f"scooped beside, under {LAYOUT_SCOOP_CLEARANCE}")
        # What the arm sweeps dumping into either of the pair. Any container
        # that is not a reagent dish can be a dump target.
        for target, obst, txy, oxy in ((p, other, xy, other.pos), (other, p, other.pos, xy)):
            if target.kind != "container" or target.powder_level > 0:
                continue
            box = _dump_box(target.height, _standing_height(obst, all_props))
            if box is None:
                continue
            u = np.asarray(txy, float) / max(np.hypot(*txy), 1e-9)
            rel = np.asarray(oxy, float) - np.asarray(txy, float)
            ahead = float(rel @ u)
            side = abs(float(rel @ np.array([-u[1], u[0]])))
            r_o = _footprint(obst)
            if box[0] - r_o < ahead < box[1] + r_o and side < box[2] + r_o:
                out.append(f"{obst.name} is in the arm's way dumping into the "
                           f"{target.name} ({ahead:+.3f} ahead, {side:.3f} aside)")
    return out


def layout_violations(bench: "Bench") -> List[str]:
    """Every rule the bench breaks, each pair once. Empty for a clean layout."""
    props = list(bench.props)
    out: List[str] = []
    for i, p in enumerate(props):
        later = props[i + 1:]
        found = _conflicts(p, p.pos, later, props)
        # A rule about p alone (reach) is reported by p's own pass only.
        out.extend(found)
    return list(dict.fromkeys(out))


def _swap_class(prop: Prop):
    """Containers that can trade places: the same body, so the same rules."""
    return ("dish" if prop.powder_level > 0 else "cup",
            round(prop.radius, 4), round(prop.height, 4), round(prop.wall, 4))


def randomize_layout(bench: "Bench", seed: int, jitter: float = LAYOUT_JITTER,
                     tries: int = 300, shuffles: int = 50,
                     visible=None) -> Tuple["Bench", List[str]]:
    """
    A copy of ``bench`` with its containers shuffled and moved a little.

    Only containers move -- cups, the beaker, the powder dishes; tools and the
    stirrer's holder keep their places, since the skills pick those up by
    memory or seat things in them.

    First, containers with the same body trade places at random: the three
    powder dishes among themselves, the four clear cups among themselves. That
    is what makes one bench different from the next -- which reagent is where
    -- and since a swapped pair has the same footprint and height, the layout
    rules hold as they did (labels can still collide, and such a shuffle is
    drawn again). With the whole kit on the bench every container sits close
    to its neighbours' limits, so jitter alone barely moves anything.

    Then each container is drawn from a disc of ``jitter`` around its spot,
    and redrawn until it breaks none of the rules in :func:`_conflicts` and,
    given ``visible(xy) -> bool``, the cameras still see it there. The same
    seed always gives the same bench.

    Returns (bench, notes); a note names any container that could not be
    placed and stayed where it was.
    """
    rng = np.random.default_rng(seed)
    props = list(bench.props)
    movable = [i for i, p in enumerate(props) if p.kind == "container" and not p.static]
    notes: List[str] = []

    groups: dict = {}
    for i in movable:
        groups.setdefault(_swap_class(props[i]), []).append(i)
    for members in groups.values():
        if len(members) < 2:
            continue
        spots = [props[i].pos for i in members]
        for _ in range(shuffles):
            order = rng.permutation(len(members))
            trial = list(props)
            for i, k in zip(members, order):
                trial[i] = replace(props[i], pos=spots[k])
            if not any(_conflicts(trial[i], trial[i].pos,
                                  [o for j, o in enumerate(trial) if j != i], trial)
                       for i in members):
                props = trial
                break

    for i in movable:
        p = props[i]
        # Against everything where it stands NOW -- the ones already moved and
        # the ones still to come at their usual spots -- so no later draw can
        # undo an earlier check.
        others = [o for k, o in enumerate(props) if k != i]
        chosen = None
        for _ in range(tries):
            ang = rng.uniform(0.0, 2 * np.pi)
            rad = jitter * np.sqrt(rng.uniform())
            # Rounded BEFORE it is checked, so the spot printed is the one tested.
            xy = (round(float(p.pos[0] + rad * np.cos(ang)), 4),
                  round(float(p.pos[1] + rad * np.sin(ang)), 4))
            if (not _conflicts(p, xy, others, props)
                    and (visible is None or visible(xy))):
                chosen = xy
                break
        if chosen is None:
            chosen = p.pos
            notes.append(f"{p.name} kept at {p.pos}: no clear spot within {jitter * 100:.0f} cm")
        props[i] = replace(p, pos=chosen)
    return replace(bench, props=props), notes


@dataclass
class Bench:
    """A bench layout plus the table it stands on."""

    props: List[Prop] = field(default_factory=default_bench)
    table_z: float = TABLE_Z
    table_rgba: Tuple[float, float, float, float] = (0.35, 0.36, 0.40, 1.0)

    def find(self, query: str) -> Optional[Prop]:
        """
        Resolve a skill's object name to a prop.

        Matching mirrors what the grounding service does loosely in the real
        stack: exact name, then containment either way, then best token
        overlap. A reagent name ("citric acid") resolves through the label.
        """
        q = query.strip().lower()
        if not q:
            return None

        for prop in self.props:
            if (prop.name.lower() == q or (prop.label or "").lower() == q
                    or q in {a.lower() for a in prop.aliases}):
                return prop

        for prop in self.props:
            hay = f"{prop.name} {prop.label or ''} {' '.join(prop.aliases)}".lower()
            if q in hay or prop.name.lower() in q:
                return prop

        tokens = set(q.split())
        best, best_score = None, 0
        for prop in self.props:
            hay = set(f"{prop.name} {prop.label or ''} "
                      f"{' '.join(prop.aliases)}".lower().split())
            score = len(tokens & hay)
            if score > best_score:
                best, best_score = prop, score
        return best

    def without(self, *names: str) -> "Bench":
        """A copy of this bench with the named props removed."""
        drop = {n.lower() for n in names}
        return replace(self, props=[p for p in self.props if p.name.lower() not in drop])
