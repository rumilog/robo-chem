"""
The bench: what sits on the table in front of the simulated arm.

Props are described semantically (a cup here, a spoon there) and compiled into
MJCF by :mod:`robochem.sim.scene`. Names are the ones the skills already pass to
``VisionSystem.locate`` -- "plastic beaker", "white paper cup", "larger spoon" --
so a simulated run uses exactly the same parameters as a real one.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import List, Optional, Tuple

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
    fill: int = 0                   # granules to drop in at reset
    fill_rgba: Tuple[float, float, float, float] = (0.95, 0.95, 0.9, 1.0)
    grain_radius: float = 0.0035   # coarse grains cost fewer bodies for a given bed
    static: bool = False            # True = welded to the table, never moves

    # A prop whose real shape matters to perception carries its CAD instead of
    # being approximated. The mesh is what the cameras see; MuJoCo would only
    # collide with its convex hull, which fills an open bowl solid, so contact
    # still comes from the primitives below.
    mesh: Optional[str] = None      # STL path, relative to the repo root
    mesh_scale: float = 1.0         # 0.001 for a CAD file authored in mm
    mesh_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    # An open box the granules can actually sit in, built from five thin
    # panels. Half-extents and centre, in the body frame.
    bowl_size: Optional[Tuple[float, float, float]] = None
    bowl_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def body(self) -> str:
        """MuJoCo body name -- the semantic name is not XML-safe."""
        return "prop_" + "".join(c if c.isalnum() else "_" for c in self.name).lower()

    @property
    def grasp_offset(self) -> Tuple[float, float, float]:
        """
        Where the gripper is expected to close, relative to the body origin.

        Containers are grasped near the rim, a rod (spoon) across its handle.
        Used only to decide whether a closing gripper has caught this prop.
        """
        if self.kind == "rod":
            return (0.0, 0.0, 0.0)
        return (0.0, 0.0, self.height * 0.75)

    @property
    def grasp_width(self) -> float:
        """Jaw opening that corresponds to holding this prop."""
        if self.kind == "rod":
            return 2 * self.wall
        return 2 * self.radius


def default_bench() -> List[Prop]:
    """
    The bench the validated skill commands in the README were tuned against:
    a beaker to pick and pour, two labelled reagent cups to scoop from, a
    target cup to pour into, and a spoon to pick up first.
    """
    return [
        Prop(
            name="plastic beaker",
            pos=(0.46, 0.14),
            radius=0.035,
            height=0.095,
            wall=0.002,
            rgba=(0.75, 0.85, 0.95, 0.55),
            mass=0.03,
            fill=30,
            fill_rgba=(0.35, 0.65, 0.95, 1.0),
        ),
        Prop(
            name="white paper cup",
            pos=(0.52, -0.16),
            radius=0.038,
            height=0.09,
            wall=0.0015,
            rgba=(0.97, 0.97, 0.97, 1.0),
            mass=0.01,
        ),
        Prop(
            name="citric acid cup",
            pos=(0.38, -0.26),
            radius=0.0345,     # 65mm inner diameter + a 2mm wall
            height=0.031,      # the lab's reagent cups are shallow dishes
            wall=0.002,
            rgba=(0.97, 0.97, 0.97, 1.0),
            mass=0.01,
            label="citric acid",
            fill=90,           # a bed to scoop from, not a scatter
            grain_radius=0.003,
            fill_rgba=(0.98, 0.98, 0.94, 1.0),
        ),
        Prop(
            name="baking soda cup",
            pos=(0.38, 0.26),
            radius=0.0345,
            height=0.031,
            wall=0.002,
            rgba=(0.97, 0.97, 0.97, 1.0),
            mass=0.01,
            label="baking soda",
            fill=90,
            grain_radius=0.003,
            fill_rgba=(1.0, 1.0, 1.0, 1.0),
        ),
        # The lab's printed scoop, straight off spoon.stl: a 30mm flat handle,
        # a crank down, and a 27x20mm open bowl. Every number below is measured
        # from that file (authored in mm, hence mesh_scale), with the body
        # origin put at the middle of the handle -- where the jaws close.
        Prop(
            name="larger spoon",
            kind="rod",
            pos=(0.58, -0.02),
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
        ),
    ]


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
            if prop.name.lower() == q or (prop.label or "").lower() == q:
                return prop

        for prop in self.props:
            hay = f"{prop.name} {prop.label or ''}".lower()
            if q in hay or prop.name.lower() in q:
                return prop

        tokens = set(q.split())
        best, best_score = None, 0
        for prop in self.props:
            hay = set(f"{prop.name} {prop.label or ''}".lower().split())
            score = len(tokens & hay)
            if score > best_score:
                best, best_score = prop, score
        return best

    def without(self, *names: str) -> "Bench":
        """A copy of this bench with the named props removed."""
        drop = {n.lower() for n in names}
        return replace(self, props=[p for p in self.props if p.name.lower() not in drop])
