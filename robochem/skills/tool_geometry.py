"""
CAD geometry of the tools the arm picks up, for the skills that need it.

A scoop's stroke and a dump's clearance are planned around the BOWL: how long,
wide and deep it is, and where it hangs below the jaws. None of that can be
seen once the tool is in the gripper -- the cage looks down, so the bowl's
underside is occluded and ``pick_up``'s measured offset is a lower bound in z --
and a planning model has no way to know it at all. So it is looked up here, by
the name the tool was picked up by, and ``SkillsExecutor`` fills in whatever a
call left out.

Numbers are from the part's CAD; the simulated bench builds the same part from
the same file (``robochem.sim.bench``), so sim and robot agree.
"""

from typing import Any, Dict, Optional

#: name -> geometry. Lengths in metres; tool_offset in the TOOL frame, from
#: the TCP to the bottom of the bowl with the jaws across the handle.
TOOLS: Dict[str, Dict[str, Any]] = {
    "larger spoon": {
        # The lab's printed scoop, spoon.stl: a 30 mm flat handle, a crank
        # down, and a 27.5 x 20.5 x 11.5 mm open bowl (outside).
        # The booklet's "big scoop".
        "aliases": ("large spoon", "spoon", "scoop", "white plastic tool",
                    "rectangular scoop", "big scoop", "big spoon"),
        "bowl_length": 0.0275,
        "bowl_width": 0.0205,
        "bowl_depth": 0.0115,
        # The bowl's circumscribed width: in a round dish its corners meet the
        # wall before its faces do.
        "tool_span": 0.0343,
        # How far the tool reaches back from the jaws: the handle.
        "tool_back_reach": 0.030,
        # x depends on where along the handle the jaws closed, so a measured
        # value wins for x and y. z does not: the handle is flat, so the bowl
        # hangs the same 29.4 mm below the TCP wherever it is gripped.
        "tool_offset": (0.0257, 0.0, 0.0294),
        # From the middle of the part's length to the bowl's centre (spoon.stl
        # spans -15.0..54.5 mm from the handle's middle; the bowl's centre is
        # at 40.8). With where pick_up saw the two ends, it gives x for the
        # grasp actually made (SkillsExecutor._offset_from_tip).
        "span_mid_to_bowl": 0.02105,
        # End to end along the handle, spoon.stl: -15.0..54.5 mm.
        "length": 0.0695,
        # Lies on the bench: it goes back on its pick site with a plain place.
        "home": (),
        "put_back": {},
    },
    "smaller spoon": {
        # The booklet's "small scoop". PLACEHOLDER: no printed part exists
        # yet, so this is the larger spoon at 75% (sim/bench.py builds the
        # same). Replace with the real part's CAD once there is one.
        "aliases": ("small scoop", "small spoon", "smaller scoop"),
        "bowl_length": 0.0206,
        "bowl_width": 0.0154,
        "bowl_depth": 0.0086,
        "tool_span": 0.0257,
        "tool_back_reach": 0.0225,
        # Measured in sim off the scaled part (bowl bottom 19.7 / 22.6 mm from
        # the TCP), corrected by the larger spoon's CAD-to-sim ratio.
        "tool_offset": (0.0208, 0.0, 0.0235),
        # Span -11.3..40.9 mm, bowl centre 30.6 (the scaled spoon.stl).
        "span_mid_to_bowl": 0.0158,
        "length": 0.0522,
        "home": (),
        "put_back": {},
    },
    "stirrer": {
        # The printed stirrer: a 30 mm cube on a 66 mm rod, kept standing in
        # its holder. Put back, it has to go straight down into the bore.
        "aliases": ("stirring rod", "stir rod", "stirring stick", "stir stick",
                    "stirrer rod", "white cube"),
        "home": ("stirrer holder", "holder", "tool holder"),
        # TCP to rod tip: 15 mm to the cube centre plus the 66 mm rod (CAD,
        # stirrer v1.stl). pick_up measures ~0.017: the rod is in the holder.
        "tool_length": 0.081,
        # place's vertical insert: home, cross at height, straight down, and
        # release only once the probe feels the seat (validated in sim
        # 2026-09-25 and 2026-09-27, see progress.md).
        "put_back": {"vertical_insert": True, "release_clearance": 0.0,
                     "stop_force_n": 1.0},
    },
}


def _norm(name: str) -> str:
    return " ".join(str(name).lower().replace("_", " ").split())


def names_for(tool: Dict[str, Any]) -> set:
    """Every name a tool goes by, normalised."""
    return {_norm(tool["name"])} | {_norm(a) for a in tool.get("aliases", ())}


def lookup(name: Optional[str]) -> Optional[Dict[str, Any]]:
    """The geometry for a tool picked up as ``name``, or None if not known."""
    if not name:
        return None
    key = _norm(name)
    for tool, geo in TOOLS.items():
        if key == _norm(tool) or key in {_norm(a) for a in geo.get("aliases", ())}:
            return {"name": tool, **geo}
    return None
