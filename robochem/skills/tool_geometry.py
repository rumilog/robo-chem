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
        "aliases": ("large spoon", "spoon", "scoop", "white plastic tool",
                    "rectangular scoop"),
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
    },
}


def _norm(name: str) -> str:
    return " ".join(str(name).lower().replace("_", " ").split())


def lookup(name: Optional[str]) -> Optional[Dict[str, Any]]:
    """The geometry for a tool picked up as ``name``, or None if not known."""
    if not name:
        return None
    key = _norm(name)
    for tool, geo in TOOLS.items():
        if key == _norm(tool) or key in {_norm(a) for a in geo.get("aliases", ())}:
            return {"name": tool, **geo}
    return None
