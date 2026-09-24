"""
The action vocabulary: this repo's real skills, rendered for an LLM.

This module is the join between the two halves of the merge. robomail_Aliyah's
pipeline planned against a closed six-action enum (PICKUP / POUR / SCOOP /
PIPETTE_DISPENSE / STIR / PLACE) and handed the result to a fake executor that
reported success without moving. Here the vocabulary IS
:data:`robochem.skills.SKILL_REGISTRY`, and what comes out the other end is a
call the real :class:`~robochem.skills.SkillsExecutor` runs against the arm --
or against the MuJoCo cell, which presents the same surface.

What is kept from upstream is the discipline, which is the part that was load
bearing: the vocabulary is closed, the model picks from it through a strict JSON
schema, and a sub-task that cannot be expressed in it is a reported failure
(:class:`InfeasibleSkill`) that feeds replanning -- never an invented skill name
or a quietly substituted one.

Two things the catalogue deliberately does NOT do:

  * It does not advertise every parameter. Skills like ``scoop`` and
    ``arc_scoop`` carry twenty-odd tuning knobs -- dig angles, untilt distances,
    waypoint counts -- measured against this bench over many runs. The model has
    no basis to re-derive them and every one it sets is a chance to undo that
    tuning, so only the semantically meaningful parameters are exposed and the
    rest keep their defaults. The executor still accepts the others if a human
    passes them.
  * It does not paraphrase the skills by hand and hope they stay in sync. The
    required and optional parameter names come from the skill classes
    themselves, and :func:`_audit` fails at import if a curated entry names a
    parameter a skill does not have.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from robochem.skills import SKILL_REGISTRY, SkillsExecutor


class InfeasibleSkill(Exception):
    """
    A sub-task that cannot be expressed as one call in the skill vocabulary.

    A reported planning failure the orchestrator logs and feeds back into
    replanning, not a fallback to free-text actions.
    """

    def __init__(self, subtask: str, reason: str) -> None:
        self.subtask = subtask
        self.reason = reason
        super().__init__(
            f"Cannot express sub-task {subtask!r} as a single skill call: {reason}"
        )


@dataclass
class Param:
    """One planner-visible parameter of a skill."""

    name: str
    kind: str                  # "string" | "number" | "integer" | "boolean" | "list"
    description: str
    required: bool = False
    default: Any = None


@dataclass
class SkillSpec:
    """How one skill is described to the planner."""

    name: str
    summary: str
    requires: str              # gripper precondition, in words
    effect: str                # what is true afterwards
    params: List[Param] = field(default_factory=list)
    moves_arm: bool = True
    note: str = ""

    def as_prompt_block(self) -> str:
        lines = [f"  {self.name}: {self.summary}"]
        lines.append(f"      requires: {self.requires}")
        lines.append(f"      effect:   {self.effect}")
        if self.note:
            lines.append(f"      note:     {self.note}")
        for p in self.params:
            tag = "required" if p.required else f"optional, default {p.default!r}"
            lines.append(f"      - {p.name} ({p.kind}, {tag}): {p.description}")
        return "\n".join(lines)


# --------------------------------------------------------------------------
# The catalogue
# --------------------------------------------------------------------------

_HELD = "the gripper to already hold"
_EMPTY = "an empty gripper"

CATALOG: Dict[str, SkillSpec] = {
    "pick_up": SkillSpec(
        name="pick_up",
        summary="Locate an object in the cage and grasp it, then lift.",
        requires=_EMPTY,
        effect="the gripper holds that object; nothing else can be picked up until it is placed",
        params=[
            Param("object_name", "string",
                  "What to grasp. Must be a name perception can resolve.", required=True),
            Param("z_offset", "number",
                  "Raise (+) or lower (-) the grasp relative to the measured one, metres. "
                  "0.02 is the validated value for the beaker.", default=0.0),
            Param("squeeze", "number",
                  "Close this far inside the measured object width, metres. Larger grips "
                  "harder. 0.013 is the validated value for the beaker.", default=0.004),
            Param("grasp_force", "number",
                  "Gripper force, newtons. Lower for a thin or fragile item; 1.0 is the "
                  "validated value for the spoon.", default=5.0),
        ],
    ),
    "place": SkillSpec(
        name="place",
        summary="Set the held object down at a target and release it.",
        requires=f"{_HELD} something",
        effect="the gripper is empty again",
        params=[
            Param("target_location", "string",
                  "Either the name of an object to set this down BESIDE (on the side "
                  "nearer the robot base), or explicit coordinates as [x, y] or "
                  "[x, y, z] in metres. There is no name for bare bench: 'table' is "
                  "not special and perception will fail to find it. To clear the tool "
                  "out of the way, name something near where it should go, or give "
                  "coordinates.", required=True),
            Param("on_top", "boolean",
                  "Stack it on top of the target instead of beside it.", default=False),
        ],
    ),
    "pour": SkillSpec(
        name="pour",
        summary="Tip the held container so its contents run into a target container.",
        requires=f"{_HELD} the source container",
        effect="the source's contents are in the target; the source is still held",
        params=[
            Param("target_container", "string",
                  "Container to pour into.", required=True),
            Param("pour_angle", "number",
                  "How far to tip, degrees. 90 empties it.", default=90.0),
            Param("hold_duration", "number",
                  "Seconds held at the pour angle before righting.", default=2.0),
            Param("forward_offset", "number",
                  "Shift the pour point along +X, metres. Negative pulls it back toward "
                  "the base. The default is the bench-validated value; change it only "
                  "if a pour has already missed.", default=-0.08),
        ],
    ),
    "scoop": SkillSpec(
        name="scoop",
        summary="Dig a measure of powder out of a source container with the held scoop.",
        requires=f"{_HELD} a scoop or spoon",
        effect="the scoop's bowl is loaded with powder; NOTHING has been delivered anywhere yet",
        note="A scoop only fills the bowl. Deliver it with 'dump' before scooping again "
             "or putting the scoop down, or the powder never reaches the target.",
        params=[
            Param("powder_source", "string",
                  "Container or reagent to dig from. A reagent name resolves through the "
                  "label under the cup.", required=True),
            Param("scoop_depth", "number",
                  "How far below the measured surface to dig, metres.", default=0.015),
            Param("tool_offset", "list",
                  "[x, y, z] from the grasp point to the bowl, tool frame, metres. "
                  "Pass the measurement the 'pick_up' that grasped this scoop reported "
                  "as suggested_tool_offset -- it is stated with the gripper contents "
                  "below. Never invent one: it depends on where the jaws actually "
                  "closed, not on the tool alone, and a wrong one digs beside the "
                  "powder.", default=None),
        ],
    ),
    "arc_scoop": SkillSpec(
        name="arc_scoop",
        summary="Same as scoop, but a curved stroke that leaves the bed climbing.",
        requires=f"{_HELD} a scoop or spoon",
        effect="the scoop's bowl is loaded with powder; nothing has been delivered yet",
        note="An alternative stroke to 'scoop', not a replacement: 'scoop' is the one "
             "tuned against this bench. Prefer 'scoop' unless it has already failed to "
             "pick up material. Follow either with 'dump'.",
        params=[
            Param("powder_source", "string", "Container or reagent to dig from.",
                  required=True),
            Param("scoop_depth", "number",
                  "Depth of the lowest point of the arc below the powder surface, metres.",
                  default=0.015),
            Param("tool_offset", "list",
                  "[x, y, z] from the grasp point to the bowl, tool frame, metres. As "
                  "for 'scoop': pass the measurement stated with the gripper contents, "
                  "never an invented one.", default=None),
        ],
    ),
    "dump": SkillSpec(
        name="dump",
        summary="Tip the loaded scoop out over a target container.",
        requires=f"{_HELD} a loaded scoop",
        effect="the powder is in the target container; the scoop is still held, now empty",
        params=[
            Param("target_container", "string", "Where the powder goes.", required=True),
            Param("dump_angle_deg", "number",
                  "How far past level to tip. Must exceed 90 -- powder sits in the corner "
                  "of a bowl held at 90.", default=120.0),
            Param("tool_offset", "list",
                  "[x, y, z] from the grasp point to the bowl, tool frame, metres. As "
                  "for 'scoop': pass the measurement stated with the gripper contents.",
                  default=None),
        ],
    ),
    "dispense": SkillSpec(
        name="dispense",
        summary="Draw up and/or release drops of liquid with the held pipette.",
        requires=f"{_HELD} a pipette",
        effect="drops are released into the target (and/or liquid drawn from the source)",
        params=[
            Param("target_container", "string", "Where the drops go.", required=True),
            Param("mode", "string",
                  "'dispense' releases drops, 'aspirate' draws from source_container, "
                  "'transfer' does both.", default="dispense"),
            Param("num_drops", "integer", "How many drops to release.", default=3),
            Param("source_container", "string",
                  "Where to draw from. Required for 'aspirate' and 'transfer'.",
                  default=None),
        ],
    ),
    "stir": SkillSpec(
        name="stir",
        summary="Trace circles inside a container with the held stirring implement.",
        requires=f"{_HELD} a stirrer, spoon or rod",
        effect="the container's contents are mixed; the implement is still held",
        params=[
            Param("target_container", "string", "Container to stir.", required=True),
            Param("revolutions", "integer", "How many circles to trace.", default=3),
            Param("stir_depth", "number",
                  "How far below the rim to immerse the implement, metres.", default=0.03),
            Param("tool_length", "number",
                  "How far the implement's tip hangs below the grasp point, metres. "
                  "'stir' takes only this depth, not a full offset vector. Use the z "
                  "component of the measurement stated with the gripper contents.",
                  default=0.0),
        ],
    ),
    "move_to": SkillSpec(
        name="move_to",
        summary="Move the end effector to an object, named pose or explicit coordinates.",
        requires="nothing in particular",
        effect="the arm is at the requested pose; nothing is grasped or released",
        note="Rarely needed. Every manipulation skill approaches its own target.",
        params=[
            Param("target", "string",
                  "An object name, the word 'home', or exactly three coordinates "
                  "[x, y, z] in metres. Nothing else is recognised -- 'safe' and 'table' "
                  "are not named poses and will fail to resolve.", required=True),
            Param("offset", "list",
                  "[dx, dy, dz] added to the target, metres.", default=[0.0, 0.0, 0.0]),
        ],
    ),
    "open_gripper": SkillSpec(
        name="open_gripper",
        summary="Open the jaws.",
        requires="nothing in particular",
        effect="the jaws are open; anything held is dropped where it is",
        note="Dropping a held object is not the same as putting it down. Use 'place'.",
        params=[Param("width", "number", "Opening in metres.", default=0.08)],
    ),
    "close_gripper": SkillSpec(
        name="close_gripper",
        summary="Close the jaws.",
        requires="nothing in particular",
        effect="the jaws are closed",
        note="This is a blind close on whatever is between the fingers. To acquire an "
             "object, use 'pick_up', which looks first.",
        params=[
            # No width here: closing drives to a force, not to an opening, and
            # the shared 'width' parameter is ignored on this path.
            Param("force", "number", "Grasp force, newtons.", default=15.0),
        ],
    ),
    "tilt": SkillSpec(
        name="tilt",
        summary="Rotate the wrist to an angle, holding position.",
        requires="nothing in particular",
        effect="the end effector is at the requested tilt",
        note="'pour' and 'dump' do their own tilting. Use this only to right a wrist or "
             "drain something the other skills do not cover.",
        params=[
            Param("angle", "number", "Angle in degrees; 0 returns to upright.",
                  required=True),
            Param("axis", "string", "'x', 'y' or 'z'.", default="y"),
            Param("hold_duration", "number", "Seconds to hold there.", default=0.0),
        ],
    ),
    "wait": SkillSpec(
        name="wait",
        summary="Pause. For reactions to run, liquids to settle, foam to rise.",
        requires="nothing",
        effect="time passes; nothing moves",
        moves_arm=False,
        params=[
            Param("duration", "number", "Seconds to wait.", required=True),
            Param("reason", "string", "Why, for the log.", default=""),
        ],
    ),
    "analyze_scene": SkillSpec(
        name="analyze_scene",
        summary="Describe everything on the bench from the cage cameras.",
        requires="nothing",
        effect="a scene description is returned; the arm does not move",
        moves_arm=False,
        note="The orchestrator already grounds the scene before planning. Ask for this "
             "only when something may have changed and you need to look again.",
        params=[],
    ),
    "locate_object": SkillSpec(
        name="locate_object",
        summary="Find one object in 3D and report its centroid and size.",
        requires="nothing",
        effect="a position is returned; the arm does not move",
        moves_arm=False,
        params=[
            Param("object_name", "string", "What to look for.", required=True),
            Param("compute_grasp", "boolean", "Also suggest grasp poses.", default=True),
        ],
    ),
    "check_container": SkillSpec(
        name="check_container",
        summary="Report a container's fill level and dominant colour.",
        requires="nothing",
        effect="a state reading is returned; the arm does not move",
        moves_arm=False,
        note="This is how a colour change or a fill level is observed.",
        params=[
            Param("container_name", "string", "Container to inspect.", required=True),
            Param("check_color", "boolean", "Report the dominant colour.", default=True),
            Param("check_fill_level", "boolean", "Estimate how full it is.", default=True),
        ],
    ),
}

#: Ordering and pairing rules the parameter list cannot express.
CHAINING_RULES: tuple = (
    "The gripper starts empty. Any skill that needs a tool must be preceded by a "
    "'pick_up' of that tool, and the tool must be put down with 'place' before a "
    "different one is picked up.",
    "Powder moves in two steps, never one: 'scoop' fills the bowl, 'dump' empties it "
    "into the target. A 'scoop' with no 'dump' after it delivers nothing.",
    "Liquid in a container moves by picking the container up and using 'pour'. Powder "
    "is never poured from its tub.",
    "Drop-scale liquid additions use a held pipette and 'dispense'.",
    "'place' the tool down at the end, so the bench is left as it was found.",
    "A successful 'pick_up' of a tool measures where that tool's working end sits "
    "relative to the grasp, and that measurement is given to you with the gripper "
    "contents. Every later skill that uses the tool needs it. Passing it on is not "
    "optional and guessing a replacement is worse than any value you could guess.",
)


def _registry_params(skill_name: str) -> tuple:
    """(required, optional-with-defaults) as the skill class itself declares them."""
    cls = SKILL_REGISTRY[skill_name]
    blank = cls.__new__(cls)
    required = cls.required_params
    if isinstance(required, property):
        required = required.fget(blank)
    optional = cls.optional_params
    if isinstance(optional, property):
        optional = optional.fget(blank)
    # An alias pins a parameter (open_gripper pins action="open"), so the planner
    # neither needs to supply it nor may override it.
    pinned = set(SkillsExecutor.SKILL_ALIASES.get(skill_name, {}))
    return [p for p in required if p not in pinned], dict(optional or {})


def _audit() -> None:
    """
    Fail loudly if the catalogue and the skills have drifted apart.

    A curated entry that names a parameter the skill dropped would otherwise be
    advertised to the model, accepted into a call, and then discarded by
    ``get_params_with_defaults`` with only a printed warning -- the exact silent
    failure mode that comment in base_skill.py was written about.
    """
    problems = []
    for skill_name in SKILL_REGISTRY:
        spec = CATALOG.get(skill_name)
        if spec is None:
            problems.append(f"{skill_name}: in SKILL_REGISTRY but not in the catalogue")
            continue
        required, optional = _registry_params(skill_name)
        known = set(required) | set(optional)
        for p in spec.params:
            if p.name not in known:
                problems.append(
                    f"{skill_name}.{p.name}: advertised but the skill does not accept it"
                )
                continue
            if p.required and p.name not in required:
                problems.append(
                    f"{skill_name}.{p.name}: advertised as required but the skill "
                    f"does not require it"
                )
            # An advertised default that is not the skill's own is worse than no
            # default: the model is told a value it will not get, and reasons
            # about the difference between that and the one it sets.
            if not p.required and p.default != optional.get(p.name):
                problems.append(
                    f"{skill_name}.{p.name}: advertised default {p.default!r} but the "
                    f"skill defaults to {optional.get(p.name)!r}"
                )
        for p in required:
            if p not in {q.name for q in spec.params}:
                problems.append(
                    f"{skill_name}.{p}: required by the skill but not advertised"
                )
    if problems:
        raise RuntimeError(
            "robochem.agents.skill_catalog is out of step with robochem.skills:\n  "
            + "\n  ".join(problems)
        )


_audit()

SKILL_NAMES: List[str] = list(CATALOG)


def vocabulary_prompt_block(skills: Optional[List[str]] = None) -> str:
    """Render the catalogue for an LLM system prompt."""
    names = list(skills or SKILL_NAMES)
    lines = [
        "You may ONLY use the skills below. This list is exhaustive and closed.",
        "You must not invent, rename, compose or hyphenate a skill name, and you must",
        "not pass a parameter that is not listed for that skill.",
        "",
    ]
    lines += [CATALOG[n].as_prompt_block() for n in names if n in CATALOG]
    lines += ["", "How skills chain:"]
    lines += [f"  - {rule}" for rule in CHAINING_RULES]
    lines += [
        "",
        "If a sub-task cannot be accomplished with the skills above, do NOT approximate",
        'it with a different skill and do NOT invent a name. Set "feasible" to false and',
        'say what capability is missing in "infeasible_reason".',
    ]
    return "\n".join(lines)


def skill_enum_schema(skills: Optional[List[str]] = None) -> dict:
    """JSON-schema fragment restricting a field to the closed vocabulary."""
    return {"type": "string", "enum": list(skills or SKILL_NAMES)}


def parse_skill(raw: str, *, subtask: str = "<unknown>") -> str:
    """
    Coerce a model-supplied string to a registry skill name.

    Structured output should already guarantee membership; this is the check for
    providers whose schema enforcement turns out to be advisory, and for the
    JSON-object fallback path where it is not enforced at all.
    """
    key = str(raw).strip().lower().replace("-", "_").replace(" ", "_")
    if key in CATALOG:
        return key
    raise InfeasibleSkill(
        subtask, f"model returned skill {raw!r}, which is not one of {SKILL_NAMES}"
    )


def coerce_params(skill_name: str, pairs, *, subtask: str = "<unknown>") -> dict:
    """
    Turn the model's ``[{name, value}]`` list into a parameter dict.

    Values arrive as JSON literals in a string, because a strict JSON schema
    cannot describe an object whose keys vary by skill. A value that is not
    valid JSON is taken as the plain string it is, which is what the model meant
    when it wrote ``plastic beaker`` instead of ``"plastic beaker"``.

    Raises :class:`InfeasibleSkill` when a required parameter is missing or an
    unknown one is supplied -- both silently survivable otherwise, and both
    produce a skill that runs and does the wrong thing.
    """
    required, optional = _registry_params(skill_name)
    known = set(required) | set(optional)

    params: dict = {}
    for pair in pairs or []:
        name = str(pair.get("name", "")).strip()
        if not name:
            continue
        raw = pair.get("value", "")
        try:
            value = json.loads(raw) if isinstance(raw, str) else raw
        except (json.JSONDecodeError, TypeError):
            value = raw
        # The model declining an optional parameter. It writes that as JSON
        # null (which decodes to None) or, less correctly, as the word "none".
        if value is None or (
            isinstance(value, str) and value.strip().lower() in ("none", "null", "")
        ):
            continue
        params[name] = value

    unknown = sorted(set(params) - known)
    if unknown:
        raise InfeasibleSkill(
            subtask,
            f"'{skill_name}' was called with parameter(s) {unknown}, which it does not "
            f"accept. It takes {sorted(known)}.",
        )
    missing = [p for p in required if p not in params]
    if missing:
        raise InfeasibleSkill(
            subtask, f"'{skill_name}' is missing required parameter(s) {missing}"
        )
    return params


def summarise() -> str:
    """One line per skill, for a CLI listing."""
    return "\n".join(f"{n:16s} {CATALOG[n].summary}" for n in SKILL_NAMES)
