"""
Scene Understanding Agent: what is on the bench, named so the skills can act on it.

Merged in from robomail_Aliyah/agents/scene_understanding.py, itself adapted from
PLATO's scene_comprehension.py. What changes here is where the picture comes from
and what a name is worth.

Upstream took a path to a photograph. This takes the live cage -- four RealSense
views, or four rendered ones from the MuJoCo cell -- through the same
``VisionSystem`` the skills use, so the agent is looking at the same bench the
arm is about to reach into, in the channel order that cell actually produces.

The more important change is grounding. A name from this agent is not a caption:
it is the string a skill will hand to ``VisionSystem.locate``, and a skill whose
name does not resolve fails. Where the cell can enumerate what it holds
(:meth:`VisionSystem.known_object_names`) the agent is required to name objects
from that inventory, so "measuring scoop" cannot be planned against a bench whose
scoop is called "larger spoon". Where it cannot -- the real cage, where grounding
is open-vocabulary -- the agent names things itself and a name that fails to
resolve becomes a reported failure that feeds replanning.

The agent still has NO hard-coded scene knowledge. It is told the kit's
vocabulary so it names things consistently, not which objects are present or
where. That is PLATO's design and the claim the project rests on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from . import config, llm_client
from .robot_profile import DEFAULT_PROFILE, RobotProfile

AGENT_NAME = "SceneUnderstandingAgent"

#: Candidate names for the wet-chemistry kit. Supplied for consistency, not as a
#: claim about what is on the bench.
KIT_VOCABULARY: tuple = (
    "plastic beaker",
    "white paper cup",
    "clear cup",
    "pipette",
    "measuring scoop",
    "spoon",
    "stirring rod",
    "citric acid",
    "baking soda",
    "sodium bicarbonate",
    "red cabbage powder",
    "instant snow powder",
    "hydrophobic sand",
    "pH indicator paper",
    "water container",
    "tray",
)

#: Categories drive downstream reasoning about what can be poured or scooped.
OBJECT_CATEGORIES: tuple = (
    "container",     # beakers, cups -- hold contents, generally not relocated
    "reagent",       # powders and solutions, usually in a labelled cup
    "tool",          # pipette, scoop, stirring rod
    "indicator",     # pH paper, prepared indicator solution
    "distractor",    # in frame but irrelevant to the goal
)

_SYSTEM_PROMPT = """You are the Scene Understanding Agent of an autonomous robotic chemist.

You are given camera views of the robot's workspace from a four-camera cage, and
the goal downstream agents must achieve in that workspace. List the objects
actually visible on the bench.

For each object report:
  name             -- {name_rule}
  category         -- one of: {categories}
  has_handle       -- true if it has a graspable handle or stem (a pipette and a
                      scoop do; a cup and a beaker do not). This drives grasp
                      selection downstream.
  contents         -- what it appears to contain, or "empty", or "unknown".
                      Report what you SEE ("white powder", "dark purple liquid"),
                      not what you infer the chemistry to be.
  relevant_to_goal -- true if it is plausibly needed for the goal.

{vocabulary_block}

Rules:
  * Report only what is visible. Do not invent objects because the goal mentions
    them, and do not omit objects because the goal does not.
  * Include irrelevant objects, categorised as "distractor".
  * Ignore markings, tape and fiducials on the bench surface itself, except
    handwritten reagent labels on paper under a cup -- those tell you what the
    cup holds, and you should report the reagent as that cup's contents.
  * The views are of ONE bench from four angles. Do not report the same object
    once per camera.
  * Do not plan. Do not suggest actions. Only describe what is there.
  * Sort objects alphabetically by name.

{robot_block}
"""

_SCHEMA = {
    "type": "object",
    "properties": {
        "objects": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "category": {"type": "string", "enum": list(OBJECT_CATEGORIES)},
                    "has_handle": {"type": "boolean"},
                    "contents": {"type": "string"},
                    "relevant_to_goal": {"type": "boolean"},
                },
                "required": [
                    "name", "category", "has_handle", "contents", "relevant_to_goal",
                ],
                "additionalProperties": False,
            },
        },
        "scene_notes": {"type": "string"},
    },
    "required": ["objects", "scene_notes"],
    "additionalProperties": False,
}


@dataclass
class SceneObject:
    """One object grounded in the workspace."""

    name: str
    category: str
    has_handle: bool
    contents: str
    relevant_to_goal: bool
    #: False when an inventory exists and this name is not in it -- the skills
    #: will not be able to locate it.
    resolvable: bool = True

    def as_dict(self) -> dict:
        return {
            "name": self.name,
            "category": self.category,
            "has_handle": self.has_handle,
            "contents": self.contents,
            "relevant_to_goal": self.relevant_to_goal,
            "resolvable": self.resolvable,
        }


@dataclass
class Scene:
    """The Scene Understanding Agent's output."""

    objects: List[SceneObject] = field(default_factory=list)
    scene_notes: str = ""
    inventory: List[str] = field(default_factory=list)
    unresolvable: List[str] = field(default_factory=list)

    @property
    def names(self) -> List[str]:
        return [o.name for o in self.objects]

    def usable_names(self) -> List[str]:
        """
        Names a skill may be given.

        The inventory when there is one -- a planner may legitimately want an
        object the scene agent overlooked -- otherwise what the agent saw.
        """
        return list(self.inventory) if self.inventory else self.names

    def as_dict(self) -> dict:
        return {
            "objects": [o.as_dict() for o in self.objects],
            "scene_notes": self.scene_notes,
            "inventory": self.inventory,
            "unresolvable": self.unresolvable,
        }

    def as_prompt_block(self) -> str:
        """Render for injection into the planner's prompt."""
        if not self.objects:
            return "Objects visible in the workspace: (none detected)"
        lines = ["Objects visible in the workspace:"]
        for o in self.objects:
            handle = "has handle" if o.has_handle else "no handle"
            flags = "" if o.relevant_to_goal else "  [likely distractor]"
            if not o.resolvable:
                flags += "  [perception cannot resolve this name -- do not use it]"
            lines.append(
                f"  - {o.name} ({o.category}, {handle}, contains: {o.contents}){flags}"
            )
        if self.scene_notes:
            lines.append(f"Scene notes: {self.scene_notes}")
        if self.inventory:
            lines.append(
                "Object names perception can resolve (use one of these verbatim "
                "wherever a skill takes an object, container or source): "
                + ", ".join(f'"{n}"' for n in self.inventory)
            )
        return "\n".join(lines)


def comprehend_scene(
    vision,
    goal: str,
    *,
    max_views: int = 2,
    profile: RobotProfile = DEFAULT_PROFILE,
    model: Optional[str] = None,
    client=None,
) -> Scene:
    """
    Ground the objects on the bench, in the context of ``goal``.

    Args:
        vision: The :class:`~robochem.vision.VisionSystem` (or ``SimVision``)
            the skills will use. Frames come from it, and so does the name
            inventory when it has one.
        goal: What downstream agents must achieve here.
        max_views: How many of the cage's views to send. Two opposing views
            resolve most occlusion; all four mostly cost tokens.
        profile: Robot capability profile injected into the prompt.
        model: Override the cheap tier.
        client: Transport double, for tests.
    """
    frames = vision.capture_scene() or []
    if not frames:
        raise RuntimeError(
            "Scene understanding needs camera frames and capture_scene() returned "
            "none. On hardware check the cage; in simulation check the renderer."
        )
    order = getattr(vision, "frame_color_order", "bgr")
    inventory = list(getattr(vision, "known_object_names", list)() or [])

    if inventory:
        name_rule = (
            "the object's name, chosen VERBATIM from the resolvable-names list "
            "below. These are the only names the robot's perception can find. If a "
            "visible object matches none of them, name it briefly anyway and it "
            "will be marked unusable."
        )
        vocabulary_block = (
            "Resolvable names (this bench holds these and only these; pick the one "
            "that matches what you see):\n"
            + "\n".join(f"  - {n}" for n in inventory)
        )
    else:
        name_rule = (
            "a brief noun phrase. Prefer a name from the kit vocabulary below when "
            "the object plausibly matches one, so every agent uses the same words. "
            "This name is passed straight to an open-vocabulary detector, so it must "
            "describe the object as seen (\"plastic beaker\", \"white paper cup\")."
        )
        vocabulary_block = (
            "Kit vocabulary (candidate names, NOT a list of what is present):\n"
            + "\n".join(f"  - {v}" for v in KIT_VOCABULARY)
        )

    content = [
        llm_client.text_block(
            f"What objects are present in this workspace? The goal downstream "
            f"agents must achieve here is: <{goal}>"
        )
    ]
    for frame in frames[:max_views]:
        content.append(llm_client.image_block_from_array(frame, color_order=order))

    response = llm_client.structured_completion(
        agent=AGENT_NAME,
        model=model or config.cheap_model(),
        system_prompt=_SYSTEM_PROMPT.format(
            name_rule=name_rule,
            categories=", ".join(OBJECT_CATEGORIES),
            vocabulary_block=vocabulary_block,
            robot_block=profile.as_prompt_block(),
        ),
        user_content=content,
        schema=_SCHEMA,
        schema_name="scene_description",
        client=client,
    )

    known = {n.lower() for n in inventory}
    objects = []
    for o in response.get("objects", []):
        name = str(o.get("name", "")).strip()
        if not name:
            continue
        objects.append(
            SceneObject(
                name=name,
                category=str(o.get("category", "distractor")),
                has_handle=bool(o.get("has_handle", False)),
                contents=str(o.get("contents", "unknown")),
                relevant_to_goal=bool(o.get("relevant_to_goal", True)),
                resolvable=(not known) or name.lower() in known,
            )
        )
    objects.sort(key=lambda o: o.name.lower())
    return Scene(
        objects=objects,
        scene_notes=str(response.get("scene_notes", "")).strip(),
        inventory=inventory,
        unresolvable=[o.name for o in objects if not o.resolvable],
    )
