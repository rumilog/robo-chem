"""
Skill-Call Agent: one sub-task in, one executable skill call out.

This replaces two modules from robomail_Aliyah at once -- ``agents/affordance.py``
and ``agents/step_planner.py`` -- and it is the substantive part of the merge.

Upstream that pair chose an action from a six-word enum and then reasoned out a
GOTO / GRASP / TILT trajectory for it, which was handed to a stub executor that
reported success without the arm moving. Both halves are unnecessary here and the
second is actively wrong: this repo's skills do their own perception, their own
grasp analysis, their own approach and their own arrival checks, tuned against
the bench over many runs. An LLM emitting Cartesian deltas would be overriding
all of that with numbers it cannot measure.

So the model's job shrinks to the one thing it is actually better at than the
code: deciding WHICH skill and WITH WHAT ARGUMENTS. The output is a call that
:meth:`robochem.skills.SkillsExecutor.execute` runs on the real arm, or on the
MuJoCo cell, which presents the same surface.

What is kept from upstream is the part that was doing real work: the vocabulary
is closed and enforced through a strict JSON schema; the model is given the
gripper's actual state rather than being told to assume every previous step
worked; and a sub-task that cannot be expressed raises
:class:`~robochem.agents.skill_catalog.InfeasibleSkill` rather than being
approximated with a neighbouring skill.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from . import config, llm_client
from .planner import SubTask
from .robot_profile import DEFAULT_PROFILE, RobotProfile
from .scene import Scene
from .skill_catalog import (
    InfeasibleSkill,
    coerce_params,
    parse_skill,
    skill_enum_schema,
    vocabulary_prompt_block,
)

AGENT_NAME = "SkillCallAgent"

_SYSTEM_PROMPT = """You are the Skill-Call Agent of an autonomous robotic chemist.

You are given ONE sub-task from the high-level plan, the objects grounded in the
workspace, what the gripper is currently holding, and what the previous steps of
this run actually did. Emit the single skill call that carries out this sub-task.

{vocabulary_block}

{robot_block}

How to choose:
  * Exactly one skill call. If the sub-task needs two (pick the scoop up AND
    scoop with it), that is a planning error upstream: emit the call for the
    FIRST thing that must happen and say so in "rationale".
  * Respect the gripper state you are given. It is measured, not assumed. A
    skill that needs a held tool cannot run with an empty gripper, and 'pick_up'
    cannot run with a full one.
  * Object, container and source names are passed straight to the robot's
    perception. Use the names exactly as the scene lists them. Do not invent a
    name, pluralise one, or describe an object a different way.
  * Set only the parameters you have a reason to set. Every parameter you leave
    out keeps a value measured against this bench; every one you set overrides
    it. Guessing a distance is worse than saying nothing.
  * Values go in "params" as JSON literals: a string as "plastic beaker" (with
    the quotes), a number as 0.02, a boolean as true.

If this sub-task cannot be carried out by exactly one skill from the list, set
"feasible" to false and say what capability is missing. Do NOT approximate it
with a different skill and do NOT invent a skill name.
"""

_SCHEMA = {
    "type": "object",
    "properties": {
        "feasible": {"type": "boolean"},
        "infeasible_reason": {"type": "string"},
        "skill": skill_enum_schema(),
        "params": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "value": {
                        "type": "string",
                        "description": "The value as a JSON literal, e.g. "
                                       '"plastic beaker" or 0.02 or true',
                    },
                },
                "required": ["name", "value"],
                "additionalProperties": False,
            },
        },
        "rationale": {"type": "string"},
        "expected_outcome": {"type": "string"},
    },
    "required": [
        "feasible", "infeasible_reason", "skill", "params", "rationale",
        "expected_outcome",
    ],
    "additionalProperties": False,
}


@dataclass
class SkillCall:
    """One executable call: what :class:`SkillsExecutor` takes."""

    skill: str
    params: Dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    expected_outcome: str = ""
    subtask_index: int = 0
    subtask: str = ""

    def as_dict(self) -> dict:
        return {
            "skill": self.skill,
            "params": self.params,
            "rationale": self.rationale,
            "expected_outcome": self.expected_outcome,
            "subtask_index": self.subtask_index,
            "subtask": self.subtask,
        }

    def as_command(self) -> str:
        """The equivalent run_experiment.py invocation, for the log and the console."""
        import json

        return (f"--skill {self.skill} --params "
                f"'{json.dumps(self.params, default=str)}'")


@dataclass
class StepOutcome:
    """What one executed call did, as history for the next one."""

    call: SkillCall
    success: bool
    detail: str = ""

    def as_line(self) -> str:
        verdict = "ok" if self.success else "FAILED"
        detail = f" -- {self.detail}" if self.detail else ""
        return f"{self.call.skill}({self.call.params}) -> {verdict}{detail}"


def plan_skill_call(
    subtask: SubTask,
    scene: Scene,
    *,
    held_object: Optional[str] = None,
    history: Sequence[StepOutcome] = (),
    profile: RobotProfile = DEFAULT_PROFILE,
    model: Optional[str] = None,
    client=None,
) -> SkillCall:
    """
    Turn one sub-task into one executable skill call.

    Args:
        subtask: The step to realise.
        scene: The grounded workspace, for object names.
        held_object: What the gripper actually holds, as the orchestrator has
            tracked it from executed calls. ``None`` means empty.
        history: The last few executed calls and their verdicts. Upstream told
            the model to "assume every upstream step already succeeded"; here it
            is told what really happened, which is the difference between
            planning open loop and closed.
        profile: Robot capability profile injected into the prompt.
        model: Override the strong tier.
        client: Transport double, for tests.

    Raises:
        InfeasibleSkill: The sub-task cannot be expressed as one call, the model
            named a skill that is not in the registry, or the parameters it gave
            do not fit the skill it chose. All three are reported planning
            failures the orchestrator feeds back into replanning.
    """
    recent = "\n".join(f"  {o.as_line()}" for o in list(history)[-4:]) or "  (none yet)"

    response = llm_client.structured_completion(
        agent=AGENT_NAME,
        model=model or config.strong_model(),
        system_prompt=_SYSTEM_PROMPT.format(
            vocabulary_block=vocabulary_prompt_block(),
            robot_block=profile.as_prompt_block(),
        ),
        user_content=[
            llm_client.text_block(
                f"Sub-task {subtask.index}: {subtask.description}\n"
                f"Why this step exists: {subtask.rationale}\n"
                f"Container this step changes: {subtask.target_container}\n\n"
                f"{scene.as_prompt_block()}\n\n"
                f"The gripper is currently holding: {held_object or 'nothing'}\n"
                f"Steps already executed in this run:\n{recent}\n\n"
                "Emit the single skill call that carries out this sub-task."
            )
        ],
        schema=_SCHEMA,
        schema_name="skill_call",
        client=client,
    )

    if not response.get("feasible", True):
        raise InfeasibleSkill(
            subtask.description,
            str(response.get("infeasible_reason", "")).strip()
            or "the skill-call agent reported no single skill realises this sub-task",
        )

    skill = parse_skill(response.get("skill", ""), subtask=subtask.description)
    params = coerce_params(skill, response.get("params", []), subtask=subtask.description)

    return SkillCall(
        skill=skill,
        params=params,
        rationale=str(response.get("rationale", "")).strip(),
        expected_outcome=str(response.get("expected_outcome", "")).strip(),
        subtask_index=subtask.index,
        subtask=subtask.description,
    )
