"""
High-Level Planner Agent: goal in, ordered sub-tasks out, and corrections after.

Merged in from robomail_Aliyah/agents/high_level_planner.py (itself adapted from
PLATO's overall_planner.py), with two changes for this stack.

The vocabulary block is now the real skill catalogue rather than a six-action
enum, so the planner decomposes against what the arm can actually do -- including
the pairings the catalogue records, like "a scoop fills the bowl and a dump
empties it", which a planner that has only been shown SCOOP gets wrong every
time.

``verification_modality`` is gone. Upstream had to pick color / motion / volume
up front so the right OpenCV detector could be called; this repo's
:class:`~robochem.verification.chemistry_verifier.ChemistryVerifier` routes
itself from the wording of the expected outcome, so the planner just says what
should become visible and the verifier decides how to look.

The planner is the planning step. Nothing upstream pre-decides the plan and
nothing downstream second-guesses its chemistry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from . import config, llm_client
from .robot_profile import DEFAULT_PROFILE, CapabilityWorkaround, RobotProfile
from .scene import Scene
from .skill_catalog import vocabulary_prompt_block

AGENT_NAME = "HighLevelPlannerAgent"

_PLAN_PROMPT = """You are the High-Level Planner Agent of an autonomous robotic chemist.

You are given ONE goal describing an outcome, plus the objects the Scene
Understanding Agent grounded in the workspace. Decompose the goal into an ordered
list of sub-tasks a single robot arm can carry out to achieve it.

You are the planning step. Nobody upstream has decided the plan for you and
nobody downstream will second-guess your chemistry. Reason about which reagents,
and in which order, actually produce the requested outcome.

{robot_block}

Each sub-task must be a short imperative clause naming what to do and to what,
and must correspond to EXACTLY ONE skill below. A downstream agent turns each
sub-task into that skill call with its parameters; you do not emit coordinates,
joint angles or gripper commands.

{vocabulary_block}

Do NOT add a sub-task for taking a photo, observing or verifying the final
outcome. The orchestrator captures verification frames itself. Name the container
to watch in "observation_target" instead.

Plan the gripper explicitly. The arm starts empty, so a sub-task that uses a tool
must be preceded by a sub-task that picks that tool up, and that tool must be put
down before a different one is picked up.

Use object names exactly as the scene lists them. An object the scene marks as
unresolvable cannot be acted on at all.

Autonomy rules -- these are strict:
  * Never plan a step that asks a human to do, check, approve or confirm anything.
  * Never plan two things happening at once. If the source material calls for
    simultaneity, serialise it and record a capability workaround.
  * If the goal is a demonstration that something does NOT change (showing a
    material stays dry, say), the plan still performs the action; absence of
    change is then the SUCCESS condition. Set "expected_observation" accordingly
    so the verifier is not told to hunt for a change that should not occur.
  * If the goal cannot be achieved with the objects present, set "feasible" to
    false and explain. Do not invent objects.

For each sub-task give:
  description      -- the imperative clause.
  rationale        -- one clause on why this step is needed.
  target_container -- the container whose state this step changes, or "none".

For the plan as a whole give:
  observation_target     -- the container whose state proves the goal was met.
  expected_observation   -- what should become visible at the end, phrased as an
                            observable outcome ("the liquid in the paper cup turns
                            pink", "the mixture fizzes"). The verifier picks its
                            detector from these words, so name the colour, the
                            bubbling or the volume change explicitly.
  capability_workarounds -- one entry for EVERY time the capability profile
                            changed your plan relative to a literal reading of the
                            goal. Empty list if none.
"""

_REPLAN_PROMPT = """You are the High-Level Planner Agent of an autonomous robotic chemist,
performing closed-loop correction.

A plan you produced was executed on the real arm and it did not reach the goal.
Produce a corrective sub-plan.

{robot_block}

{vocabulary_block}

You will be given: the goal, the workspace objects, the plan as executed, which
sub-tasks succeeded, which one failed, what the failure was, and how many
corrective attempts have already been made.

Rules:
  * Diagnose first: say in "diagnosis" why you think the outcome did not occur.
    Distinguish a chemistry problem (not enough reagent) from a manipulation one
    (the scoop came up empty, the object could not be found).
  * Then emit a corrective plan that starts from the CURRENT state of the bench.
    Steps that already succeeded have already happened -- do not blindly repeat
    them unless repeating is genuinely part of the correction. Adding a further
    measure of reagent is a legitimate correction.
  * Account for what the gripper is holding right now. It is stated below. If a
    tool is still held and the correction needs a different one, plan the
    'place' yourself.
  * If a skill failed because an object could not be located, the name was
    probably wrong for this bench. Use a different name from the scene listing
    rather than retrying the same one.
  * Never ask a human for help, approval or diagnosis.
  * If the goal is genuinely unreachable with the objects present, set "feasible"
    to false and explain. An honest reported failure beats a plan you do not
    believe in.
"""


def _subtask_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "description": {"type": "string"},
            "rationale": {"type": "string"},
            "target_container": {"type": "string"},
        },
        "required": ["description", "rationale", "target_container"],
        "additionalProperties": False,
    }


def _workaround_schema() -> dict:
    return {
        "type": "object",
        "properties": {
            "constraint": {"type": "string"},
            "literal_instruction": {"type": "string"},
            "adopted_approach": {"type": "string"},
            "outcome_differs": {"type": "boolean"},
            "note": {"type": "string"},
        },
        "required": [
            "constraint", "literal_instruction", "adopted_approach",
            "outcome_differs", "note",
        ],
        "additionalProperties": False,
    }


_PLAN_FIELDS = {
    "feasible": {"type": "boolean"},
    "infeasible_reason": {"type": "string"},
    "subtasks": {"type": "array", "items": _subtask_schema()},
    "observation_target": {"type": "string"},
    "expected_observation": {"type": "string"},
    "capability_workarounds": {"type": "array", "items": _workaround_schema()},
}

_PLAN_SCHEMA = {
    "type": "object",
    "properties": dict(_PLAN_FIELDS),
    "required": list(_PLAN_FIELDS),
    "additionalProperties": False,
}

_REPLAN_SCHEMA = {
    "type": "object",
    "properties": {"diagnosis": {"type": "string"}, **_PLAN_FIELDS},
    "required": ["diagnosis", *_PLAN_FIELDS],
    "additionalProperties": False,
}


@dataclass
class SubTask:
    """One ordered step of the high-level plan."""

    index: int
    description: str
    rationale: str = ""
    target_container: str = "none"

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "description": self.description,
            "rationale": self.rationale,
            "target_container": self.target_container,
        }


@dataclass
class Plan:
    """The High-Level Planner Agent's output."""

    goal: str
    subtasks: List[SubTask] = field(default_factory=list)
    observation_target: str = ""
    expected_observation: str = ""
    capability_workarounds: List[CapabilityWorkaround] = field(default_factory=list)
    feasible: bool = True
    infeasible_reason: str = ""
    diagnosis: str = ""
    is_correction: bool = False

    def as_dict(self) -> dict:
        return {
            "goal": self.goal,
            "subtasks": [s.as_dict() for s in self.subtasks],
            "observation_target": self.observation_target,
            "expected_observation": self.expected_observation,
            "capability_workarounds": [w.as_dict() for w in self.capability_workarounds],
            "feasible": self.feasible,
            "infeasible_reason": self.infeasible_reason,
            "diagnosis": self.diagnosis,
            "is_correction": self.is_correction,
        }

    def as_prompt_block(self) -> str:
        if not self.subtasks:
            return "(empty plan)"
        return "\n".join(f"  {s.index}. {s.description}" for s in self.subtasks)


def _system_prompt(template: str, profile: RobotProfile) -> str:
    return template.format(
        robot_block=profile.as_prompt_block(),
        vocabulary_block=vocabulary_prompt_block(),
    )


def _parse_plan(goal: str, response: dict, *, is_correction: bool) -> Plan:
    subtasks = [
        SubTask(
            index=i,
            description=str(s.get("description", "")).strip(),
            rationale=str(s.get("rationale", "")).strip(),
            target_container=str(s.get("target_container", "none")).strip(),
        )
        for i, s in enumerate(response.get("subtasks", []), start=1)
    ]
    subtasks = [s for s in subtasks if s.description]
    workarounds = [
        CapabilityWorkaround(
            constraint=str(w.get("constraint", "")),
            literal_instruction=str(w.get("literal_instruction", "")),
            adopted_approach=str(w.get("adopted_approach", "")),
            outcome_differs=bool(w.get("outcome_differs", False)),
            note=str(w.get("note", "")),
        )
        for w in response.get("capability_workarounds", [])
    ]
    return Plan(
        goal=goal,
        subtasks=subtasks,
        observation_target=str(response.get("observation_target", "")).strip(),
        expected_observation=str(response.get("expected_observation", "")).strip(),
        capability_workarounds=workarounds,
        feasible=bool(response.get("feasible", True)),
        infeasible_reason=str(response.get("infeasible_reason", "")).strip(),
        diagnosis=str(response.get("diagnosis", "")).strip(),
        is_correction=is_correction,
    )


def plan_goal(
    goal: str,
    scene: Scene,
    *,
    profile: RobotProfile = DEFAULT_PROFILE,
    model: Optional[str] = None,
    client=None,
) -> Plan:
    """Decompose ``goal`` into an ordered sub-task plan, grounded in ``scene``."""
    response = llm_client.structured_completion(
        agent=AGENT_NAME,
        model=model or config.strong_model(),
        system_prompt=_system_prompt(_PLAN_PROMPT, profile),
        user_content=[
            llm_client.text_block(
                f"Goal: {goal}\n\n{scene.as_prompt_block()}\n\n"
                "Decompose this goal into an ordered sub-task plan."
            )
        ],
        schema=_PLAN_SCHEMA,
        schema_name="subtask_plan",
        client=client,
    )
    return _parse_plan(goal, response, is_correction=False)


def replan(
    goal: str,
    scene: Scene,
    failed_plan: Plan,
    *,
    failed_subtask: Optional[SubTask],
    completed_subtasks: Sequence[SubTask],
    failure_reason: str,
    held_object: Optional[str],
    attempt: int,
    max_attempts: int,
    profile: RobotProfile = DEFAULT_PROFILE,
    model: Optional[str] = None,
    client=None,
) -> Plan:
    """Generate a corrective sub-plan after an execution or verification failure."""
    completed = "\n".join(f"  - {s.description}" for s in completed_subtasks) or "  (none)"
    failed_text = (
        failed_subtask.description if failed_subtask else "(whole-plan outcome check)"
    )
    response = llm_client.structured_completion(
        agent=AGENT_NAME,
        model=model or config.strong_model(),
        system_prompt=_system_prompt(_REPLAN_PROMPT, profile),
        user_content=[
            llm_client.text_block(
                f"Goal: {goal}\n\n"
                f"{scene.as_prompt_block()}\n\n"
                f"Plan as executed:\n{failed_plan.as_prompt_block()}\n\n"
                f"Sub-tasks that succeeded:\n{completed}\n\n"
                f"Sub-task that failed: {failed_text}\n\n"
                f"Expected observation: {failed_plan.expected_observation}\n"
                f"What went wrong: {failure_reason}\n\n"
                f"The gripper is currently holding: {held_object or 'nothing'}\n\n"
                f"This is corrective attempt {attempt} of at most {max_attempts}.\n"
                "Diagnose and produce a corrective sub-plan."
            )
        ],
        schema=_REPLAN_SCHEMA,
        schema_name="corrective_plan",
        client=client,
    )
    return _parse_plan(goal, response, is_correction=True)
