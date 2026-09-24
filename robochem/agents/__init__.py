"""
The LLM agent layer: robomail_Aliyah's ChatGPT pipeline, bound to real skills.

The split between the two repositories was clean and complementary. robochem had
skills that move the arm and no planner worth the name -- one flat prompt that
described nine skills in a comment block and drifted out of date. robomail_Aliyah
had a careful multi-agent planner with structured output, a capability profile
and a closed-loop correction cycle, wired to an executor that printed
``[FAKE EXECUTION]`` and never moved anything.

This package is the planner half, re-pointed at the real skills:

    scene.py           what is on the bench, named so a skill can find it
    planner.py         goal -> ordered sub-tasks, and corrections after a failure
    skill_planner.py   one sub-task -> one executable SkillsExecutor call
    skill_catalog.py   the closed vocabulary, generated from SKILL_REGISTRY
    robot_profile.py   the cell's hard constraints, in prompt form
    llm_client.py      the ChatGPT call: structured output, vision, key handling
    config.py          model tiers and .env loading

The loop that drives them is
:class:`robochem.orchestrator.agent_orchestrator.AgentOrchestrator`.
"""

from .llm_client import LLMResponseError, MissingAPIKeyError
from .planner import Plan, SubTask, plan_goal, replan
from .robot_profile import DEFAULT_PROFILE, CapabilityWorkaround, RobotProfile
from .scene import Scene, SceneObject, comprehend_scene
from .skill_catalog import CATALOG, SKILL_NAMES, InfeasibleSkill, vocabulary_prompt_block
from .skill_planner import SkillCall, StepOutcome, plan_skill_call

__all__ = [
    "CATALOG", "SKILL_NAMES", "DEFAULT_PROFILE",
    "CapabilityWorkaround", "InfeasibleSkill", "LLMResponseError",
    "MissingAPIKeyError", "Plan", "RobotProfile", "Scene", "SceneObject",
    "SkillCall", "StepOutcome", "SubTask",
    "comprehend_scene", "plan_goal", "plan_skill_call", "replan",
    "vocabulary_prompt_block",
]
