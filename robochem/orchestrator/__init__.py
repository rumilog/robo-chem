"""
RoboChem Orchestrator Module

VLM-based orchestration for autonomous chemistry manipulation.

:class:`AgentOrchestrator` is the current one: robomail_Aliyah's multi-agent
pipeline -- scene understanding, high-level planning, per-step skill choice and
closed-loop replanning, all through strict structured output -- driving this
repo's real skills.

:class:`VLMOrchestrator` is the older single-prompt planner. It is kept because
it is what ``--task`` has always run and some notes refer to it, but its skill
descriptions are a hand-maintained comment block that has already drifted from
the registry, and it retries a failed step unchanged rather than replanning.
New work should use :class:`AgentOrchestrator`.
"""

from .agent_orchestrator import AgentOrchestrator, TaskOutcome
from .vlm_orchestrator import VLMOrchestrator
from .task_planner import TaskPlanner

__all__ = ["AgentOrchestrator", "TaskOutcome", "VLMOrchestrator", "TaskPlanner"]
