"""
Adapters between robochem's real skills and other agent stacks.

``plato_bridge`` wires robomail_Aliyah's PLATO-derived planner into the Franka
cell by replacing its stub skill executor. Nothing here imports robomail_Aliyah
or frankapy at module load, so it stays importable off the robot PC.
"""

from .plato_bridge import (
    ACTION_TO_SKILL,
    RoboChemExecutor,
    SkillExecutionResult,
    UnmappedAction,
    build_position_lookup,
    executor_provenance,
    normalize_object_name,
)

__all__ = [
    "ACTION_TO_SKILL",
    "RoboChemExecutor",
    "SkillExecutionResult",
    "UnmappedAction",
    "build_position_lookup",
    "executor_provenance",
    "normalize_object_name",
]
