"""
Robot capability profile: the cell's hard physical constraints, in prompt form.

Merged in from robomail_Aliyah/config/robot_profile.py. Injected into the
planner's and the skill planner's system prompts so that physically impossible
instructions ("pour both cups in at the same time") are reasoned around at plan
time rather than discovered when the arm cannot do it.

This is not a review gate. Nothing here blocks a plan or asks a human. It
changes what the planner produces, and every time it does, the planner reports a
workaround note that the trial log records.

Upstream's ``workspace_positions`` are gone. PLATO worked against a fixed set of
taught, semantically-named poses; this stack does not -- every skill locates its
target by name through the camera cage on the spot, so the equivalent vocabulary
here is the set of object names perception can actually resolve, which is
per-bench and lives in :mod:`robochem.agents.scene`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RobotProfile:
    """Hard capability constraints, embedded at plan time."""

    name: str = "Franka Emika Panda (single-arm wet-chemistry cell, 4-camera RealSense cage)"

    num_arms: int = 1
    num_grippers: int = 1
    gripper_type: str = "parallel-plate, two-finger"
    max_simultaneously_held_objects: int = 1

    payload_kg: float = 3.0
    gripper_max_opening_cm: float = 8.0

    hard_constraints: tuple = (
        "There is exactly ONE arm with ONE parallel-plate gripper.",
        "Only ONE object may be held at any moment. To use a second object, the "
        "first must be put down with 'place' first.",
        "The robot CANNOT perform two spatially-separate actions simultaneously. "
        "Any instruction of the form 'do A and B at the same time' must be "
        "re-expressed as a strictly sequential plan.",
        "The robot CANNOT exert two independent forces at once -- it cannot hold a "
        "container steady while stirring it. Containers must be stable on the "
        "bench unaided.",
        "There is no wrist force/torque sensing fine enough to meter a pour by "
        "weight. Quantities are metered by scoop or by pipette drops.",
        "Containers are not relocated unless the task requires it. Prefer moving "
        "contents into a stationary container over moving the container.",
        "Every skill finds its own target by name through the camera cage. An "
        "object the cameras cannot resolve by the name given cannot be acted on, "
        "and the skill reports that as a failure.",
    )

    workaround_directives: tuple = (
        "Serialise simultaneous requirements into an explicit ordered sequence.",
        "Insert an explicit 'place' before any step that needs a different tool.",
        "If serialising changes the observable outcome -- a reaction the source "
        "text expects on simultaneous contact, say -- say so plainly in the "
        "workaround note rather than pretending it is equivalent.",
    )

    def as_prompt_block(self) -> str:
        """Render the profile for injection into an LLM system prompt."""
        lines = [
            "ROBOT CAPABILITY PROFILE (hard constraints -- plan around these yourself):",
            f"  Platform: {self.name}",
            f"  Arms: {self.num_arms}   Grippers: {self.num_grippers} ({self.gripper_type})",
            f"  Max objects held at once: {self.max_simultaneously_held_objects}",
            f"  Payload: {self.payload_kg} kg   "
            f"Max gripper opening: {self.gripper_max_opening_cm} cm",
            "",
            "  Constraints:",
        ]
        lines += [f"    - {c}" for c in self.hard_constraints]
        lines += ["", "  When a constraint makes the literal instruction impossible:"]
        lines += [f"    - {d}" for d in self.workaround_directives]
        lines += [
            "",
            "  You must NOT ask a human for help, approval or clarification. Decide",
            "  yourself and record what you changed.",
            "",
            "  Whenever a constraint above changes your plan relative to a literal",
            '  reading of the goal, add an entry to "capability_workarounds" naming',
            "  the constraint, what the literal instruction asked for, and what you",
            "  did instead.",
        ]
        return "\n".join(lines)


#: The profile used across the pipeline.
DEFAULT_PROFILE = RobotProfile()


@dataclass
class CapabilityWorkaround:
    """A single planner-reported adjustment forced by the profile."""

    constraint: str
    literal_instruction: str
    adopted_approach: str
    outcome_differs: bool = False
    note: str = ""

    def as_dict(self) -> dict:
        return {
            "constraint": self.constraint,
            "literal_instruction": self.literal_instruction,
            "adopted_approach": self.adopted_approach,
            "outcome_differs": self.outcome_differs,
            "note": self.note,
        }
