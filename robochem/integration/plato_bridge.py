"""
Bridge between robomail_Aliyah's agent pipeline and the robochem skills.

robomail_Aliyah's orchestrator plans in a closed six-action vocabulary
(``agents/action_vocabulary.py``) and hands each ``PlannedStep`` to
``skills/executor.py``, which is a deliberate stub: it replays the LLM's
centimetre-level GOTO/GRASP/TILT primitives at a mock arm and then invents a
success verdict from ``FAKE_SUCCESS_RATE``. Nothing moves.

This module is the real executor. ``RoboChemExecutor`` is a drop-in replacement
for that stub: same ``execute(step) -> result`` shape, same result fields, but
it dispatches to the perception-driven robochem skills and reports what the
robot actually did.

Two things about that swap are load-bearing and are NOT hidden:

1.  **The motion primitives are not replayed.** A robochem skill re-grounds its
    target with SAM 3 and the camera cage, computes its own grasp or rim
    geometry, and verifies arrival. Feeding it an LLM's "move 4cm in -X" on top
    of that would fight it. The primitives are still recorded in the result, so
    the trial log keeps the planner's real output, but they are provenance, not
    commands. See PRIMITIVES_ARE_ADVISORY below.

2.  **That changes one of the repo's stated invariants.** robomail_Aliyah's
    definition-of-done says "no hand-scripted / fixed-trajectory motion code
    anywhere; all motion comes from the LLM Step Planner". robochem skills are
    hand-written parameterised motion. Under this bridge the LLM still chooses
    *which* skill and *with what arguments*, but it no longer authors the
    trajectory. That is a deliberate architectural trade — real manipulation for
    a weaker autonomy claim — and it must be restated in the paper rather than
    quietly inherited. ``executor_provenance()`` returns the wording to log.

Import safety: nothing from robomail_Aliyah is imported at module load. Steps
are consumed structurally (``.action``, ``.target_object``, ``.tool``,
``.location``, ``.primitives``), so this module is importable and testable on a
machine that has neither that package nor frankapy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import re
import time

#: Aliyah's six-action vocabulary -> robochem skill names.
#: Keep this exhaustive: an action with no mapping is a reported failure, never
#: a silent no-op, exactly as InfeasibleStep is on the planning side.
ACTION_TO_SKILL: Dict[str, str] = {
    "PICKUP": "pick_up",
    "POUR": "pour",
    "SCOOP": "scoop",
    "PIPETTE_DISPENSE": "dispense",
    "STIR": "stir",
    "PLACE": "place",
}

#: Actions that require something already in the gripper. Mirrors
#: ``REQUIRES_HELD_OBJECT`` in agents/action_vocabulary.py; duplicated rather
#: than imported so this module stays standalone. If that list changes there,
#: change it here — the bridge test asserts the two agree when both are present.
REQUIRES_HELD_OBJECT = frozenset({"POUR", "SCOOP", "PIPETTE_DISPENSE", "STIR", "PLACE"})

#: Recorded on every result so a reader of the trial log knows the LLM's
#: primitives were provenance rather than the commands that moved the arm.
PRIMITIVES_ARE_ADVISORY = (
    "LLM motion primitives were logged but not executed; the robochem skill "
    "re-grounded the target with SAM 3 and planned its own motion."
)

#: PLATO-style workspace position names, e.g. "Original Position of beaker".
_ORIGINAL_POSITION = re.compile(r"^\s*original position of\s+", re.IGNORECASE)

#: Named poses that are arm configurations, not objects in the scene.
NAMED_POSES = {"home pose", "home", "verification pose"}


def normalize_object_name(name: Optional[str]) -> Optional[str]:
    """
    Turn a planner-supplied name into something SAM 3 can be asked to segment.

    The planner speaks in the robot profile's workspace vocabulary
    ("Original Position of clear cup 1"), which is a *location* label. Our
    grounding service wants an object phrase ("clear cup 1"). Named arm poses
    are passed through unchanged for the caller to special-case.
    """
    if name is None:
        return None
    cleaned = _ORIGINAL_POSITION.sub("", str(name)).strip()
    return cleaned or None


def is_named_pose(name: Optional[str]) -> bool:
    """True for "Home Pose" / "Verification Pose" — arm configs, not objects."""
    if name is None:
        return False
    return str(name).strip().lower() in NAMED_POSES


@dataclass
class SkillExecutionResult:
    """
    Outcome of executing one PlannedStep on the real cell.

    Field-compatible with ``skills/executor.py::ExecutionResult`` so the
    orchestrator's logging path needs no changes — with ``is_fake`` set False,
    which is the whole point.
    """

    action: str
    success: bool
    reason: str
    primitives_issued: int
    simulated_seconds: float
    is_fake: bool = False
    detail: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "success": self.success,
            "reason": self.reason,
            "primitives_issued": self.primitives_issued,
            "simulated_seconds": round(self.simulated_seconds, 2),
            "is_fake": self.is_fake,
            "detail": self.detail,
        }


class UnmappedAction(Exception):
    """Raised when a planned action has no robochem skill behind it."""


class RoboChemExecutor:
    """
    Executes robomail_Aliyah ``PlannedStep``s against the real Franka cell.

    Drop-in for ``skills.executor.SkillExecutor``::

        from robochem.integration.plato_bridge import RoboChemExecutor

        orchestrator = Orchestrator(cell=cell)
        orchestrator.executor = RoboChemExecutor(skills_executor)

    Args:
        skills: a ``robochem.skills.SkillsExecutor``
        param_overrides: per-skill parameter defaults, e.g.
            ``{"pour": {"forward_offset": -0.08}}``. This is where the
            bench-tuned values from progress.md belong — the planner does not
            know them and should not have to.
        arm: optional object with ``acquire``/``release``, so the mock arm's
            single-gripper bookkeeping stays honest (matches what the stub
            executor does).
        dry_run: translate and log, but do not move. Useful for checking the
            action->skill mapping against a real plan with no robot attached.
    """

    def __init__(
        self,
        skills,
        *,
        param_overrides: Optional[Dict[str, Dict[str, Any]]] = None,
        arm=None,
        dry_run: bool = False,
        verbose: bool = True,
    ) -> None:
        self.skills = skills
        self.param_overrides = param_overrides or {}
        self.arm = arm
        self.dry_run = bool(dry_run)
        self.verbose = bool(verbose)

        #: Name of the object currently in the gripper, tracked here because
        #: the Franka cannot tell us *what* it is holding, only how wide it is.
        self.held_object: Optional[str] = None

        #: Where each object was picked up from, so "PLACE at Original Position
        #: of beaker" can be honoured. The held object cannot be visually
        #: grounded at its own original position — it is in the gripper, not
        #: there — so the coordinates have to come from the pick that moved it.
        self.pickup_sites: Dict[str, List[float]] = {}

    # ==================== translation ====================

    def translate(self, step) -> Tuple[str, Dict[str, Any]]:
        """
        Map one PlannedStep onto ``(skill_name, params)``.

        Raises:
            UnmappedAction: if the action has no skill behind it.
        """
        action = self._action_name(step)
        skill_name = ACTION_TO_SKILL.get(action)
        if skill_name is None:
            raise UnmappedAction(
                f"action {action!r} has no robochem skill; known actions are "
                f"{sorted(ACTION_TO_SKILL)}"
            )

        target = normalize_object_name(getattr(step, "target_object", None))
        tool = normalize_object_name(getattr(step, "tool", None))
        location = normalize_object_name(getattr(step, "location", None))

        if action == "PICKUP":
            # PLATO plans name the thing to grasp as the tool when it is an
            # implement and as the target otherwise.
            obj = tool or target or location
            if obj is None:
                raise UnmappedAction("PICKUP with no object to grasp")
            params: Dict[str, Any] = {"object_name": obj}

        elif action == "POUR":
            # The source is whatever is held; only the destination is named.
            dest = target or location
            if dest is None:
                raise UnmappedAction("POUR with no target container")
            params = {"target_container": dest}

        elif action == "SCOOP":
            source = target or location
            if source is None:
                raise UnmappedAction("SCOOP with no powder source")
            params = {"powder_source": source}

        elif action == "PIPETTE_DISPENSE":
            dest = target or location
            if dest is None:
                raise UnmappedAction("PIPETTE_DISPENSE with no target container")
            params = {"target_container": dest}

        elif action == "STIR":
            dest = target or location
            if dest is None:
                raise UnmappedAction("STIR with no container to stir")
            params = {"target_container": dest}

        elif action == "PLACE":
            dest = location or target
            if dest is None:
                raise UnmappedAction("PLACE with no target location")
            params = {"target_location": self._resolve_place_target(dest)}

        else:  # pragma: no cover - guarded by ACTION_TO_SKILL above
            raise UnmappedAction(f"unhandled action {action!r}")

        # Bench-tuned defaults last so they win over anything inferred here,
        # and so an operator can pin e.g. pour's validated forward_offset.
        params.update(self.param_overrides.get(skill_name, {}))
        return skill_name, params

    def _resolve_place_target(self, dest: str):
        """
        Turn a PLACE destination into something the place skill can use.

        "Original Position of beaker" while holding the beaker cannot be
        segmented — the beaker is in the gripper. Fall back to the coordinates
        recorded when we picked it up.
        """
        if is_named_pose(dest):
            # Nothing to ground; hand the skill the bench in front of the base.
            return [0.45, 0.0]

        if self.held_object and dest.lower() == self.held_object.lower():
            site = self.pickup_sites.get(self.held_object)
            if site is not None:
                if self.verbose:
                    print(f"[Bridge] PLACE back at '{dest}' -> recorded pick site "
                          f"{[round(v, 3) for v in site]}")
                return list(site)
            if self.verbose:
                print(f"[Bridge] PLACE back at '{dest}' but no pick site was "
                      f"recorded; falling back to the default bench spot")
            return [0.45, 0.0]

        return dest

    # ==================== execution ====================

    def execute(self, step) -> SkillExecutionResult:
        """Execute one PlannedStep. Never raises; failures come back as results."""
        action = self._action_name(step)
        primitives = list(getattr(step, "primitives", []) or [])
        started = time.time()

        try:
            skill_name, params = self.translate(step)
        except UnmappedAction as exc:
            return SkillExecutionResult(
                action=action,
                success=False,
                reason=f"could not translate the planned step: {exc}",
                primitives_issued=len(primitives),
                simulated_seconds=time.time() - started,
                detail={"failure_stage": "translation"},
            )

        # Precondition the planner's sequence checker cannot see: the *real*
        # gripper state. A plan that passes validate_sequence can still arrive
        # here with an empty gripper because an earlier pick actually failed.
        if action in REQUIRES_HELD_OBJECT and self.held_object is None:
            return SkillExecutionResult(
                action=action,
                success=False,
                reason=(f"{action} needs a held object but nothing is recorded as "
                        f"held; an earlier PICKUP must have failed"),
                primitives_issued=len(primitives),
                simulated_seconds=time.time() - started,
                detail={"failure_stage": "precondition", "skill": skill_name},
            )

        if self.verbose:
            print(f"[Bridge] {action} -> skill '{skill_name}' params={params}")

        if self.dry_run:
            return SkillExecutionResult(
                action=action,
                success=True,
                reason=f"DRY RUN: would run '{skill_name}' with {params}",
                primitives_issued=len(primitives),
                simulated_seconds=time.time() - started,
                is_fake=True,
                detail={"skill": skill_name, "params": params, "dry_run": True},
            )

        success, result = self.skills.execute(skill_name, params)
        elapsed = time.time() - started

        self._update_held_state(action, success, params, result)

        reason = (
            f"{skill_name} completed: {self._summarize(result)}"
            if success
            else f"{skill_name} failed: {result.get('error', 'no reason reported')}"
        )

        return SkillExecutionResult(
            action=action,
            success=bool(success),
            reason=reason,
            primitives_issued=len(primitives),
            simulated_seconds=elapsed,
            is_fake=False,
            detail={
                "skill": skill_name,
                "params": params,
                "skill_result": result,
                "held_object": self.held_object,
                "primitives_note": PRIMITIVES_ARE_ADVISORY,
                "planner_primitives": [
                    p.as_dict() if hasattr(p, "as_dict") else p for p in primitives
                ],
            },
        )

    def _update_held_state(self, action: str, success: bool,
                           params: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Keep our idea of what is in the gripper in step with reality."""
        if action == "PICKUP" and success:
            self.held_object = params.get("object_name")
            centroid = (result or {}).get("centroid")
            if centroid is not None and self.held_object:
                # Remember where it came from, so PLACE can put it back.
                self.pickup_sites[self.held_object] = list(centroid)
            self._arm_call("acquire", self.held_object)

        elif action == "PLACE" and success:
            self._arm_call("release")
            self.held_object = None

        elif not success and (result or {}).get("still_holding") is False:
            # A skill that reports it dropped the object corrects our state.
            self._arm_call("release")
            self.held_object = None

    def _arm_call(self, method: str, *args) -> None:
        """Forward gripper bookkeeping to the arm object, if it wants it."""
        fn = getattr(self.arm, method, None)
        if callable(fn):
            try:
                fn(*args)
            except Exception as exc:  # pragma: no cover - bookkeeping only
                print(f"[Bridge] arm.{method} failed: {exc}")

    @staticmethod
    def _action_name(step) -> str:
        """Action name as a plain string, whether it is an Enum or already str."""
        action = getattr(step, "action", None)
        return str(getattr(action, "value", action)).upper()

    @staticmethod
    def _summarize(result: Optional[Dict[str, Any]]) -> str:
        """One-line digest of a skill result for the trial log."""
        if not result:
            return "no detail reported"
        keys = ("grasped_object", "poured_into", "scooped_from", "stirred_container",
                "drops_dispensed", "placed_at", "aspirated_from")
        parts = [f"{k}={result[k]}" for k in keys if k in result]
        return ", ".join(parts) if parts else "ok"


def executor_provenance() -> Dict[str, Any]:
    """
    What to record in the trial log about how motion was produced.

    robomail_Aliyah's logging schema records ``is_fake`` per step; this adds the
    architectural caveat that goes with a real executor, so the claim in the
    paper matches the code that ran.
    """
    return {
        "executor": "robochem.integration.plato_bridge.RoboChemExecutor",
        "is_fake": False,
        "motion_source": "robochem parameterised skills (hand-written)",
        "llm_authors_trajectory": False,
        "llm_authors_skill_and_arguments": True,
        "note": PRIMITIVES_ARE_ADVISORY,
        "caveat": (
            "The 'no hand-scripted motion code' criterion in "
            "robomail_Aliyah/docs/progress.md does not hold under this "
            "executor. The LLM selects the skill and its arguments; the "
            "trajectory inside each skill is written by hand and grounded by "
            "SAM 3 perception."
        ),
    }


def build_position_lookup(vision, position_names, verbose: bool = True) -> Dict[str, Any]:
    """
    Ground the robot profile's workspace positions into base-frame coordinates.

    ``hardware/real.py::RealFrankaArm.goto_delta`` raises unless it is given a
    ``position_lookup``, with the comment that "the position lookup is populated
    by the perception stack at run start". This is that function — useful if
    anyone does want to run the primitive-replay path against the real arm
    rather than the skills path.

    Names that cannot be segmented are omitted rather than guessed, so a caller
    can see exactly which positions are ungrounded.
    """
    lookup: Dict[str, Any] = {}
    for name in position_names:
        if is_named_pose(name):
            continue
        query = normalize_object_name(name)
        if query is None:
            continue
        centroid = vision.get_object_centroid(query)
        if centroid is None:
            if verbose:
                print(f"[Bridge] could not ground workspace position {name!r} "
                      f"(query {query!r})")
            continue
        lookup[name] = centroid
        if verbose:
            print(f"[Bridge] grounded {name!r} -> {[round(float(v), 3) for v in centroid]}")
    return lookup
