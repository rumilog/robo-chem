"""
The closed loop: LLM agents in, real skills out.

Merged from robomail_Aliyah/orchestrator.py, which ran

    booklet page -> goal -> scene -> plan -> affordance -> motion primitives
                 -> FAKE executor (the arm never moved) -> verify -> replan

Everything up to the executor is kept, and the executor is replaced by
:class:`robochem.skills.SkillsExecutor`, which is the same object whether the
cell is the Franka in the camera cage or the MuJoCo model of it. So:

    task (or instruction photo)
      -> Scene agent          what is on the bench, named so a skill can find it
      -> Planner              ordered sub-tasks
      -> per sub-task:
           Skill-call agent   one executable call, chosen from SKILL_REGISTRY
           SkillsExecutor     the arm actually moves
      -> ChemistryVerifier    real CV on real (or rendered) frames
      -> on failure: the planner replans, up to ``max_replans``
      -> one JSON trial record

Differences from upstream worth knowing about:

  * The gripper state fed back to the agents is measured from the arm, not
    assumed from the plan. Upstream's step planner was told to "assume every
    upstream step already succeeded"; here a tool that was dropped shows up as a
    disagreement between what we think is held and what the jaws report, and
    that disagreement is what gets handed to the replanner.
  * Verification is skippable, and skipped by default in simulation. MuJoCo has
    no chemistry -- the granules are bouncy spheres -- so a colour-change check
    there fails for reasons that have nothing to do with the plan, and every
    failure burns a replanning attempt. See ``verify``.
  * There is no retry-the-same-call recovery. The old VLMOrchestrator re-ran a
    failed step up to three times unchanged, which for a skill that failed
    because it could not see its target is three identical failures. A failure
    goes to the planner, which can change the approach.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from robochem.agents import config as agent_config
from robochem.agents import planner as planner_agent
from robochem.agents import scene as scene_agent
from robochem.agents import skill_planner
from robochem.agents.llm_client import LLMResponseError, MissingAPIKeyError
from robochem.agents.planner import Plan, SubTask
from robochem.agents.robot_profile import DEFAULT_PROFILE, RobotProfile
from robochem.agents.skill_catalog import CATALOG, InfeasibleSkill
from robochem.agents.skill_planner import SkillCall, StepOutcome

#: Default cap on corrective replanning attempts.
MAX_REPLANS = 2

#: Guard against a planner that emits an unbounded list of sub-tasks.
MAX_SUBTASKS = 24


@dataclass
class StepRecord:
    """One executed (or refused) sub-task, for the trial log."""

    attempt: int
    subtask_index: int
    subtask: str
    skill: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    success: bool = False
    failure_kind: Optional[str] = None      # planning | execution | None
    reason: str = ""
    result: Dict[str, Any] = field(default_factory=dict)
    seconds: float = 0.0

    def as_dict(self) -> dict:
        return {
            "attempt": self.attempt,
            "subtask_index": self.subtask_index,
            "subtask": self.subtask,
            "skill": self.skill,
            "params": self.params,
            "success": self.success,
            "failure_kind": self.failure_kind,
            "reason": self.reason,
            "result": self.result,
            "seconds": round(self.seconds, 2),
        }


@dataclass
class TaskOutcome:
    """What one run produced."""

    success: bool
    outcome: str                # success | failed | blocked
    goal: Optional[str]
    reason: str
    replans: int
    steps: List[StepRecord] = field(default_factory=list)
    record: Dict[str, Any] = field(default_factory=dict)
    log_path: Optional[str] = None

    def as_dict(self) -> dict:
        return {
            "success": self.success,
            "outcome": self.outcome,
            "goal": self.goal,
            "reason": self.reason,
            "replans": self.replans,
            "steps": [s.as_dict() for s in self.steps],
            "log_path": self.log_path,
        }


class AgentOrchestrator:
    """Runs one task end to end, closed loop and unattended."""

    def __init__(
        self,
        skills_executor,
        vision_system,
        robot=None,
        *,
        verifier=None,
        profile: RobotProfile = DEFAULT_PROFILE,
        max_replans: int = MAX_REPLANS,
        verify: bool = True,
        dry_run: bool = False,
        log_dir: Optional[str] = "experiments",
        llm_client_override=None,
        verbose: bool = True,
    ) -> None:
        """
        Args:
            skills_executor: :class:`~robochem.skills.SkillsExecutor` over the
                arm, real or simulated.
            vision_system: The same ``VisionSystem`` those skills use. Frames
                for scene understanding and verification come from it.
            robot: The arm, for reading gripper width. Optional -- without it the
                held-object state is bookkeeping only, and says so.
            verifier: :class:`ChemistryVerifier`, or None to build one lazily.
            profile: Robot capability profile injected into every prompt.
            max_replans: Corrective attempts before reporting failure.
            verify: Run the chemistry check at the end. Turn this off in
                simulation, where there is no chemistry to check.
            dry_run: Plan and print every skill call without executing any of
                them. The LLM half of the pipeline runs in full; the arm does
                not move. This is the cheap way to see what the model would do.
            log_dir: Where to write the trial record, or None for no file.
            llm_client_override: Transport double, for tests.
            verbose: Narrate to stdout.
        """
        self.skills = skills_executor
        self.vision = vision_system
        self.robot = robot
        self._verifier = verifier
        self.profile = profile
        self.max_replans = int(max_replans)
        self.verify = bool(verify)
        self.dry_run = bool(dry_run)
        self.log_dir = log_dir
        self.llm = llm_client_override
        self.verbose = verbose

        # What we believe is in the gripper. The arm is the authority on whether
        # anything is held; only the name is bookkeeping.
        self.held_object: Optional[str] = None
        # Where the held tool's working end sits relative to the grasp, as the
        # pick_up that grasped it measured off the point cloud. Not a property
        # of the tool -- it moves with the grasp -- so it is re-measured every
        # time something is picked up and forgotten the moment it is put down.
        self.held_tool_offset: Optional[List[float]] = None
        self.history: List[StepOutcome] = []

    # ------------------------------------------------------------- narration
    def _say(self, message: str) -> None:
        if self.verbose:
            print(message)

    # ------------------------------------------------------------ the run
    def run(
        self,
        task: Optional[str] = None,
        *,
        instruction_image: Optional[str] = None,
        experiment_name: Optional[str] = None,
    ) -> TaskOutcome:
        """
        Run one task from a description (or an instruction photo) to a verdict.

        Never raises for an ordinary failure: a missing API key, an infeasible
        plan, a skill that could not find its target and a spent retry budget all
        come back as a reported outcome with a reason.
        """
        started = time.time()
        self.history = []
        self.held_object = None
        self.held_tool_offset = None
        record: Dict[str, Any] = {
            "started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "models": agent_config.describe(),
            "dry_run": self.dry_run,
            "verify": self.verify,
            "max_replans": self.max_replans,
            "provenance": _provenance(),
            "events": [],
        }

        # --- the goal ----------------------------------------------------
        if instruction_image:
            self._say(f"[goal] reading instructions from {instruction_image}")
            try:
                parsed = self.vision.instruction_parser.parse_instruction_image(
                    instruction_image
                )
            except Exception as exc:
                return self._blocked(record, "instruction_parse", str(exc), None, started)
            record["instruction"] = parsed
            task = parsed.get("goal") or task
        if not task:
            return self._blocked(
                record, "goal", "no task description and no instruction image", None, started
            )
        record["goal"] = task
        self._say(f"[goal] {task}")

        # --- the scene ---------------------------------------------------
        try:
            scene = scene_agent.comprehend_scene(
                self.vision, task, profile=self.profile, client=self.llm
            )
        except MissingAPIKeyError as exc:
            return self._blocked(record, "scene_understanding", str(exc), task, started)
        except (LLMResponseError, RuntimeError) as exc:
            return self._blocked(record, "scene_understanding", str(exc), task, started)
        record["scene"] = scene.as_dict()
        self._say(f"[scene] {len(scene.objects)} objects: "
                  f"{', '.join(scene.names) or '(none)'}")
        for name in scene.unresolvable:
            self._say(f"[scene] WARNING: '{name}' is not a name this cell can locate")

        # --- the plan ----------------------------------------------------
        try:
            plan = planner_agent.plan_goal(
                task, scene, profile=self.profile, client=self.llm
            )
        except MissingAPIKeyError as exc:
            return self._blocked(record, "planning", str(exc), task, started)
        except LLMResponseError as exc:
            return self._blocked(record, "planning", str(exc), task, started)
        record["plans"] = [plan.as_dict()]
        self._announce_plan(plan)

        if not plan.feasible:
            return self._finish(
                record, False, "failed", task,
                plan.infeasible_reason or "the planner reported the goal infeasible",
                0, [], started,
            )

        # --- execute, verify, replan -------------------------------------
        # Four camera captures, so only when something will compare against them.
        before = self._frames() if (self.verify and not self.dry_run) else []
        steps: List[StepRecord] = []
        completed: List[SubTask] = []
        attempt = 0
        replans = 0
        active = plan
        reason = ""

        while True:
            ok, failed_subtask, failure, blocker = self._execute_plan(
                active, scene, steps, completed, attempt
            )
            if blocker is not None:
                return self._blocked(record, blocker[0], blocker[1], task, started,
                                     replans=replans, steps=steps)

            if ok and self.dry_run:
                return self._finish(
                    record, True, "success", task,
                    "dry run: every sub-task resolved to a valid skill call; "
                    "nothing was executed",
                    replans, steps, started,
                )

            if ok and self.verify:
                verdict = self._verify(active, before)
                record.setdefault("verifications", []).append(verdict)
                self._say(f"[verify] {'PASS' if verdict.get('success') else 'FAIL'} "
                          f"({verdict.get('verification_type')}) "
                          f"{verdict.get('details', '')}")
                if verdict.get("success"):
                    return self._finish(record, True, "success", task,
                                        str(verdict.get("details", "verified")),
                                        replans, steps, started)
                reason = (f"every sub-task executed but the outcome was not observed: "
                          f"{verdict.get('details', 'no detail')}")
            elif ok:
                return self._finish(
                    record, True, "success", task,
                    "every sub-task executed; chemistry verification was not run",
                    replans, steps, started,
                )
            else:
                reason = failure or "a sub-task could not be carried out"

            if attempt >= self.max_replans:
                break

            attempt += 1
            replans += 1
            self._say(f"\n[replan] attempt {attempt}/{self.max_replans}: {reason}")
            try:
                active = planner_agent.replan(
                    task, scene, active,
                    failed_subtask=failed_subtask,
                    completed_subtasks=completed,
                    failure_reason=reason,
                    held_object=self._held_description(),
                    attempt=attempt,
                    max_attempts=self.max_replans,
                    profile=self.profile,
                    client=self.llm,
                )
            except MissingAPIKeyError as exc:
                return self._blocked(record, "replanning", str(exc), task, started,
                                     replans=replans, steps=steps)
            except LLMResponseError as exc:
                return self._blocked(record, "replanning", str(exc), task, started,
                                     replans=replans, steps=steps)
            record["plans"].append(active.as_dict())
            if active.diagnosis:
                self._say(f"[replan] diagnosis: {active.diagnosis}")
            self._announce_plan(active)
            if not active.feasible:
                return self._finish(
                    record, False, "failed", task,
                    f"the planner gave up during correction: {active.infeasible_reason}",
                    replans, steps, started,
                )

        return self._finish(
            record, False, "failed", task,
            f"exhausted {self.max_replans} corrective attempts; last failure was: {reason}",
            replans, steps, started,
        )

    # ------------------------------------------------------------ internals
    def _announce_plan(self, plan: Plan) -> None:
        self._say(f"[plan] {len(plan.subtasks)} sub-tasks")
        for s in plan.subtasks:
            self._say(f"   {s.index}. {s.description}")
        if plan.expected_observation:
            self._say(f"[plan] expecting: {plan.expected_observation}")
        for w in plan.capability_workarounds:
            self._say(f"[capability] {w.constraint} -> {w.adopted_approach}")

    def _execute_plan(self, plan: Plan, scene, steps: List[StepRecord],
                      completed: List[SubTask], attempt: int):
        """
        Resolve and run every sub-task of ``plan``.

        Returns ``(all_ok, failed_subtask, reason, blocker)``. ``blocker`` is set
        only for a stage that could not run at all (no API key), which is not a
        failure of the plan and must not be replanned around.
        """
        if len(plan.subtasks) > MAX_SUBTASKS:
            return (False, None,
                    f"the planner emitted {len(plan.subtasks)} sub-tasks, more than the "
                    f"{MAX_SUBTASKS} this loop will run", None)

        for subtask in plan.subtasks:
            record = StepRecord(attempt=attempt, subtask_index=subtask.index,
                                subtask=subtask.description)
            started = time.time()

            try:
                call = skill_planner.plan_skill_call(
                    subtask, scene,
                    held_object=self._held_description(),
                    history=self.history,
                    profile=self.profile,
                    client=self.llm,
                )
            except MissingAPIKeyError as exc:
                return False, subtask, str(exc), ("skill_planning", str(exc))
            except LLMResponseError as exc:
                return False, subtask, str(exc), ("skill_planning", str(exc))
            except InfeasibleSkill as exc:
                record.success = False
                record.failure_kind = "planning"
                record.reason = exc.reason
                record.seconds = time.time() - started
                steps.append(record)
                self._say(f"[step {subtask.index}] INFEASIBLE: {exc.reason}")
                return (False, subtask,
                        f"sub-task {subtask.index} ({subtask.description!r}) could not be "
                        f"expressed as a skill call: {exc.reason}", None)

            record.skill = call.skill
            record.params = dict(call.params)
            self._say(f"\n[step {subtask.index}] {subtask.description}")
            self._say(f"   -> {call.as_command()}")
            if call.rationale:
                self._say(f"   ({call.rationale})")

            if self.dry_run:
                record.success = True
                record.reason = "dry run: not executed"
                record.seconds = time.time() - started
                steps.append(record)
                self.history.append(StepOutcome(call, True, "dry run"))
                self._apply_effect(call, True)   # no measurement without a grasp
                completed.append(subtask)
                continue

            success, result = self.skills.execute(call.skill, call.params)
            record.success = bool(success)
            record.result = _jsonable(result)
            record.reason = (
                "" if success else str(result.get("error", "skill reported failure"))
            )
            record.failure_kind = None if success else "execution"
            record.seconds = time.time() - started
            steps.append(record)
            self.history.append(
                StepOutcome(call, bool(success), record.reason or "ok")
            )
            self._apply_effect(call, bool(success), record.result)
            self._say(f"   {'ok' if success else 'FAILED'}"
                      + (f": {record.reason}" if record.reason else ""))

            if not success:
                return (False, subtask,
                        f"sub-task {subtask.index} ({subtask.description!r}) failed: "
                        f"{call.skill} reported {record.reason}", None)
            completed.append(subtask)

        return True, None, "", None

    def _apply_effect(self, call: SkillCall, success: bool, result=None) -> None:
        """Update what we believe the gripper holds, from the call's declared effect."""
        if not success:
            return
        if call.skill == "pick_up":
            self.held_object = call.params.get("object_name")
            offset = (result or {}).get("suggested_tool_offset")
            self.held_tool_offset = (
                [float(v) for v in offset] if offset is not None else None
            )
        elif call.skill in ("place", "open_gripper"):
            self.held_object = None
            self.held_tool_offset = None

    def _held_description(self) -> Optional[str]:
        """
        What the gripper holds, with the arm's own reading folded in.

        The name is bookkeeping and can be wrong -- a dropped tool leaves it
        stale. The jaw width is measured. When the two disagree, say so: that
        disagreement is exactly what a replanner needs to see.
        """
        name = self.held_object
        if self.robot is None or self.dry_run:
            return self._with_offset(name)
        try:
            width = float(self.robot.get_gripper_width())
        except Exception:
            return self._with_offset(name)
        holding = 0.001 < width < 0.075
        if holding and name:
            return self._with_offset(f"{name} (jaws at {width * 1000:.0f} mm)")
        if holding:
            return f"something unidentified (jaws at {width * 1000:.0f} mm)"
        if name:
            return (f"nothing -- the jaws read {width * 1000:.0f} mm, so the {name} it "
                    f"was holding has been dropped or was never grasped")
        return None

    def _with_offset(self, description: Optional[str]) -> Optional[str]:
        """Append the measured tool offset, so the model can pass it on verbatim."""
        if not description or self.held_tool_offset is None:
            return description
        x, y, z = self.held_tool_offset
        return (
            f"{description}. Its working end was measured at "
            f"tool_offset = [{x:.4f}, {y:.4f}, {z:.4f}] metres from the grasp point, "
            f"in the tool frame. Pass this verbatim to any skill that takes "
            f"tool_offset, and its z ({z:.4f}) to any that takes tool_length. The z "
            f"is a lower bound -- the cage looks down, so the underside of a bowl is "
            f"occluded and the cloud stops at its rim"
        )

    def _frames(self) -> List[np.ndarray]:
        """Camera frames in BGR, which is what the verifier's detectors expect."""
        frames = self.vision.capture_scene() or []
        if getattr(self.vision, "frame_color_order", "bgr") == "rgb":
            import cv2
            frames = [cv2.cvtColor(np.asarray(f), cv2.COLOR_RGB2BGR) for f in frames]
        return list(frames)

    def _verify(self, plan: Plan, before: List[np.ndarray]) -> dict:
        if self._verifier is None:
            from robochem.verification.chemistry_verifier import ChemistryVerifier
            self._verifier = ChemistryVerifier()
        after = self._frames()
        expected = plan.expected_observation or plan.goal
        try:
            return self._verifier.verify_chemistry_outcome(expected, after, before)
        except Exception as exc:
            return {
                "success": False,
                "verification_type": "error",
                "details": f"verification could not run: {exc}",
            }

    # -------------------------------------------------------------- endings
    def _blocked(self, record, stage, reason, goal, started,
                 *, replans=0, steps=None) -> TaskOutcome:
        """A stage that could not run at all, named explicitly."""
        record.setdefault("events", []).append({"blocked_at": stage, "reason": reason})
        self._say(f"[blocked] {stage}: {reason}")
        return self._finish(record, False, "blocked", goal,
                            f"pipeline blocked at {stage}: {reason}",
                            replans, steps or [], started)

    def _finish(self, record, success, outcome, goal, reason,
                replans, steps, started) -> TaskOutcome:
        record["outcome"] = outcome
        record["success"] = success
        record["reason"] = reason
        record["replans"] = replans
        record["seconds"] = round(time.time() - started, 1)
        record["steps"] = [s.as_dict() for s in steps]
        path = self._write_log(record)
        self._say(f"\n[{outcome}] {reason}")
        if path:
            self._say(f"[log] {path}")
        return TaskOutcome(success, outcome, goal, reason, replans, list(steps),
                           record, path)

    def _write_log(self, record: dict) -> Optional[str]:
        if not self.log_dir:
            return None
        try:
            out = Path(self.log_dir)
            out.mkdir(parents=True, exist_ok=True)
            path = out / f"agent_run_{time.strftime('%Y%m%d_%H%M%S')}.json"
            path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
            return str(path)
        except OSError as exc:
            self._say(f"[log] could not write the trial record: {exc}")
            return None


def _provenance() -> dict:
    """
    How motion was produced, recorded on every run.

    robomail_Aliyah's definition of done says "no hand-scripted or
    fixed-trajectory motion code anywhere; all motion comes from the LLM Step
    Planner". That does not hold here and must not be quietly inherited: the
    model chooses the skill and its arguments, and the trajectory inside each
    skill is hand-written and grounded by SAM 3. That is a deliberate trade --
    real manipulation for a weaker autonomy claim -- and it belongs in the log
    so the paper's claim matches the code that ran. The same wording is in
    :func:`robochem.integration.plato_bridge.executor_provenance`.
    """
    return {
        "orchestrator": "robochem.orchestrator.agent_orchestrator.AgentOrchestrator",
        "is_fake": False,
        "motion_source": "robochem parameterised skills (hand-written, SAM 3 grounded)",
        "llm_authors_trajectory": False,
        "llm_authors_skill_and_arguments": True,
        "caveat": (
            "The 'no hand-scripted motion code' criterion in "
            "robomail_Aliyah/docs/progress.md does not hold under this "
            "orchestrator. The LLM selects the skill and its arguments; the "
            "trajectory inside each skill is written by hand."
        ),
    }


def _jsonable(value):
    """Strip numpy and point clouds out of a skill result before it is logged."""
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items() if k != "points"}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value][:32]
    if isinstance(value, np.ndarray):
        return value.tolist() if value.size <= 32 else f"<ndarray {value.shape}>"
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value
