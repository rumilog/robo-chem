"""
Offline checks for the LLM agent pipeline — no API key, no robot, no cameras.

What these cover is the machinery around the model: the closed vocabulary, the
parameter checking, the gripper bookkeeping, the replanning loop, the ordering
of stages and what lands in the trial record. The model itself is replaced by a
transport double (:class:`FakeLLM`), in exactly the way the skills tests replace
the arm — so this runs on a laptop with numpy and nothing else.

What these do NOT cover, and cannot: whether a real model plans these tasks well.
That needs a key and a bench (or the simulated one):

    perception_env/bin/python scripts/run_experiment.py --sim --dry-run \
        --task "Scoop citric acid into the white paper cup"

Usage:
    perception_env/bin/python scripts/test_agents_offline.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PASS, FAIL = [], []


def check(label, condition, detail=""):
    if condition:
        PASS.append(label)
        print(f"  ok    {label}")
    else:
        FAIL.append(label)
        print(f"  FAIL  {label}" + (f"  [{detail}]" if detail else ""))


# ==========================================================================
# Transport double
# ==========================================================================

class _Message:
    def __init__(self, content): self.content = content


class _Choice:
    def __init__(self, message): self.message = message


class _Completion:
    def __init__(self, choices): self.choices = choices


class FakeLLM:
    """
    Quacks like ``openai.OpenAI`` for :func:`llm_client.structured_completion`.

    Canned responses are keyed by schema name, so a test says what each agent
    returns without caring about call order. A value may be a callable
    ``(call_index, request) -> dict`` for responses that must change between
    calls, which is how the replanning test makes the second plan differ.
    """

    def __init__(self, responses):
        self._responses = responses
        self.calls = []
        self._counts = {}
        self.chat = _Chat(self)

    def _respond(self, request):
        fmt = request.get("response_format", {})
        name = fmt.get("json_schema", {}).get("name", "")
        messages = request.get("messages", [])
        user = messages[1]["content"] if len(messages) > 1 else []
        text = " ".join(b.get("text", "") for b in user if isinstance(b, dict)) \
            if isinstance(user, list) else str(user)
        images = sum(1 for b in user if isinstance(b, dict)
                     and b.get("type") == "image_url") if isinstance(user, list) else 0
        self.calls.append({
            "schema": name,
            "model": request.get("model", ""),
            "system": messages[0]["content"] if messages else "",
            "user": text,
            "images": images,
        })
        index = self._counts.get(name, 0)
        self._counts[name] = index + 1
        if name not in self._responses:
            raise AssertionError(f"FakeLLM has no canned response for {name!r}; "
                                 f"known: {sorted(self._responses)}")
        payload = self._responses[name]
        if callable(payload):
            payload = payload(index, request)
        return _Completion([_Choice(_Message(json.dumps(payload)))])

    def calls_for(self, name):
        return [c for c in self.calls if c["schema"] == name]


class _Chat:
    def __init__(self, parent): self.completions = _Completions(parent)


class _Completions:
    def __init__(self, parent): self._parent = parent
    def create(self, **request): return self._parent._respond(request)


# ==========================================================================
# Cell doubles
# ==========================================================================

class FakeVision:
    """Enough VisionSystem for the agents: frames, colour order, an inventory."""

    frame_color_order = "rgb"

    def __init__(self, inventory=("larger spoon", "citric acid cup", "white paper cup")):
        self._inventory = list(inventory)
        self.captures = 0

    def capture_scene(self):
        self.captures += 1
        return [np.zeros((48, 64, 3), dtype=np.uint8) for _ in range(4)]

    def known_object_names(self):
        return list(self._inventory)


class FakeArm:
    """Reports a gripper width, which is what the orchestrator actually reads."""

    def __init__(self, width=0.08):
        self.width = width

    def get_gripper_width(self):
        return self.width


class RecordingSkills:
    """A SkillsExecutor that records calls and can be told to fail some."""

    def __init__(self, fail=(), results=None):
        self.calls = []
        self.fail = set(fail)
        self.results = results or {}

    def execute(self, name, params):
        self.calls.append((name, dict(params)))
        if name in self.fail:
            return False, {"error": f"{name} could not find its target"}
        return True, dict(self.results.get(name, {}))


# ==========================================================================
# Canned response builders
# ==========================================================================

def obj(name, category="container", handle=False, contents="unknown", relevant=True):
    return {"name": name, "category": category, "has_handle": handle,
            "contents": contents, "relevant_to_goal": relevant}


def scene_response(*objects, notes=""):
    return {"objects": list(objects), "scene_notes": notes}


def subtask(description, container="none"):
    return {"description": description, "rationale": f"needed to {description}",
            "target_container": container}


def plan_response(subtasks, *, feasible=True, reason="",
                  expected="the powder is in the white paper cup",
                  target="white paper cup", workarounds=None):
    return {"feasible": feasible, "infeasible_reason": reason, "subtasks": subtasks,
            "observation_target": target, "expected_observation": expected,
            "capability_workarounds": workarounds or []}


def corrective_response(subtasks, *, diagnosis="the scoop came up empty", **kwargs):
    payload = plan_response(subtasks, **kwargs)
    payload["diagnosis"] = diagnosis
    return payload


def call_response(skill, params, *, feasible=True, reason=""):
    return {"feasible": feasible, "infeasible_reason": reason, "skill": skill,
            "params": [{"name": k, "value": json.dumps(v)} for k, v in params.items()],
            "rationale": f"{skill} realises this sub-task",
            "expected_outcome": "as planned"}


#: The scoop-and-dump run, which is the one the sim exercises.
SCOOP_SUBTASKS = [
    subtask("pick up the larger spoon"),
    subtask("scoop citric acid", "citric acid cup"),
    subtask("dump the powder into the white paper cup", "white paper cup"),
    subtask("put the larger spoon back down"),
]

SCOOP_CALLS = [
    ("pick_up", {"object_name": "larger spoon"}),
    ("scoop", {"powder_source": "citric acid cup"}),
    ("dump", {"target_container": "white paper cup"}),
    ("place", {"target_location": "table"}),
]


def scoop_responses(**overrides):
    def calls(index, request):
        skill, params = SCOOP_CALLS[index % len(SCOOP_CALLS)]
        return call_response(skill, params)

    base = {
        "scene_description": scene_response(
            obj("citric acid cup", "reagent", contents="white powder"),
            obj("larger spoon", "tool", handle=True, contents="empty"),
            obj("white paper cup", "container", contents="empty"),
        ),
        "subtask_plan": plan_response(SCOOP_SUBTASKS),
        "skill_call": calls,
    }
    base.update(overrides)
    return base


def build(responses, *, skills=None, arm=None, **kwargs):
    from robochem.orchestrator.agent_orchestrator import AgentOrchestrator

    llm = FakeLLM(responses)
    executor = skills if skills is not None else RecordingSkills()
    orchestrator = AgentOrchestrator(
        skills_executor=executor,
        vision_system=FakeVision(),
        robot=arm if arm is not None else FakeArm(),
        max_replans=kwargs.pop("max_replans", 1),
        verify=kwargs.pop("verify", False),
        log_dir=None,
        llm_client_override=llm,
        verbose=False,
        **kwargs,
    )
    return orchestrator, llm, executor


# ==========================================================================
# The catalogue
# ==========================================================================

def test_catalogue_covers_the_registry():
    print("\n[catalogue: agreement with SKILL_REGISTRY]")
    from robochem.skills import SKILL_REGISTRY
    from robochem.agents import skill_catalog as sc

    check("every registered skill is in the catalogue",
          set(sc.CATALOG) == set(SKILL_REGISTRY),
          f"difference: {set(sc.CATALOG) ^ set(SKILL_REGISTRY)}")

    # _audit() runs at import and would have raised; re-run it explicitly so a
    # failure here names this check rather than an import error somewhere else.
    try:
        sc._audit()
        check("the audit passes", True)
    except RuntimeError as exc:
        check("the audit passes", False, str(exc))

    # The audit also compares advertised defaults against the skills' own; it
    # caught seven that had drifted when it was added, which is the whole point.
    from robochem.agents.skill_catalog import Param, SkillSpec
    original = sc.CATALOG["pour"]
    sc.CATALOG["pour"] = SkillSpec(
        name="pour", summary=original.summary, requires=original.requires,
        effect=original.effect,
        params=[Param("target_container", "string", "x", required=True),
                Param("pour_angle", "number", "x", default=12345.0)],
    )
    try:
        sc._audit()
        check("a drifted default is caught", False)
    except RuntimeError as exc:
        check("a drifted default is caught", "12345" in str(exc), str(exc))
    finally:
        sc.CATALOG["pour"] = original

    block = sc.vocabulary_prompt_block()
    check("the prompt block names every skill",
          all(name in block for name in sc.CATALOG))
    check("the prompt block states the scoop/dump pairing",
          "dump" in block and "delivers nothing" in block)


def test_parameters_are_checked():
    print("\n[catalogue: parameter checking]")
    from robochem.agents.skill_catalog import (
        InfeasibleSkill, coerce_params, parse_skill)

    check("a registry name parses", parse_skill("pick_up") == "pick_up")
    check("a spelling variant parses", parse_skill("Arc-Scoop") == "arc_scoop")
    try:
        parse_skill("teleport")
        check("an invented skill is refused", False)
    except InfeasibleSkill:
        check("an invented skill is refused", True)

    params = coerce_params("pick_up", [
        {"name": "object_name", "value": '"plastic beaker"'},
        {"name": "z_offset", "value": "0.02"},
    ])
    check("JSON literals are decoded",
          params == {"object_name": "plastic beaker", "z_offset": 0.02}, f"{params}")

    bare = coerce_params("pick_up", [{"name": "object_name", "value": "plastic beaker"}])
    check("an unquoted string still works",
          bare == {"object_name": "plastic beaker"}, f"{bare}")

    dropped = coerce_params("pick_up", [
        {"name": "object_name", "value": '"beaker"'},
        {"name": "grasp_force", "value": "null"},
    ])
    check("a declined optional parameter is dropped",
          dropped == {"object_name": "beaker"}, f"{dropped}")

    for label, skill, pairs in [
        ("a missing required parameter is refused", "pick_up", []),
        ("an unknown parameter is refused", "pick_up",
         [{"name": "object_name", "value": '"cup"'},
          {"name": "how_hard", "value": "11"}]),
    ]:
        try:
            coerce_params(skill, pairs)
            check(label, False)
        except InfeasibleSkill:
            check(label, True)

    # open_gripper's required "action" is pinned by the executor alias, so the
    # planner must neither have to supply it nor be able to override it.
    try:
        coerce_params("open_gripper", [])
        check("an alias-pinned parameter is not demanded of the planner", True)
    except InfeasibleSkill as exc:
        check("an alias-pinned parameter is not demanded of the planner", False,
              exc.reason)


# ==========================================================================
# The agents
# ==========================================================================

def test_scene_agent_grounds_to_the_inventory():
    print("\n[scene: grounding to what perception can resolve]")
    from robochem.agents import scene as scene_agent

    llm = FakeLLM({"scene_description": scene_response(
        obj("larger spoon", "tool", handle=True),
        obj("measuring scoop", "tool", handle=True),
    )})
    vision = FakeVision()
    scene = scene_agent.comprehend_scene(vision, "scoop the powder", client=llm)

    check("frames were captured from the cell", vision.captures == 1, vision.captures)
    check("the images reached the model", llm.calls[0]["images"] >= 1,
          llm.calls[0]["images"])
    check("the inventory was put in the prompt",
          "larger spoon" in llm.calls[0]["system"])
    check("a name in the inventory is marked resolvable",
          scene.objects[0].resolvable)
    check("a name outside it is marked unresolvable",
          scene.unresolvable == ["measuring scoop"], f"{scene.unresolvable}")
    check("the planner is warned about it in the prompt block",
          "do not use it" in scene.as_prompt_block())
    check("usable names come from the inventory, not the model",
          scene.usable_names() == vision.known_object_names(),
          f"{scene.usable_names()}")


def test_scene_agent_without_an_inventory():
    print("\n[scene: open-vocabulary cell]")
    from robochem.agents import scene as scene_agent

    class NoInventory(FakeVision):
        def known_object_names(self):
            return []

    llm = FakeLLM({"scene_description": scene_response(obj("plastic beaker"))})
    scene = scene_agent.comprehend_scene(NoInventory(), "pour it", client=llm)
    check("every name is usable when the cell cannot enumerate",
          scene.objects[0].resolvable and not scene.unresolvable)
    check("the kit vocabulary is offered instead",
          "Kit vocabulary" in llm.calls[0]["system"])
    check("no inventory line is added to the prompt block",
          "perception can resolve" not in scene.as_prompt_block())


def test_skill_agent_is_told_the_real_gripper_state():
    print("\n[skill call: closed loop, not assumed]")
    from robochem.agents import skill_planner
    from robochem.agents.planner import SubTask
    from robochem.agents.scene import Scene, SceneObject

    scene = Scene(objects=[SceneObject("larger spoon", "tool", True, "empty", True)])
    llm = FakeLLM({"skill_call": call_response("scoop",
                                               {"powder_source": "citric acid cup"})})
    history = [skill_planner.StepOutcome(
        skill_planner.SkillCall("pick_up", {"object_name": "larger spoon"}),
        False, "could not find it")]

    call = skill_planner.plan_skill_call(
        SubTask(2, "scoop citric acid"), scene,
        held_object="nothing -- the jaws read 78 mm", history=history, client=llm)

    prompt = llm.calls[0]["user"]
    check("the measured gripper state is in the prompt",
          "jaws read 78 mm" in prompt)
    check("what actually happened is in the prompt, not an assumption of success",
          "FAILED" in prompt and "could not find it" in prompt, prompt[-300:])
    check("the call comes back executable",
          (call.skill, call.params) == ("scoop", {"powder_source": "citric acid cup"}),
          f"{call.as_dict()}")


def test_infeasible_is_reported_not_approximated():
    print("\n[skill call: infeasible sub-tasks]")
    from robochem.agents import skill_planner
    from robochem.agents.planner import SubTask
    from robochem.agents.scene import Scene
    from robochem.agents.skill_catalog import InfeasibleSkill

    for label, response in [
        ("a sub-task the vocabulary cannot express is refused",
         {"feasible": False, "infeasible_reason": "nothing weighs anything",
          "skill": "wait", "params": [], "rationale": "", "expected_outcome": ""}),
        ("a call with parameters the skill does not take is refused",
         call_response("pour", {"target_container": "cup", "pour_speed": 3})),
        ("a call missing a required parameter is refused",
         call_response("scoop", {})),
    ]:
        llm = FakeLLM({"skill_call": response})
        try:
            skill_planner.plan_skill_call(SubTask(1, "weigh the powder"), Scene(),
                                          client=llm)
            check(label, False)
        except InfeasibleSkill:
            check(label, True)


# ==========================================================================
# The loop
# ==========================================================================

def test_happy_path():
    print("\n[loop: a run that works]")
    orchestrator, llm, skills = build(scoop_responses())
    outcome = orchestrator.run("scoop citric acid into the white paper cup")

    check("the run succeeds", outcome.success, outcome.reason)
    check("every planned sub-task was executed",
          [c[0] for c in skills.calls] == [c[0] for c in SCOOP_CALLS],
          f"{[c[0] for c in skills.calls]}")
    check("the parameters reached the executor unchanged",
          skills.calls[1][1] == {"powder_source": "citric acid cup"},
          f"{skills.calls[1][1]}")
    check("the stages ran in order",
          [c["schema"] for c in llm.calls][:2] == ["scene_description", "subtask_plan"],
          f"{[c['schema'] for c in llm.calls]}")
    check("one skill-call agent invocation per sub-task",
          len(llm.calls_for("skill_call")) == len(SCOOP_SUBTASKS))
    check("no replanning was needed", outcome.replans == 0)
    check("the record marks the motion as real but hand-written",
          outcome.record["provenance"]["is_fake"] is False
          and outcome.record["provenance"]["llm_authors_trajectory"] is False)
    check("every step is in the record",
          len(outcome.record["steps"]) == len(SCOOP_SUBTASKS))


def test_dry_run_moves_nothing():
    print("\n[loop: dry run]")
    orchestrator, llm, skills = build(scoop_responses(), dry_run=True)
    outcome = orchestrator.run("scoop citric acid into the white paper cup")

    check("the dry run reports success", outcome.success, outcome.reason)
    check("the executor was never called", skills.calls == [], f"{skills.calls}")
    check("the model still planned every sub-task",
          len(llm.calls_for("skill_call")) == len(SCOOP_SUBTASKS))
    check("the record says nothing was executed", outcome.record["dry_run"] is True)


def test_a_failed_skill_replans_rather_than_retrying():
    print("\n[loop: failure -> replanning]")
    skills = RecordingSkills(fail={"scoop"})
    orchestrator, llm, _ = build(
        scoop_responses(corrective_plan=corrective_response(
            [subtask("pick up the larger spoon")])),
        skills=skills, max_replans=1)
    outcome = orchestrator.run("scoop citric acid into the white paper cup")

    check("the failing skill was not retried unchanged",
          [c[0] for c in skills.calls].count("scoop") == 1,
          f"{[c[0] for c in skills.calls]}")
    check("the planner was asked for a correction",
          len(llm.calls_for("corrective_plan")) == 1)
    replan_prompt = llm.calls_for("corrective_plan")[0]["user"]
    check("the correction prompt names what failed",
          "could not find its target" in replan_prompt, replan_prompt[-400:])
    check("the correction prompt states what is held",
          "gripper is currently holding" in replan_prompt)
    check("the retry cap is honoured", outcome.replans == 1, outcome.replans)


def test_an_infeasible_plan_stops_without_moving():
    print("\n[loop: the planner refusing]")
    skills = RecordingSkills()
    orchestrator, _, _ = build(
        scoop_responses(subtask_plan=plan_response(
            [], feasible=False, reason="there is no balance on this bench")),
        skills=skills)
    outcome = orchestrator.run("weigh out 2 g of citric acid")

    check("the run fails", not outcome.success)
    check("it fails for the planner's stated reason",
          "no balance" in outcome.reason, outcome.reason)
    check("nothing was executed", skills.calls == [], f"{skills.calls}")


def test_a_missing_key_is_blocked_not_failed():
    print("\n[loop: no API key]")
    from robochem.agents.llm_client import MissingAPIKeyError

    class NoKey(FakeLLM):
        def _respond(self, request):
            raise MissingAPIKeyError("SceneUnderstandingAgent")

    from robochem.orchestrator.agent_orchestrator import AgentOrchestrator
    skills = RecordingSkills()
    outcome = AgentOrchestrator(
        skills_executor=skills, vision_system=FakeVision(), robot=FakeArm(),
        verify=False, log_dir=None, llm_client_override=NoKey({}), verbose=False,
    ).run("scoop citric acid")

    check("a missing key is reported as blocked, not as a failed plan",
          outcome.outcome == "blocked", outcome.outcome)
    check("the blocked stage is named", "scene_understanding" in outcome.reason,
          outcome.reason)
    check("nothing was executed", skills.calls == [], f"{skills.calls}")


def test_the_measured_tool_offset_is_carried_forward():
    print("\n[loop: the tool offset comes from perception, not the model]")
    measured = [0.031, 0.002, 0.047]
    skills = RecordingSkills(
        results={"pick_up": {"suggested_tool_offset": measured}})
    # The jaws must read as closed on something, or the orchestrator correctly
    # reports a dropped tool instead of quoting an offset for one it is not
    # holding -- which is what test_a_dropped_tool_shows_up_as_a_disagreement
    # covers.
    orchestrator, llm, _ = build(scoop_responses(), skills=skills,
                                 arm=FakeArm(width=0.02))
    orchestrator.run("scoop citric acid into the white paper cup")

    scoop_prompt = llm.calls_for("skill_call")[1]["user"]
    check("the scoop agent is given the measurement",
          "0.0310" in scoop_prompt and "tool_offset" in scoop_prompt,
          scoop_prompt[-400:])
    check("it is stated as a lower bound, because it is one",
          "lower bound" in scoop_prompt)

    # The 'place' call is planned while the spoon is still held, so it is still
    # quoted there and should be. What must not survive is the state after the
    # tool is down: a stale offset would be quoted to the next tool's steps.
    check("putting the tool down clears the measurement",
          orchestrator.held_tool_offset is None and orchestrator.held_object is None,
          f"{orchestrator.held_object} / {orchestrator.held_tool_offset}")


def test_a_dropped_tool_shows_up_as_a_disagreement():
    print("\n[loop: bookkeeping vs the jaws]")
    from robochem.orchestrator.agent_orchestrator import AgentOrchestrator

    orchestrator = AgentOrchestrator(
        skills_executor=RecordingSkills(), vision_system=FakeVision(),
        robot=FakeArm(width=0.078),      # wide open: nothing is held
        verify=False, log_dir=None, verbose=False)
    orchestrator.held_object = "larger spoon"

    described = orchestrator._held_description()
    check("an empty gripper contradicting the bookkeeping is reported",
          "dropped" in described, described)

    orchestrator.robot = FakeArm(width=0.008)
    check("a gripper that is holding reads as holding",
          "larger spoon" in orchestrator._held_description())


def test_the_record_is_serialisable():
    print("\n[loop: the trial record]")
    skills = RecordingSkills(results={
        "pick_up": {"centroid": np.array([0.4, 0.1, 0.05]),
                    "points": np.zeros((5000, 3)),
                    "gripper_width": np.float64(0.021)}})
    orchestrator, _, _ = build(scoop_responses(), skills=skills)
    outcome = orchestrator.run("scoop citric acid into the white paper cup")

    try:
        text = json.dumps(outcome.record)
        check("the record is JSON", True)
    except TypeError as exc:
        check("the record is JSON", False, str(exc))
        return
    check("numpy scalars and small arrays survive",
          '"gripper_width": 0.021' in text and "[0.4, 0.1, 0.05]" in text)
    check("point clouds are not written into the log", "points" not in text)


def main():
    print("Offline agent checks (no API key, no robot, no cameras)")
    print("=" * 60)
    test_catalogue_covers_the_registry()
    test_parameters_are_checked()
    test_scene_agent_grounds_to_the_inventory()
    test_scene_agent_without_an_inventory()
    test_skill_agent_is_told_the_real_gripper_state()
    test_infeasible_is_reported_not_approximated()
    test_happy_path()
    test_dry_run_moves_nothing()
    test_a_failed_skill_replans_rather_than_retrying()
    test_an_infeasible_plan_stops_without_moving()
    test_a_missing_key_is_blocked_not_failed()
    test_the_measured_tool_offset_is_carried_forward()
    test_a_dropped_tool_shows_up_as_a_disagreement()
    test_the_record_is_serialisable()

    print("\n" + "=" * 60)
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    for failure in FAIL:
        print(f"  FAILED: {failure}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
