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


def test_scene_agent_on_the_real_bench():
    """
    The real cell with a bench manifest gets the sim's inventory prompt.

    Replays the first real dry run (2026-10-07): the agent described the three
    dishes and three "clear cup"s, and nothing else.
    """
    print("\n[scene: real bench manifest]")
    from robochem.agents import scene as scene_agent
    from robochem.vision.bench_manifest import MAGIC_BEAKER

    class RealBench(FakeVision):
        frame_color_order = "bgr"

        def known_object_names(self):
            return MAGIC_BEAKER.known_object_names()

        def inventory_notes(self):
            return MAGIC_BEAKER.inventory_notes()

    llm = FakeLLM({"scene_description": scene_response(
        obj("citric acid cup"), obj("baking soda cup"), obj("red cabbage powder cup"),
        obj("clear cup a"), obj("clear cup b"), obj("clear cup c"),
    )})
    scene = scene_agent.comprehend_scene(RealBench(), "make a magic beaker", client=llm)
    system = llm.calls[0]["system"]
    check("the prompt carries the inventory, not the kit vocabulary",
          "What is on this bench" in system and "Kit vocabulary" not in system)
    check("the prompt says the scoop stands in for the small one",
          "ONLY scoop" in system)
    names = sorted(o.name for o in scene.objects)
    check("objects the agent left out are listed from the bench",
          names == sorted(o.name for o in MAGIC_BEAKER.objects), f"{names}")
    beaker = next(o for o in scene.objects if o.name == "plastic beaker")
    check("a listed container carries its label as contents",
          "50 ml water" in beaker.contents, beaker.contents)
    check("nothing on the bench is unresolvable", not scene.unresolvable,
          f"{scene.unresolvable}")


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


def test_a_malformed_value_is_dropped_not_passed_on():
    """An optional number the model mangled falls back to the default."""
    print("\n[skill call: malformed values]")
    from robochem.agents.skill_catalog import coerce_params, InfeasibleSkill
    params = coerce_params("pour", [{"name": "target_container", "value": '"clear cup b"'},
                                    {"name": "pour_angle", "value": ","},
                                    {"name": "hold_duration", "value": "2"}])
    check("a pour_angle of ',' is dropped", "pour_angle" not in params, f"{params}")
    check("a number written as a string is read as a number",
          params.get("hold_duration") == 2.0 and not isinstance(params["hold_duration"], str),
          f"{params}")
    try:
        coerce_params("stir", [{"name": "target_container", "value": '"cup"'},
                               {"name": "revolutions", "value": "two"}])
        dropped = True
    except InfeasibleSkill:
        dropped = False
    check("an optional integer that is not one is dropped, not fatal", dropped)

    # The real dry run of 2026-10-07: a note where a measurement belongs.
    params = coerce_params("dump", [
        {"name": "target_container", "value": '"clear cup b"'},
        {"name": "tool_offset", "value": '"[measured offset from pick_up of larger spoon]"'}])
    check("a tool_offset that is prose is dropped", "tool_offset" not in params, f"{params}")
    params = coerce_params("dump", [
        {"name": "target_container", "value": '"clear cup b"'},
        {"name": "tool_offset", "value": "[0.029, 0.009, 0.024]"}])
    check("a tool_offset of three numbers is kept as floats",
          params.get("tool_offset") == [0.029, 0.009, 0.024], f"{params}")
    params = coerce_params("dump", [
        {"name": "target_container", "value": '"clear cup b"'},
        {"name": "tool_offset", "value": "[0.029, 0.009]"}])
    check("a tool_offset that is not [x, y, z] is dropped", "tool_offset" not in params,
          f"{params}")


def test_a_pour_that_is_not_the_last_keeps_some():
    """A container that pours again before it is put down is not emptied."""
    print("\n[loop: pour some, then the rest]")
    subtasks = [subtask("pick up the plastic beaker"),
                subtask("pour some indicator into clear cup a", "clear cup a"),
                subtask("pour the rest of the indicator into clear cup b", "clear cup b"),
                subtask("place the plastic beaker back"),
                subtask("pick up clear cup a"),
                subtask("pour clear cup a into clear cup c", "clear cup c"),
                subtask("place clear cup a back")]
    calls = [("pick_up", {"object_name": "plastic beaker"}),
             ("pour", {"target_container": "clear cup a", "pour_angle": 90}),
             ("pour", {"target_container": "clear cup b", "pour_angle": 90}),
             ("place", {"target_location": "plastic beaker"}),
             ("pick_up", {"object_name": "clear cup a"}),
             ("pour", {"target_container": "clear cup c", "pour_angle": 90}),
             ("place", {"target_location": "clear cup a"})]
    responses = scoop_responses(
        subtask_plan=plan_response(subtasks),
        skill_call=lambda i, r: call_response(*calls[i % len(calls)]))
    orchestrator, llm, skills = build(responses)
    outcome = orchestrator.run("make the magic beaker")
    pours = [p for n, p in skills.calls if n == "pour"]
    check("the run succeeds", outcome.success, outcome.reason)
    check("the first pour from the beaker is capped at 60 so some stays",
          pours[0]["pour_angle"] == 60.0, f"{pours}")
    check("the beaker's last pour still empties it", pours[1]["pour_angle"] == 90,
          f"{pours}")
    check("cup a, poured once, is emptied", pours[2]["pour_angle"] == 90, f"{pours}")
    check("the record says what was changed and why",
          "must not empty" in outcome.record["steps"][1]["adjusted"],
          f"{outcome.record['steps'][1].get('adjusted')}")


def test_a_container_put_down_between_its_pours_still_keeps_some():
    """
    60 then 90 even when the beaker is put down for another tool in between.
    2026-10-15: the old rule stopped looking at the first put-down, and the
    beaker's first pour ran at the planner's 80.
    """
    print("\n[loop: pour some, put it down, pour the rest]")
    subtasks = [subtask("pick_up the plastic beaker"),
                subtask("pour from plastic beaker into clear cup a at 80 degrees", "clear cup a"),
                subtask("place the plastic beaker back to plastic beaker"),
                subtask("pick_up the stirrer"),
                subtask("stir clear cup a with the stirrer", "clear cup a"),
                subtask("place the stirrer back to stirrer"),
                subtask("pick_up the plastic beaker"),
                subtask("pour from plastic beaker into clear cup b at 80 degrees", "clear cup b"),
                subtask("place the plastic beaker back to plastic beaker")]
    calls = [("pick_up", {"object_name": "plastic beaker"}),
             ("pour", {"target_container": "clear cup a", "pour_angle": 80}),
             ("place", {"target_location": "plastic beaker"}),
             ("pick_up", {"object_name": "stirrer"}),
             ("stir", {"target_container": "clear cup a"}),
             ("place", {"target_location": "stirrer"}),
             ("pick_up", {"object_name": "plastic beaker"}),
             ("pour", {"target_container": "clear cup b", "pour_angle": 80}),
             ("place", {"target_location": "plastic beaker"})]
    responses = scoop_responses(
        subtask_plan=plan_response(subtasks),
        skill_call=lambda i, r: call_response(*calls[i % len(calls)]))
    orchestrator, llm, skills = build(responses)
    outcome = orchestrator.run("make the magic beaker")
    pours = [p for n, p in skills.calls if n == "pour"]
    check("the run succeeds", outcome.success, outcome.reason)
    check("the beaker's first pour is 60 although it is put down before the next",
          pours[0]["pour_angle"] == 60.0, f"{pours}")
    check("its second and last pour is 90", pours[1]["pour_angle"] == 90.0, f"{pours}")
    check("the record names the later pour",
          "sub-task 8" in outcome.record["steps"][1]["adjusted"],
          f"{outcome.record['steps'][1].get('adjusted')}")
    check("a put-down for another tool is kept", len(skills.calls) == 9,
          f"{[n for n, _ in skills.calls]}")


def test_the_first_of_two_pours_into_one_cup_keeps_some():
    """A, then B, into C: 65 then 90 (the user, 2026-10-15: first 60, then 65)."""
    print("\n[loop: two cups into one]")
    subtasks = [subtask("pick_up clear cup a"),
                subtask("pour from clear cup a into clear cup c at 80 degrees", "clear cup c"),
                subtask("place clear cup a back to clear cup a"),
                subtask("pick_up clear cup b"),
                subtask("pour from clear cup b into clear cup c at 80 degrees", "clear cup c"),
                subtask("place clear cup b back to clear cup b")]
    calls = [("pick_up", {"object_name": "clear cup a"}),
             ("pour", {"target_container": "clear cup c", "pour_angle": 80}),
             ("place", {"target_location": "clear cup a"}),
             ("pick_up", {"object_name": "clear cup b"}),
             ("pour", {"target_container": "clear cup c", "pour_angle": 80}),
             ("place", {"target_location": "clear cup b"})]
    responses = scoop_responses(
        subtask_plan=plan_response(subtasks),
        skill_call=lambda i, r: call_response(*calls[i % len(calls)]))
    orchestrator, llm, skills = build(responses)
    outcome = orchestrator.run("make the magic beaker")
    pours = [p["pour_angle"] for n, p in skills.calls if n == "pour"]
    check("the run succeeds", outcome.success, outcome.reason)
    check("A into C at 65, B into C at 90", pours == [65.0, 90.0], f"{pours}")
    check("the record says C is poured into again",
          "poured into again at sub-task 5" in outcome.record["steps"][1]["adjusted"],
          f"{outcome.record['steps'][1].get('adjusted')}")


def test_a_put_down_and_pick_up_of_the_same_container_is_dropped():
    """2026-10-15: place the beaker, pick it straight up again, between two pours."""
    print("\n[plan: no needless put-down]")
    subtasks = [subtask("pick_up the plastic beaker"),
                subtask("pour from plastic beaker into clear cup a at 80 degrees", "clear cup a"),
                subtask("place the plastic beaker back to plastic beaker"),
                subtask("pick_up the plastic beaker"),
                subtask("pour from plastic beaker into clear cup b at 90 degrees", "clear cup b"),
                subtask("place the plastic beaker back to plastic beaker")]
    calls = [("pick_up", {"object_name": "plastic beaker"}),
             ("pour", {"target_container": "clear cup a", "pour_angle": 80}),
             ("pour", {"target_container": "clear cup b", "pour_angle": 90}),
             ("place", {"target_location": "plastic beaker"})]
    responses = scoop_responses(
        subtask_plan=plan_response(subtasks),
        skill_call=lambda i, r: call_response(*calls[i % len(calls)]))
    orchestrator, llm, skills = build(responses)
    outcome = orchestrator.run("make the magic beaker")
    ran = [n for n, _ in skills.calls]
    pours = [p for n, p in skills.calls if n == "pour"]
    check("the run succeeds", outcome.success, outcome.reason)
    check("the beaker is held from one pour to the next",
          ran == ["pick_up", "pour", "pour", "place"], f"{ran}")
    check("the stored plan is the one that ran, numbered 1-4",
          [s["index"] for s in outcome.record["plans"][0]["subtasks"]] == [1, 2, 3, 4],
          f"{outcome.record['plans'][0]['subtasks']}")
    check("the record says what was dropped",
          "dropped sub-tasks 3-4" in " ".join(outcome.record.get("plan_edits") or []),
          f"{outcome.record.get('plan_edits')}")
    check("still 60 then 90", [p["pour_angle"] for p in pours] == [60.0, 90],
          f"{pours}")

    from robochem.agents.planner import Plan, SubTask
    from robochem.orchestrator.agent_orchestrator import drop_needless_regrasps
    moved = Plan(goal="g", subtasks=[
        SubTask(1, "pick_up the plastic beaker"),
        SubTask(2, "place the plastic beaker at the centre of the bench"),
        SubTask(3, "pick_up the plastic beaker")])
    check("a put-down somewhere else is kept",
          drop_needless_regrasps(moved) == [] and len(moved.subtasks) == 3)


def test_the_bench_is_photographed_after_each_pour():
    """2026-10-15: the report found B empty and could not say it had turned blue."""
    print("\n[loop: photos after each pour]")
    subtasks = [subtask("pick_up the plastic beaker"),
                subtask("pour from plastic beaker into clear cup a", "clear cup a"),
                subtask("pour from plastic beaker into clear cup b", "clear cup b"),
                subtask("place the plastic beaker back to plastic beaker")]
    calls = [("pick_up", {"object_name": "plastic beaker"}),
             ("pour", {"target_container": "clear cup a", "pour_angle": 60}),
             ("pour", {"target_container": "clear cup b", "pour_angle": 90}),
             ("place", {"target_location": "plastic beaker"})]
    taken = []

    def snapshot(subtask, call):
        taken.append(subtask.index)
        if subtask.index == 3:
            raise RuntimeError("cam 4 did not stream")
        return [f"/tmp/s{subtask.index:02d}_{call.skill}_cam{c}.jpg" for c in (2, 3)]

    responses = scoop_responses(
        subtask_plan=plan_response(subtasks),
        skill_call=lambda i, r: call_response(*calls[i % len(calls)]))
    orchestrator, llm, skills = build(responses, snapshot=snapshot)
    outcome = orchestrator.run("make the magic beaker")
    steps = outcome.record["steps"]
    check("photographed after each pour and nothing else", taken == [2, 3], f"{taken}")
    check("the paths are in the pour's step record",
          steps[1]["photos"] == ["/tmp/s02_pour_cam2.jpg", "/tmp/s02_pour_cam3.jpg"]
          and steps[0]["photos"] == [], f"{[s['photos'] for s in steps]}")
    check("a failed snapshot does not stop the run",
          outcome.success and steps[2]["photos"] == [], outcome.reason)


def test_the_operator_can_change_a_parameter_at_the_step_prompt():
    """--step: "pour_angle=60" changes the call and is recorded."""
    print("\n[loop: --step parameter change]")
    from robochem.orchestrator.agent_orchestrator import parse_step_overrides

    check("key=value is read as JSON",
          parse_step_overrides("pour_angle=60 to_floor=false") ==
          {"pour_angle": 60, "to_floor": False})
    check("quoted text is kept whole",
          parse_step_overrides("target_container='clear cup b'") ==
          {"target_container": "clear cup b"})
    check("prose is not a change", parse_step_overrides("60 degrees") == {})

    def confirm(subtask, call):
        if call.skill == "dump":
            call.params.update(parse_step_overrides("dump_angle_deg=60"))
        return True

    orchestrator, llm, skills = build(scoop_responses(), confirm=confirm)
    outcome = orchestrator.run("scoop citric acid into the white paper cup")
    dumps = [p for n, p in skills.calls if n == "dump"]
    step = outcome.record["steps"][2]
    check("the changed value is what ran", dumps and dumps[0].get("dump_angle_deg") == 60,
          f"{dumps}")
    check("the record shows the value that ran", step["params"].get("dump_angle_deg") == 60,
          f"{step['params']}")
    check("and says the operator changed it", "operator" in (step.get("adjusted") or ""),
          f"{step.get('adjusted')}")


def test_dry_run_moves_nothing():
    print("\n[loop: dry run]")
    orchestrator, llm, skills = build(scoop_responses(), dry_run=True)
    outcome = orchestrator.run("scoop citric acid into the white paper cup")

    check("the dry run reports success", outcome.success, outcome.reason)
    check("the executor was never called", skills.calls == [], f"{skills.calls}")
    check("the model still planned every sub-task",
          len(llm.calls_for("skill_call")) == len(SCOOP_SUBTASKS))
    check("the record says nothing was executed", outcome.record["dry_run"] is True)


def test_step_mode_stops_where_the_operator_says():
    """--step: nothing runs without a yes, and a no stops without a replan."""
    print("\n[loop: one step at a time]")
    asked = []

    def confirm(subtask, call):
        asked.append(call.skill)
        return len(asked) < 2          # run the first sub-task, stop at the second

    orchestrator, llm, skills = build(scoop_responses(), confirm=confirm, max_replans=2)
    outcome = orchestrator.run("scoop citric acid into the white paper cup")
    check("the operator was asked before each skill ran", asked[:2] == [c[0] for c in
          skills.calls] + [asked[1]], f"asked {asked}, ran {[c[0] for c in skills.calls]}")
    check("only the confirmed sub-task ran", len(skills.calls) == 1, f"{skills.calls}")
    check("a stop is not a failure to replan around",
          not outcome.success and len(llm.calls_for("corrective_plan")) == 0,
          f"{outcome.outcome}: {outcome.reason}")
    check("the record says the operator stopped it", "operator" in outcome.reason,
          outcome.reason)


def test_a_run_resumes_from_its_record():
    """--resume: same goal and plan, earlier steps not redone, their state restored."""
    print("\n[loop: resume from a record]")
    first_skills = RecordingSkills(fail={"dump"})
    orch1, _, _ = build(scoop_responses(), skills=first_skills, max_replans=0)
    first = orch1.run("scoop citric acid into the white paper cup")
    record = json.loads(json.dumps(first.record, default=str))
    check("the first run stopped at the dump", not first.success
          and [st["skill"] for st in record["steps"]] == ["pick_up", "scoop", "dump"],
          f"{[st['skill'] for st in record['steps']]}")

    class Bookkeeping(RecordingSkills):
        def __init__(self):
            super().__init__()
            self.replayed = []

        def _remember_pick_site(self, skill, params, result):
            self.replayed.append(("site", skill))

        def _track_held(self, skill, params, result):
            self.replayed.append(("held", skill))

    skills = Bookkeeping()
    # Skill calls for sub-tasks 3 and 4, in that order.
    responses = scoop_responses(
        skill_call=lambda i, r: call_response(*SCOOP_CALLS[(i + 2) % len(SCOOP_CALLS)]))
    orch2, llm2, _ = build(responses, skills=skills, max_replans=0)
    out = orch2.run(resume=record)
    check("no new plan was made", len(llm2.calls_for("subtask_plan")) == 0)
    check("it starts at the failed step and runs the rest",
          [c[0] for c in skills.calls] == ["dump", "place"], f"{skills.calls}")
    check("the earlier picks and scoop were replayed, not executed",
          ("site", "pick_up") in skills.replayed and ("held", "scoop") in skills.replayed,
          f"{skills.replayed}")
    check("the record says where it resumed",
          out.record.get("resumed_from", {}).get("from_step") == 3, f"{out.record.get('resumed_from')}")
    check("and it finished", out.success, out.reason)

    skills3 = Bookkeeping()
    orch3, _, _ = build(scoop_responses(
        skill_call=lambda i, r: call_response(*SCOOP_CALLS[(i + 3) % len(SCOOP_CALLS)])),
        skills=skills3, max_replans=0)
    orch3.run(resume=record, from_step=4)
    check("--from-step picks the starting sub-task", [c[0] for c in skills3.calls] == ["place"],
          f"{skills3.calls}")

    # A resume of a resume still knows the steps of the first run.
    second = json.loads(json.dumps(out.record, default=str))
    check("a resumed run's record carries the steps before it",
          [st["skill"] for st in second.get("resumed_steps", [])] == ["pick_up", "scoop"],
          f"{[st['skill'] for st in second.get('resumed_steps', [])]}")
    skills4 = Bookkeeping()
    orch4, _, _ = build(scoop_responses(
        skill_call=lambda i, r: call_response(*SCOOP_CALLS[(i + 3) % len(SCOOP_CALLS)])),
        skills=skills4, max_replans=0)
    orch4.run(resume=second, from_step=4)
    check("and resuming it replays all three earlier steps",
          [k for k in skills4.replayed if k[0] == "held"] ==
          [("held", "pick_up"), ("held", "scoop"), ("held", "dump")], f"{skills4.replayed}")


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


class _ToolFn:
    def __init__(self, name, args): self.name, self.arguments = name, json.dumps(args)


class _ToolCall:
    def __init__(self, i, name, args):
        self.id, self.type, self.function = f"call_{i}", "function", _ToolFn(name, args)


class _ToolMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content, self.tool_calls = content, tool_calls


class ToolFakeLLM:
    """
    A model that uses tools: each scripted turn is either
    ``{"tool_calls": [(name, args), ...]}`` or ``{"final": {...}}``.
    """

    def __init__(self, script):
        self.script, self.requests, self.chat = list(script), [], _Chat(self)

    def _respond(self, request):
        # A snapshot: the caller keeps appending to the same message list.
        self.requests.append(dict(request, messages=list(request["messages"])))
        turn = self.script.pop(0)
        if "tool_calls" in turn:
            calls = [_ToolCall(i, n, a) for i, (n, a) in enumerate(turn["tool_calls"])]
            return _Completion([_Choice(_ToolMessage(None, calls))])
        return _Completion([_Choice(_ToolMessage(json.dumps(turn["final"]), None))])


class FakeBenchCamera:
    """The report writer's cameras: overview photos and close-ups of one object."""
    color_order = "rgb"
    cameras = [2, 3, 4, 5]

    def __init__(self):
        self.looked, self.photographed = [], 0

    def names(self):
        return ["clear cup a", "clear cup b"]

    def photos(self, cameras=None):
        self.photographed += 1
        return [(c, np.full((48, 64, 3), 100, dtype=np.uint8)) for c in (cameras or self.cameras)]

    def close_up(self, name, views=2):
        self.looked.append(name)
        return (f"'{name}' located at x 0.425 m, y -0.159 m; close-ups from cameras [3, 4].",
                [(3, np.full((32, 32, 3), 200, dtype=np.uint8)),
                 (4, np.full((32, 32, 3), 210, dtype=np.uint8))])


def _image_count(content):
    return sum(1 for b in content if isinstance(b, dict) and b.get("type") == "image_url") \
        if isinstance(content, list) else 0


def test_the_report_covers_every_session():
    """A failed run and its resume make one report; the writer looks at what it needs."""
    import tempfile
    from robochem.agents.reporter import (load_run_chain, write_report, measurements,
                                          latest_record, find_console_logs)
    print("\n[report]")
    tmp = Path(tempfile.mkdtemp())
    first = {
        "started": "2026-10-15 17:24:34", "goal": "Make the magic beaker",
        "instruction": {"goal": "magic beaker", "steps": ["acid cup", "base cup"],
                        "expected_outcome": "red in the acid cup, blue in the base cup"},
        "scene": {"objects": [{"name": "clear cup a", "contents": "10 ml water"}]},
        "plans": [{"subtasks": [{"index": 1, "description": "pick_up the larger spoon"}],
                   "expected_observation": "cup a turns red"}],
        "outcome": "failed", "reason": "sub-task 6 failed", "seconds": 400.0,
        "steps": [
            {"subtask_index": 1, "subtask": "pick_up the larger spoon", "skill": "pick_up",
             "params": {"object_name": "larger spoon"}, "success": True, "reason": "",
             "seconds": 59.3, "result": {"gripper_width": 0.0083,
                                         "grasp_pose": [[1, 0, 0, 0.7]] * 4}},
            {"subtask_index": 6, "subtask": "stir clear cup a", "skill": "stir",
             "params": {"target_container": "clear cup a"}, "success": False,
             "reason": "landing on the rim", "seconds": 20.0, "result": {}},
        ],
    }
    second = {
        "started": "2026-10-15 17:39:18", "goal": "Make the magic beaker",
        "resumed_from": {"started": first["started"], "log_path": None, "from_step": 6},
        "plans": first["plans"], "outcome": "success", "reason": "all done", "seconds": 300.0,
        "steps": [{"subtask_index": 6, "subtask": "stir clear cup a", "skill": "stir",
                   "params": {"target_container": "clear cup a"}, "success": True,
                   "reason": "", "seconds": 90.0,
                   "result": {"contact_force_n": -1.8876, "stopped_by_force": True}}],
    }
    (tmp / "agent_run_1.json").write_text(json.dumps(first))
    (tmp / "agent_run_2.json").write_text(json.dumps(second))
    # A later dry run (plans only) is not the experiment to report on.
    (tmp / "agent_run_3.json").write_text(json.dumps(
        {"started": "2026-10-15 18:00:00", "dry_run": True, "steps": [{"subtask_index": 1}]}))
    newest = latest_record(tmp)
    check("the newest record that moved the robot is found, dry runs skipped",
          newest is not None and newest.name == "agent_run_2.json", str(newest))
    records, paths = load_run_chain(newest)
    check("the resume is chained back to the session it resumed",
          [r["started"] for r in records] == [first["started"], second["started"]], paths)
    # The console as tee kept it, with --step's prompt line.
    (tmp / "pipeline.log").write_text("\n".join([
        "[step 6] stir clear cup a",
        "   [step 6] Enter to run it, q to stop here: [Skills] stir 'clear cup a': bench-validated",
        "[ObjectLocalizer] opening cam 2...",
        "[Stir] Lowering to z=0.1271 (rod tip 10mm above the rim)...",
        "[Stir] Contact at z=0.1071: +1.00N from baseline, rod tip 10mm below the rim",
        "   FAILED: landing on the rim",
        "[failed] exhausted 0 corrective attempts",
    ]))
    logs = find_console_logs(tmp, records)

    final = {
        "title": "Magic beaker, first attempt", "summary": "The robot made the acid cup.",
        "objective": "Show pH colours.", "materials": ["citric acid", "clear cup a"],
        "procedure": "Sub-tasks 1-6 ran; 6 was resumed.",
        "observations": [{"subject": "clear cup a", "observed": "red liquid", "source": "photos"}],
        "result_vs_expected": "Cup a turned red as expected.",
        "issues": [{"subtask": 6, "what_happened": "stir refused", "likely_cause": "landing on the rim"}],
        "conclusion": "Partly done.", "next_steps": ["Run the base cup."]}
    # The writer looks at cup a, then takes fresh views, then writes.
    llm = ToolFakeLLM([{"tool_calls": [("look_at", {"object_name": "clear cup a"})]},
                       {"tool_calls": [("photograph_bench", {"cameras": [3]})]},
                       {"final": final}])
    camera = FakeBenchCamera()
    md = write_report(records, camera=camera, record_paths=paths, out_dir=tmp,
                      model="fake-model", client=llm, console_logs=logs)
    text = md.read_text()
    first_request = llm.requests[0]
    check("the writer starts from four overview photos and is offered the camera tools",
          _image_count(first_request["messages"][1]["content"]) == 4
          and {t["function"]["name"] for t in first_request["tools"]}
          == {"look_at", "photograph_bench"})
    check("it is told to write in English", "in English" in first_request["messages"][0]["content"])
    check("look_at photographs the object it asked for",
          camera.looked == ["clear cup a"] and camera.photographed == 2)
    second_request = llm.requests[1]["messages"]
    check("the result goes back as a tool message, the close-ups as a following image message",
          any(m.get("role") == "tool" and "located at" in m.get("content", "")
              for m in second_request)
          and _image_count(second_request[-1]["content"]) == 2)
    check("the close-ups are saved and shown in the report",
          (md.parent / "closeup_01_clear_cup_a_cam3.jpg").is_file()
          and "![clear cup a, camera 3](closeup_01_clear_cup_a_cam3.jpg)" in text
          and "photo_01_cam3.jpg" in text)
    check("what it looked at is listed",
          "what the writer looked at" in text and "look_at(object_name=clear cup a)" in text)
    check("the failure reaches the model", "landing on the rim" in json.dumps(first_request))
    check("every executed step is in the table, the failed one marked",
          text.count("| stir |") == 2 and "**FAILED**: landing on the rim" in text, text[:600])
    check("the model's prose is in the report",
          "The robot made the acid cup." in text and "red liquid" in text)
    check("the failed step's own console lines reach the model and the appendix",
          "Contact at z=0.1071: +1.00N" in json.dumps(first_request)
          and "console lines of the failed steps" in text
          and "opening cam 2" not in text.split("console lines of the failed steps")[1])
    check("measured values reach the appendix, poses do not",
          "contact_force_n = -1.8876" in text and "grasp_pose" not in text)
    check("measurements keeps scalars and drops matrices",
          measurements({"a": 1.23456, "grasp_pose": [[1]], "s": "x", "v": [1, 2, 3]})
          == {"a": 1.2346, "s": "x", "v": [1.0, 2.0, 3.0]})

    # The photos the run took after a pour reach the model before today's views,
    # and are copied into the report.
    import cv2
    shots = tmp / "photos_20261015_173918"
    shots.mkdir()
    for cam in (2, 3):
        cv2.imwrite(str(shots / f"s06_pour_cam{cam}.jpg"), np.full((24, 32, 3), 50, np.uint8))
    poured = dict(second, steps=[dict(second["steps"][0], skill="pour",
                                      subtask="pour from plastic beaker into clear cup b",
                                      photos=[str(shots / "s06_pour_cam2.jpg"),
                                              str(shots / "s06_pour_cam3.jpg"),
                                              str(tmp / "gone" / "s06_pour_cam4.jpg")])])
    llm3 = ToolFakeLLM([{"final": final}])
    md3 = write_report([first, poured], camera=FakeBenchCamera(), out_dir=tmp,
                       model="fake-model", client=llm3)
    content3 = llm3.requests[0]["messages"][1]["content"]
    texts3 = [b.get("text", "") for b in content3 if b.get("type") == "text"]
    check("the run's photos go to the model with the step they follow, then today's",
          _image_count(content3) == 2 + 4
          and any("after sub-task 6" in t and "clear cup b" in t for t in texts3)
          and texts3.index(next(t for t in texts3 if t.startswith("RUN PHOTOS")))
          < texts3.index(next(t for t in texts3 if t.startswith("PHOTOS:"))), texts3)
    check("a photo no longer on disk is skipped, the rest copied into the report",
          (md3.parent / "run_s06_s06_pour_cam2.jpg").is_file()
          and "run_s06_s06_pour_cam3.jpg" in md3.read_text())
    check("the model is told what the run photos are for",
          "RUN PHOTOS show each container as it was" in llm3.requests[0]["messages"][0]["content"])

    # A writer that keeps asking is cut off and made to answer.
    llm2 = ToolFakeLLM([{"tool_calls": [("look_at", {"object_name": "clear cup b"})]}] * 2
                       + [{"final": final}])
    write_report(records, camera=FakeBenchCamera(), out_dir=tmp, model="fake-model",
                 client=llm2, max_tool_rounds=2)
    check("after the last tool round the tools are withdrawn",
          "tools" in llm2.requests[1] and "tools" not in llm2.requests[2])

    # No cameras: one structured call, no tools.
    llm3 = FakeLLM({"experiment_report": final})
    md3 = write_report(records[1:], camera=None, out_dir=tmp, model="fake-model", client=llm3,
                       notes=["step 6 was a false contact reading"])
    check("with no cameras the report says so and no photo is sent",
          "No photos were taken" in md3.read_text()
          and llm3.calls_for("experiment_report")[0]["images"] == 0)
    check("operator notes reach the model and are shown as given",
          "false contact reading" in llm3.calls_for("experiment_report")[0]["user"]
          and "## Operator notes" in md3.read_text())


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
    test_scene_agent_on_the_real_bench()
    test_skill_agent_is_told_the_real_gripper_state()
    test_infeasible_is_reported_not_approximated()
    test_happy_path()
    test_a_pour_that_is_not_the_last_keeps_some()
    test_a_container_put_down_between_its_pours_still_keeps_some()
    test_a_put_down_and_pick_up_of_the_same_container_is_dropped()
    test_the_first_of_two_pours_into_one_cup_keeps_some()
    test_the_operator_can_change_a_parameter_at_the_step_prompt()
    test_the_bench_is_photographed_after_each_pour()
    test_a_malformed_value_is_dropped_not_passed_on()
    test_dry_run_moves_nothing()
    test_step_mode_stops_where_the_operator_says()
    test_a_run_resumes_from_its_record()
    test_a_failed_skill_replans_rather_than_retrying()
    test_an_infeasible_plan_stops_without_moving()
    test_a_missing_key_is_blocked_not_failed()
    test_the_measured_tool_offset_is_carried_forward()
    test_a_dropped_tool_shows_up_as_a_disagreement()
    test_the_record_is_serialisable()
    test_the_report_covers_every_session()

    print("\n" + "=" * 60)
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    for failure in FAIL:
        print(f"  FAILED: {failure}")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
