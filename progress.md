# Progress — robo-chem (Franka + RealSense)

Last updated: 2026-09-24

Live manipulation stack for the Franka Panda + RealSense cage. Related but
separate from the PLATO/agent notes in [`robomail/docs/progress.md`](robomail/docs/progress.md).

---

## Working commands (validated 2026-09-08 / 2026-09-09)

Prereqs: franka-interface/ROS up, grounding service on SAM 3, then:

```bash
cd /home/rumi/Desktop/robo-chem
source scripts/env.sh
```

Grounding (separate terminal / env):

```bash
perception_env/bin/python perception_service/grounding_service.py \
  --backend sam3 --model weights/sam3.pt
```

### Pick beaker (upright top-down grasp)

```bash
python scripts/run_experiment.py --skill pick_up \
  --params '{"object_name":"plastic beaker","z_offset":0.02,"squeeze":0.013}'
```

- Opens gripper, **`reset_joints`** (joint-space home) before multi-cam scan
- Upright **top** grasp (no 25° pour tip / spout align)
- Width close = measured diameter − squeeze (`grasp=False`)
  — soft cups still need `"force_limited": false`; default is now close-on-contact

### Pick larger spoon (flat object — lower workspace floor)

```bash
python scripts/run_experiment.py --skill pick_up \
  --workspace-min 0.25 -0.40 -0.13 \
  --params '{"object_name":"larger spoon","z_offset":0.0,"grasp_force":1.0}'
```

- Spoon sits ~z −0.01; default workspace floor (0.015 m) clamps the grasp too high
- `--workspace-min … -0.13` lowers the Z floor so fingers can reach the table
- Force-limited close at 1 N (`force_limited` default True); omit large positive `z_offset`

### Scoop from a labelled cup (reagent name → cup instance)

White paper cups sit on handwritten paper squares (`BAKING SODA`, `CITRIC ACID`,
`RED CABBAGE POWDER`, `Water`). SAM alone cannot tell them apart; locate now:

1. Segments **all** `white paper cup` instances (`return_all` on the grounding service)
2. Asks GPT-4o which label is under each numbered cup
3. Fuses the matching cup in 3D for scoop / pour / place

**Restart the grounding service** after pulling this change (needs `return_all`).

Offline check on stills (no robot):

```bash
python scripts/check_labeled_cups.py \
  --images-dir "scene_captures/spoon graspped_20260909_155259" \
  --label "citric acid" \
  --out-dir diag_out/labeled_cups
```

Scoop (hold the spoon first), using the reagent label as `powder_source`:

```bash
python scripts/run_experiment.py --skill scoop \
  --workspace-min 0.25 -0.40 -0.13 \
  --params '{"powder_source":"citric acid","tool_length":0.08}'
```

- Measure / tune `tool_length` for the held spoon before trusting dig depth
- Direct SAM category names (`plastic beaker`, `larger spoon`, `white paper cup`)
  still skip the label path

### Pour into white paper cup

```bash
python scripts/run_experiment.py --skill pour \
  --params '{"target_container":"white paper cup","pour_angle":90,"hold_duration":2,"forward_offset":-0.08}'
```

- **`reset_joints`** before scanning the cup
- Always tips **toward robot base (−X)** to 90° (no away-from-base fallback)
- Each 10° tip step advances EE **+X by 0.25 cm** (`tip_advance_m`) so the
  mouth does not drift toward the base as it tips
- `forward_offset=-0.08` sets the initial pour site (validated)
- After hold: **`reset_joints`** (joint-space home)
- Stuck tip steps are **retried** (longer duration) before failing

---

## What works

| Area | Status |
| --- | --- |
| Env bring-up (`scripts/env.sh`, frankapy Py3.8 + ROS Noetic) | Working |
| SAM 3 grounding service (`perception_service/`, `weights/sam3.pt`) | Working; prefer over GroundingDINO |
| Sequential RealSense capture (cams 2–5, one at a time) | Working — avoid streaming all four at once (USB wedge) |
| Live factory intrinsics + board-free cube extrinsic calib | Working (~3–6 mm held-out) |
| `pick_up` upright top grasp + reset_joints pre-scan | Working (beaker + spoon params above) |
| `pour` tip toward/away base + stall detection + site offsets | Working with `forward_offset=-0.08` |
| Smoke scripts | `scripts/smoke_test_skills.py`, `scripts/smoke_test_motion.py` |

---

## Calibration / perception notes

- Corrupt `.intr` files (fy≈fx/2) were a major early failure mode; pipeline now
  prefers **live RealSense factory intrinsics**
- Extrinsics: HSV green cube fiducial + Kabsch; install under robomail
  `calib/` with dated backup
- Scripts: `calibrate_cameras.py`, `validate_calibration.py`, `check_object.py`
- Prompt `"plastic beaker"` detects the red translucent beaker more reliably
  than bare `"beaker"`

---

## LLM agent pipeline (added 2026-09-24)

robomail_Aliyah's multi-agent planner now drives the real skills, in
[`robochem/agents/`](robochem/agents/) with the loop in
[`robochem/orchestrator/agent_orchestrator.py`](robochem/orchestrator/agent_orchestrator.py).
What was merged and what was dropped:

| From robomail_Aliyah | Fate |
| --- | --- |
| `agents/llm_client.py` | Kept. The ChatGPT call, structured output, key policy. Gained numpy-frame image blocks that respect BGR vs RGB |
| `agents/scene_understanding.py` | Kept, re-pointed at live cage frames and at names perception can resolve |
| `agents/high_level_planner.py` | Kept. `verification_modality` dropped — `ChemistryVerifier` routes itself off the wording |
| `config/robot_profile.py` | Kept. PLATO's taught `workspace_positions` dropped; this stack locates by name |
| `agents/action_vocabulary.py` | **Replaced** by `agents/skill_catalog.py`, generated from `SKILL_REGISTRY` |
| `agents/affordance.py` + `agents/step_planner.py` | **Replaced** by `agents/skill_planner.py`: sub-task → one skill call with parameters, no GOTO/GRASP/TILT |
| `skills/executor.py` (fake) | **Replaced** by `SkillsExecutor`. The arm moves |

Decisions worth keeping:

- **The model chooses the skill and its arguments, not the trajectory.** Motion
  inside each skill stays hand-written and SAM 3 grounded. This breaks
  robomail_Aliyah's "no hand-scripted motion" criterion; every trial record says
  so under `provenance`, and so does `plato_bridge.executor_provenance()`.
- **Names are grounded, not captions.** `VisionSystem.known_object_names()`
  returns `[]` on hardware (open-vocabulary SAM 3) and the bench inventory in
  sim. Where there is an inventory the scene agent must name from it — without
  that the model plans against a "measuring scoop" and the sim bench's spoon is
  called "larger spoon", which resolves to nothing.
- **`tool_offset` comes from `pick_up`, not from the model.** `pick_up` already
  reported `suggested_tool_offset`; the orchestrator now carries it to the next
  step's prompt. Its z is a lower bound (the cage looks down, the bowl underside
  is occluded) and the prompt says so.
- **A failed skill replans; it is never retried unchanged.** The old
  `VLMOrchestrator` re-ran a failed step up to three times identically.
- Verification is off by default in `--sim`: MuJoCo has no chemistry, so a
  colour check there fails for reasons unrelated to the plan and burns replans.

Two overlapping merge paths now exist, deliberately:
`plato_bridge.RoboChemExecutor` keeps robomail_Aliyah as the driver and swaps
only the executor (so its six-action vocabulary and log schema survive);
`AgentOrchestrator` moves the pipeline here and plans in the real skill names,
which is what makes `dump`, `arc_scoop`, `wait` and the perception skills
reachable at all.

### First end-to-end result (sim, 2026-09-24)

`"Scoop citric acid into the white paper cup"`, granules on, four sub-tasks, no
replan needed: pick_up spoon → scoop → dump → place beside the citric acid cup.
All four executed. **2 of 90 grains reached the paper cup.**

The thin yield is the known `measure_tool_offset` limitation, not the planner.
`pick_up` measured the bowl at **z = 3.9 mm** below the grasp; CAD says
**28.0 mm** (`bowl_offset.z − bowl_size.z` in `sim/bench.py`). The cage looks
down, the bowl's underside is occluded, and the cloud stops at its rim — which
`measure_tool_offset`'s own docstring calls a lower bound. `scoop` therefore
aimed the bowl at z=0.008 with the cup floor at z=0.003, and the real bowl sat
~24 mm lower than that, i.e. into the base rather than through the bed.

`scripts/film_sim_skill.py` already works around this by feeding CAD ground
truth. The durable fix belongs in perception or in a per-tool CAD table the
orchestrator can quote alongside the measurement — **not** in the prompt: the
model must not be asked to guess a depth no camera measured. Until then the
agent run surfaces the measured value with its lower-bound caveat, which is why
the model passed 3.9 mm rather than inventing something.

```bash
# no key, no robot: vocabulary, parameter checks, gripper bookkeeping, replanning
perception_env/bin/python scripts/test_agents_offline.py

# real GPT calls, real perception, nothing moves
perception_env/bin/python scripts/run_experiment.py --sim --no-viewer --dry-run \
  --workspace-min 0.25 -0.40 -0.13 \
  --task "Scoop citric acid into the white paper cup"
```

---

## Skills inventory (registered)

Manipulation: `pick_up`, `place`, `pour`, `scoop`, `dispense`, `stir`,
`move_to`, gripper open/close, `tilt`, `wait`  
Perception: `analyze_scene`, `locate_object`, `check_container`  

Entry: `scripts/run_experiment.py --skill … --params '{…}'`  

`place`, `scoop`, `stir` and `dispense` — the four the agent stack in
`robomail_Aliyah/` still needs — were rewritten to the `pick_up` / `pour`
standard on 2026-09-08 and pass 57 offline checks, but **none has run on the
robot yet**. What they expect, how they were built, and the bench-tuning order
are in [`SKILLS_ROADMAP.md`](SKILLS_ROADMAP.md).

`move_to`, `tilt`, `wait` and the perception skills are still unhardened; the
agent vocabulary never calls them.

```bash
perception_env/bin/python scripts/test_skills_offline.py   # no robot needed
```

---

## Design choices locked in recently

1. **Pick = upright top grasp** — angled/spout-aligned side grasp removed as
   default (still available via `"grasp_type":"side"` if needed).
2. **Pour = tip toward base (−X) only**; cartesian controller
   (`use_impedance=False`); retry lagging tip steps before failing.
3. **Pre-scan clear = `reset_joints`** for both pick and pour (joint space).
4. **`max_object_width` is not hardcoded** — old 55 mm was a bandaid for one
   bloated SAM mask; optional only. Gripper max ~80 mm still enforced.
5. **Pour failure detection** — mid-trajectory tip stall / shortfall fails that
   direction and falls back; no false `success=True` at ~39° when commanding 90°.

---

## Known issues / next

- Toward-base tip can hit a physical/singularity limit mid-trajectory (~40°);
  auto fallback to away-from-base is intended; tune
  `away_from_base_forward_offset` the same way as toward-base
- Taught pour orientation file
  (`calibration_out/taught_pour_pose.json`) exists from earlier demos but
  current pour uses absolute tip toward ±X instead
- Soft paper cups vs rigid beaker need different squeeze; beaker params above
  are the current known-good set
- USB: keep cameras off the same controller as HID when possible; always
  sequential open/close

---

## Key paths

| Path | Role |
| --- | --- |
| `robochem/skills/pick_up.py` | Pick skill |
| `robochem/skills/pour.py` | Pour skill |
| `robochem/skills/{place,scoop,stir,dispense}.py` | Rewritten 2026-09-08, not yet bench-tested |
| `robochem/agents/` | LLM planner: scene, plan, skill call, vocabulary |
| `robochem/orchestrator/agent_orchestrator.py` | The closed loop over agents + skills |
| `robochem/integration/plato_bridge.py` | The other merge: runs `robomail_Aliyah` plans on the real cell |
| `scripts/test_skills_offline.py` | Offline skill + bridge checks |
| `scripts/test_agents_offline.py` | Offline agent-loop checks (no API key) |
| `SKILLS_ROADMAP.md` | Skill gap, hardening playbook, integration plan |
| `robochem/vision/` | Localizer, SAM client, grasp analyzer |
| `perception_service/grounding_service.py` | SAM 3 / GDINO HTTP service |
| `scripts/env.sh` | ROS + frankapy + PYTHONPATH |
| `scripts/run_experiment.py` | Skill / task entry |
| `calibration_out/` | Extrinsics, taught poses, reports |
| `diag_out/` | Debug overlays / fused clouds |
