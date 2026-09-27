# Progress — robo-chem (Franka + RealSense)

Last updated: 2026-09-25

Live manipulation stack for the Franka Panda + RealSense cage. Related but
separate from the PLATO/agent notes in [`robomail/docs/progress.md`](robomail/docs/progress.md).

---

## Working commands (validated 2026-09-08 / 2026-09-09)

Prereqs: franka-interface/ROS up, grounding service on SAM 3, then:

```bash
cd /home/rumi/Desktop/robo-chem
source scripts/env.sh
```

Grounding — **one terminal is enough**, background it:

```bash
scripts/grounding.sh start      # idempotent; restart | status | stop | log
```

It has to be its own *process* (perception_env is Python 3.10 for SAM 3 and
DINOv2; the skills run in the frankapy Python 3.8 venv) but not its own
*terminal*. `status` also warns when `exemplars/*.npz` are newer than the
running process.

Still available by hand if preferred:

```bash
perception_env/bin/python perception_service/grounding_service.py \
  --backend sam3 --model weights/sam3.pt
```

**The exemplar library is read once, at startup.** A service left running from
before an object was registered answers that name by *text* prompting, and the
failure looks exactly like the object not being on the bench. On 2026-09-25 a
service running since Sep 9 reported `'scoop' -> no masks` on three cameras
while `exemplars/scoop.npz` sat unread on disk; `curl -s $GROUNDING_URL/health`
distinguishes them — it lists the loaded names, and an old build has no
`exemplars` key at all. Run `scripts/grounding.sh restart` after every
`register_object.py`.

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
| Exemplar matching for the printed tools (`exemplars/`) | Working — 0.95-1.00 on all 4 cams for stirrer / holder / scoop |
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

## Custom objects: exemplar matching (added 2026-09-25)

Text prompting ran out on the printed tools. SAM 3 is open-vocabulary, not
unlimited: `scripts/sweep_prompts.py` spent twenty phrases on the scoop and the
best honest hit was `"scoop"` on 2/4 cameras (`diag_out/stirrer_prompts/` is the
same story for the stirrer), while the phrases that scored *well* —
`"white object"`, 0.95 on 4/4 — were matching the cups. The concept is simply
not in the model, so no wording fixes it.

These objects are now matched by **appearance**. SAM's automatic mask generator
segments everything in the frame (no prompt, class-agnostic), DINOv2 embeds each
candidate crop, and the best cosine match against a reference crop wins.
Reference embeddings live in `exemplars/*.npz` and are the custom vocabulary.

Dispatch is by name inside the service, so `segment("stirrer")` is unchanged for
every skill above it: registered names go to the exemplar backend, everything
else still goes to SAM 3.

### Registering an object

```bash
source scripts/env.sh
python scripts/capture_scene.py --label stirrer_white

perception_env/bin/python scripts/register_object.py --name stirrer \
    --images-dir scene_captures/stirrer_white_<ts>          # drag a box per camera
```

The service loads `exemplars/` at startup (`--no-exemplars` to skip,
`--exemplar-thresh` to tune). Verify with the existing sweep:

```bash
python scripts/sweep_prompts.py --images-dir scene_captures/<run> \
    --only stirrer --save-overlays diag_out/stirrer_exemplar
```

### Registered so far (2026-09-25)

| Object | Views | Aliases |
| --- | --- | --- |
| `stirrer` | 4 (seated in holder) | — |
| `stirrer holder` | 8 (4 occupied + 4 empty) | `holder`, `tool holder` |
| `scoop` | 4 | `white plastic tool` |

Measured on `scene_captures/stirrer_white_20260925_113812` and
`holder_empty_20260925_120924`: **0.95-1.00 on all four cameras**, margin
0.43-0.72 over the next-best registered object. Leave-one-camera-out (register
from three, query the fourth) gives 0.69-0.85 — the honest number for a viewpoint
with no reference. Cross-object reference similarity: stirrer/holder 0.52,
stirrer/scoop 0.45, holder/scoop 0.34.

### Four things that cost real debugging time

1. **Register what you want to GRASP.** `ObjectLocalizer.get_object_pose` puts
   the grasp at the *mean of the fused mask points*. A first pass registered
   "stirrer" as the white cube **plus** the black cylinder it sits in; it scored
   0.95+ and would have driven the gripper into the middle of the holder. The
   cube is the stirrer, the cylinder is its holder, and they are now two
   objects. A high score only means "this is the shape you showed me".

2. **Register every state.** A holder with the tool in it and an empty holder
   are 0.49-0.83 similar — not interchangeable. The holder carries both because
   it is looked for occupied (before a pick) and empty (to put the stirrer
   back). Use `register_object.py --append`.

3. **Multi-part objects.** A two-tone object is several objects to the mask
   generator: the stirrer's body, the cube's top and the cube's shaded front
   face come back separately, so a whole-tool reference matched none of them
   (0.63 on cam2, head cut off). Merging every adjacent pair blindly gave a lid
   floating above a mug (0.64). What works is `_grow()` — add the neighbour that
   improves similarity to the reference most, repeat, stop when nothing does.
   cam2 0.63 -> 0.95, cam3 0.74 -> 0.99.

4. **Point-grid recall on small objects.** At 32x32 over 848x480 the grid lands
   ~26px apart, wider than the scoop's shank; on one camera that produced half a
   scoop, which genuinely resembles the stirrer cube more than a scoop (0.65 vs
   0.56) and was correctly refused. Raising density everywhere costs 2-3x on
   every frame *and* shifts which candidates seed `_grow` (cam2 scoop
   1.00 -> 0.89), so the service retries **only missed frames** at 48x48
   (`--exemplar-retry-points`). Growth adjacency also had to widen from a fixed
   8px — the two scoop fragments sat 9px apart — to 16px scaled by candidate
   size; similarity gates the merge, so the geometry test can be generous.

### Cost

~4s for the first query on a frame (mask generation), ~0.15s for every further
object on that same frame (candidates and embeddings are cached per image,
bit-packed). A missed frame that triggers the denser retry costs ~13s. A
per-camera miss is not a failed run: localization fuses whatever cameras did find
the object.

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

### Re-check with a scripted planner (sim, 2026-09-25)

No API key on the workstation, so the real `AgentOrchestrator` ran on the real
MuJoCo cell with `FakeLLM` from `scripts/test_agents_offline.py` standing in for
the model. Everything else was live: SimVision frames and inventory, the skills,
the arm. The scripted skill-call agent copied the measured `tool_offset` out of
the prompt, the way a model would. pick_up → scoop → dump → place (back where
the spoon was picked) all ran and reported success, with no replan.

Reported success is not the same as clean, and this is the gap a real-model run
will hit:

- **scoop rammed the citric acid dish**: 88k contact steps between the bowl
  and the dish, the fingers touched it too, and the dish moved **22 mm**. With
  nothing passing `container_center` / `container_radius`, the stroke is sized
  from `locate_container`: centre 17 mm off, opening radius read as **66 mm**
  for a 48.3 mm inside. So it planned 116 mm of travel in a dish that fits
  about 57 mm. `film_sim_skill.py` and `smoke_test_sim.py` pass the truth, which
  is why they are clean. The open fix is to make `locate_container` fit a
  circle to the rim band, instead of taking the median plus the 90th-percentile
  radius. **Adopted the same day, see below.** RANSAC plus an algebraic (Kåsa) circle
  fit on only the top 5 mm of the cloud gave the results below. The 15 mm band
  would include the powder surface, 9 mm under the rim. The fit lands on the
  wall's mid-line (50.3 outside, 48.3 inside), so the inside radius is the fit
  minus a wall. This is sim depth with 2 mm noise and exact masks; the rim on
  real SAM masks will be rougher.

  | Container | Median + p90: centre off / radius | Circle fit: centre off / radius |
  | --- | --- | --- |
  | citric acid | 17.1 / 65.5 mm | 0.6 / 50.3 mm |
  | baking soda | 33.6 / 73.3 mm | 1.0 / 50.2 mm |
  | paper cup | 1.7 / 40.9 mm | 0.1 / 38.2 mm |

  The live VLM run (`agent_run_20260925_163429`) hit the dish in exactly this
  way. The scoop took `dish_radius` 65.9 mm from the "measured opening", planned
  ±58 mm of travel, and had `bowl_length`/`bowl_width`/`bowl_depth` at 0 because
  the model has no way to know them, so the bowl was treated as a point.
- **dump was fine from perception alone**. It found the paper cup centre within
  1.7 mm. By the geometric estimate 1.31 of 1.83 ml went into the cup. Reach at
  0.54 m capped the straight-ahead tip at 68°.
- **measured tool offset**: [0.031, 0.008, 0.027] against CAD [0.0257, 0, 0.0294].
- **a plan that names `"table"` as the place target fails** with `Cannot locate
  target 'table'`. `place` wants something on the inventory, for example the
  tool's own name to put it back where it was picked.
- **real-model dry run** (gpt-4.1, same day, `--sim --dry-run`): it planned
  the same four steps and all four resolved to valid calls. It passed no
  `tool_offset`, because a dry run never executes pick_up and so there is no
  measurement to carry. For "beside the citric acid cup" it chose `place`
  target `"citric acid cup"`; watch where that puts the spoon in a live run.

### Full chain without the planner (sim, 2026-09-25)

This runs pick → scoop → dump → spoon back → pick stirrer → stir → stirrer back
in one session, with no model. Repeated `--skill`/`--params` run in order
against one cell, so the held tool carries over. Since the circle fit and the
tool geometry fill-in below, the scoop needs no ground truth except the floor
(see "Still open").

```bash
python scripts/run_experiment.py --sim --workspace-min 0.25 -0.40 -0.13 \
  --skill pick_up --params '{"object_name":"larger spoon","z_offset":0.0,"grasp_force":1.0}' \
  --skill scoop   --params '{"powder_source":"citric acid","container_floor_z":0.008}' \
  --skill dump    --params '{"target_container":"white paper cup"}' \
  --skill place   --params '{"target_location":"larger spoon"}' \
  --skill pick_up --params '{"object_name":"stirring rod"}' \
  --skill stir    --params '{"target_container":"white paper cup","tool_axis":"z","tool_length":0.081,"revolutions":2}' \
  --skill place   --params '{"target_location":"stirrer holder","on_top":true,"release_clearance":0.015,"stop_force_n":1.0}'
```

Measured headless (`--no-viewer --sim-fast`, with every arm and held-tool
contact counted per skill): all 7 steps succeed, and nothing on the bench moves
more than 3 mm. The only contacts are the grasps themselves, the spoon leaving
the table, and the stirrer's rod in its holder's bore. The dump tips to 68° (the
reach cap at 0.54 m).

**Using the plastic beaker as the dump + stir target instead is not clean.** The
beaker is closer, so the dump reaches 89°. But the stirrer holder stands 16 cm
straight ahead of the beaker at (0.62, 0.14). With the bowl pointing ahead,
link7 swept into the seated stirrer during the tip (12.8k contact steps).

### Powder dish, rim circle fit, tool geometry (2026-09-25)

**The dish, measured on the bench:** 91 mm inside, 102 mm outside, 29 mm tall,
8 mm base. It is now modelled exactly. `Prop.floor_thickness` is new;
`inner_radius`, `outer_radius` and `floor_height` are the properties everything
reads. The old model had 96.6 / 104.6 / 31 / 4 mm; its "outside" was the
panels' mid-line, not their outer face. `scoop`'s `floor_thickness` default
went 4 → 8 mm with it. 4 mm would now put the 3 mm floor gap 1 mm into the
floor.

**`locate_container` fits a circle to the rim** (`_fit_rim_circle`): RANSAC plus
Kåsa on the top 5 mm of the cloud. `rim_radius` is still the inside of the rim,
because scoop, arc_scoop and stir read it as the opening. It is taken as the
5th percentile of the rim points' distance from the fitted centre. The fitted
circle itself, the wall's mid-line, is `rim_mid_radius`. With too little rim to
fit, it falls back to the old median and says so (`rim_method`). Measured in sim:

| Container | Centre off | Inside radius: read / true |
| --- | --- | --- |
| citric acid dish | 1.0 mm (median: 17) | 44.6 / 45.5 mm |
| baking soda dish | 1.4 mm (median: 34) | 44.3 / 45.5 mm |
| paper cup | 0.2 mm | 35.4 / 36.5 mm |
| beaker | 0.8 mm | 31.9 / 33.0 mm |

All four read about 1 mm small, which is the safe side.

**Tool geometry is filled in by the executor** (`robochem/skills/tool_geometry.py`).
`SkillsExecutor` remembers what `pick_up` put in the jaws (`held`), and before
scoop, dump or arc_scoop it fills any missing `bowl_length` / `bowl_width` /
`bowl_depth` / `tool_span` / `tool_back_reach` from the tool's CAD. It fills
`tool_offset` too when none was passed. A passed one keeps its x/y but has its
z raised to the CAD 29.4 mm, since the measured z is a lower bound. With the
circle fit, this took the no-truth agent run from ramming the dish (88k
contacts, dish moved 22 mm) to **zero contacts**.

**Powder depth decides whether the floor-referenced scoop collects anything.**
The tip stays 3 mm off the floor and the mouth is about 10 mm above the tip, so
in this 21 mm-deep dish (sim, all truth passed): 12 mm of powder gave 0%,
14 mm 4%, 16 mm 41%, 18 mm 100%. The sim default is now 18 mm, surface 3 mm
under the rim. **The real fill level is not known yet.**

**The drawn bed has to LOOK like powder, or the planner will not scoop it.**
A flat `fill_rgba` bed in a white dish rendered as one white disc under the
cage's lights. The scene agent reported the citric acid dish "empty" on every
run, and in one live run the planner stopped with 0 sub-tasks: "The 'citric
acid cup' is empty". The bed now has a speckled texture, with the geom colour
at 0.55 of `fill_rgba`; at full white the upward face saturates and the speckle
vanishes. After that, two of two dry runs said "white powder" and planned all
four steps.

**Still open: the floor estimate.** With no truth passed, the scoop took the
inside floor as `base_z + floor_thickness` = 10.5 mm against a true 8.0 mm.
The cloud's lowest points are the dish's outer wall about 2.5 mm up, not its
bottom edge. Effect on the collected powder, no truth passed otherwise:

| tool_offset | floor | collected |
| --- | --- | --- |
| measured (z raised to CAD) | estimated 10.5 mm | 2% |
| CAD | estimated 10.5 mm | 3% |
| measured (z raised to CAD) | true 8.0 mm | 36% |
| CAD | true 8.0 mm | 64% |

The measured tool offset is 5 mm long in x and 8 mm off in y. At the 60° bite,
x error becomes about 4 mm of height. Candidates: reference the floor to the
table (z = 0 on this bench) instead of the cloud, and prefer the CAD offset.

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
- **USB: the cameras and the mouse are on the same controller, and it has now
  wedged the host three times.** Measured 2026-09-25 — this machine has three
  USB controllers and the cameras are sharing the busy one with the input
  devices:

  | Controller | Buses | On it |
  | --- | --- | --- |
  | `00:14.0` Intel Z370 xHCI | 1 + 2 | **all four cameras AND the mouse/keyboard** |
  | `01:00.2` NVIDIA TU102 | 3 + 4 | *nothing* |
  | `04:00.0` ASMedia ASM1142 | 5 + 6 | *nothing* |

  Buses 1 and 2 are the two root hubs of one physical controller (both
  `0000:00:14.0`), so "different bus" is not separation. The freeze signature in
  `dmesg` is the Realtek hub at `1-3.1` — the one the mouse and keyboard hang
  off — dropping with `error -71` (EPROTO) and taking its children with it:

  ```
  usb 1-3.1: Failed to suspend device, error -71
  usb 1-3.1: device descriptor read/all, error -71
  usb 1-3.1.1: USB disconnect, device number 20    <- HID
  usb 1-3.1.4: USB disconnect, device number 21    <- HID
  ```

  Sequential capture reduces this but does not remove it: the contention is for
  the controller, not just for bandwidth at one instant. **The fix is physical**
  — move the camera hub to the ASMedia (`04:00.0`) or NVIDIA (`01:00.2`) ports,
  both of which are completely unused. Until then, expect the mouse to die
  occasionally during a multi-camera scan, and prefer the NVIDIA controller only
  as a second choice since that GPU is also running SAM 3 and DINOv2.
- `stirrer` has only the 4 views of it seated in its holder. Once it is lying
  loose on the bench, capture that and
  `register_object.py --name stirrer --append` — a tool on its side is a
  different view, and it will be on its side right after every pick

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
| `perception_service/grounding_service.py` | SAM 3 / GDINO / exemplar HTTP service |
| `perception_service/exemplar_matcher.py` | Appearance matching for objects SAM has no word for |
| `scripts/register_object.py` | Teach an object by boxing it once |
| `exemplars/` | Registered objects + `README.md` on how to add one |
| `scripts/env.sh` | ROS + frankapy + PYTHONPATH |
| `scripts/run_experiment.py` | Skill / task entry |
| `calibration_out/` | Extrinsics, taught poses, reports |
| `diag_out/` | Debug overlays / fused clouds |
