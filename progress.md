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

It has to be its own *process* (perception_env is Python 3.10 for SAM 3; the
skills run in the frankapy Python 3.8 venv) but not its own *terminal*.

Still available by hand if preferred:

```bash
perception_env/bin/python perception_service/grounding_service.py \
  --backend sam3 --model weights/sam3.pt --no-exemplars
```

**DINOv2 is not loaded.** `scripts/grounding.sh` passes `--no-exemplars`.
The printed scoop is the SAM 3 phrase `"white plastic tool"` again
(`scoop`, `rectangular scoop`, `printed scoop`, and `measuring scoop` all
rewrite to it). Exemplar matching still exists and still works — measured
0.95-1.00 on 2026-09-25 — but it loads DINOv2 plus a second SAM on top of
SAM 3, and that combination exhausted this 15GB host. Turn it back on only
with an explicit `--exemplars exemplars` on the python command, not via
`grounding.sh`.

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
- **Camera 2's crop was the cup, not the paper** (2026-09-25,
  `scene_captures/b10_water_20260925_153459`). SAM boxed the clear cup at
  (390, 233)–(457, 302). The 0.9× pad ended at y=331. `B 10 ML WATER` is on
  the sheet in front of the cup, down to about y=450, so the reader saw a
  sliver of `10 ML` and returned null (`labels [none]`). A second mask of
  the same cup (score 0.32) was then greyed over that sliver. The crop now
  follows the white sheet downward and sideways, and does not pad upward
  into the reagent behind the cup — that upward pad made the reader say
  `A 10 ML WATER` and `B 5 ML NaCl` for a crop that also contained the B
  sheet. Re-read of those stills: cam 2 matches `B 10 ML WATER` at score
  0.79, cam 3 at 0.91, cam 5 at 0.89. Cam 4 still reads the B cup as
  `A 10 ML WATER` and is not used.
- **One camera that reads the label is enough** (2026-09-25 dump of
  `B 10 ml water`). Cam 3 read `B 10 ML WATER` at score 0.90, anchor
  `[0.4081, 0.0357, 0.0055]`. Cam 4 was a projection match (score 0.73) whose
  centroid was 52 mm away, so the 50 mm pairwise rule rejected both and the
  direct SAM prompt `"B 10 ml water"` found nothing. If exactly one camera
  read the text, that camera's cloud is used alone. Two cameras that both
  read the label and still disagree are still refused. The 50 mm tolerance
  was not loosened.

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
| SAM 3 grounding service (`perception_service/`, `weights/sam3.pt`) | Working; prefer over GroundingDINO. DINOv2 is off |
| Scoop via the text prompt `"white plastic tool"` | The live path again (see below). Partial on the 2026-09-18 sweep: cam5 the whole tool, cam2 the handle only |
| Exemplar matching (`exemplars/`, DINOv2 + SAM-vit-base) | Measured 0.95-1.00, **not loaded**. Opt-in with `--exemplars exemplars` |
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

## Custom objects: exemplar matching (added 2026-09-25, unloaded the same day)

Turned back off on 2026-09-25 because the live service was SAM 3 **plus**
DINOv2-base **plus** SAM-vit-base. The process that had all three loaded was
idle at **7.1GB resident, 9.2GB peak, 1.4GB swap**. After
`scripts/grounding.sh restart` with `--no-exemplars`, `/health` reports
`exemplars: []` and the idle process is **360MB, 0 swap**. The 3.3GB
weights are not on the GPU until the first `/segment`. Measured on the
14:40 pick of `"white plastic tool"`: camera 2 took **94.6s**, camera 3
took **0.35s**, and the robot terminal printed nothing during that wait,
which reads as a hang. After that query the service sits at 3.2GB resident
and 5.2GB of GPU, 0 swap. A four-camera `segment_instances` was already
5.85GB resident / 8.90GB peak before the extra models. The scoop is
`"white plastic tool"` again.

**The stirrer is `"white cube"` (found 2026-09-25, real robot).** All the
prompts in `diag_out/stirrer_prompts/` described the whole black assembly or
guessed at tool words (`"stirrer"`, `"stirring rod"`, `"black cap"`) and none
of them separated the grasp point from the holder cleanly enough to trust.
Describing only the part the gripper actually needs — the **white** cube
sitting on the black cylinder, not the assembly, not the cylinder — worked:
`pick_up --params '{"object_name":"white cube",...}'` grasped it and the chain
carried into `stir` (which requires `is_gripper_holding()`, so reaching that
step is itself the confirmation). Added `"cube"` and `"cylinder"` to
`label_resolver._DIRECT_SAM_KEYWORDS` so `"white cube"` skips the reagent-label
path and goes straight to a SAM 3 text prompt — without that it would first
scan every `"white bowl"` for a paper reading "white cube" and only fall back
after failing there.

The holder itself is still unconfirmed by this route — try `"black cylinder"`
next, now that it is also a direct keyword.

**`"white cube"` then failed on the robot (2026-09-25).** The error was
`Rejecting 'white cube': extent [0.294 0.39 0.13] exceeds 0.35m`. All four
cameras agreed on the centroid (max pair 34-41 mm), so SAM had found the
right object. Mask areas in `grounding_service.log` are 1,000-2,900 px, the
same as the earlier pick that worked. The mask was fine; the depth behind
it was not.

`_depth_to_points` back-projected every mask pixel's raw depth. Edge pixels
of a small object often carry the bench depth behind it, plus RealSense
flying pixels in between. Those become streaks along each camera ray. They
are dense enough to survive `_remove_outliers` (20 neighbours, 2σ) and hardly
move the mean, which is why the centroids still agreed. Replay of
`scene_captures/stirrer_white_20260925_113812`:
- cam 2's cube cloud: **17.7 × 9.6 × 11.9 cm**, depth tail to 0.66 m behind a
  cube at 0.47 m.
- cam 3's: **11 × 12.9 × 12 cm**.
- Fused after consensus: **19 × 6 × 12 cm** for a 30 mm cube, with z
  reaching down to the table. That frame happened to pass under 0.35 m; the
  live one did not. So the earlier success was luck, not a regression.

Fix: `ObjectLocalizer._depth_gate` keeps only pixels within the object's
apparent size of the mask's median depth. The size is the 2-98 percentile
pixel box, converted to metres at that median depth, with a 20 mm minimum. A
cup gets a band as deep as it is wide; a small object gets a tight one.
Same stills afterwards:
- cube: cam 2 **3.9 × 4.3 × 3.1 cm**, cam 3 **4.3 × 4.1 × 3.0 cm**, z
  0.071-0.103 (cube only).
- `"clear plastic cup"` (b10_water): 3 cameras identical, cam 3 dropped 80
  of 3,331 points, a tail that made it 2.5 cm too long in x.
- `"white bowl"` (b10_water and bowls_20260924): identical on every camera.

The replay used nominal 848×480 intrinsics (fx = fy = 610); the robomail
`.intr` files are 640-wide and unusable. Tried and rejected: **2 px mask
erosion.** It left cam 2 at 14.5 cm, because the streak pixels are not all
on the outer ring. The live failing frames were not saved, so this is
diagnosed from stills of the same cube, not from that exact run. **Not yet
re-run on the robot.** The replay script is small, so rebuild it rather than
look for it: grounding client + `_depth_to_points` on `cam{N}_depth.png`.
`test_depth_gate_drops_edge_streaks_not_containers` added;
`test_skills_offline.py` 198 pass (the same 2 scoop failures as before),
`test_agents_offline.py` 61 pass.

The measurements below are still the ones to trust if `--exemplars` comes back
on.

## Custom objects: what the exemplar path measured

Text prompting ran out on the printed tools. SAM 3 is open-vocabulary, not
unlimited: `scripts/sweep_prompts.py` spent twenty phrases on the scoop and the
best honest hit was `"scoop"` on 2/4 cameras (`diag_out/stirrer_prompts/` is the
same story for the stirrer), while the phrases that scored *well* —
`"white object"`, 0.95 on 4/4 — were matching the cups. The concept is simply
not in the model, so no wording fixes it.

When `--exemplars` is passed, these objects are matched by **appearance**.
SAM's automatic mask generator segments everything in the frame (no prompt,
class-agnostic), DINOv2 embeds each candidate crop, and the best cosine match
against a reference crop wins. Reference embeddings live in `exemplars/*.npz`.
That flag is not the default as of 2026-09-25; see the note above.

### Registering an object

```bash
source scripts/env.sh
python scripts/capture_scene.py --label stirrer_white

perception_env/bin/python scripts/register_object.py --name stirrer \
    --images-dir scene_captures/stirrer_white_<ts>          # drag a box per camera
```

The service does **not** load `exemplars/` unless started with
`--exemplars exemplars` (`--exemplar-thresh` to tune). Verify with the
existing sweep:

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

## Scoop -> dump -> place-back chain (validated 2026-09-25, text prompts only)

One `run_experiment.py` call, four `--skill`/`--params` pairs so the gripper
stays loaded across steps (a second invocation opens a fresh cell with an
empty gripper):

```bash
source scripts/env.sh
scripts/grounding.sh status   # confirm up, and exemplars: [] -- see below

python scripts/run_experiment.py --workspace-min 0.25 -0.40 -0.13 \
  --skill pick_up --params '{"object_name":"scoop","z_offset":0.0,"grasp_force":1.0,"forward_offset":-0.010}' \
  --skill scoop   --params '{"powder_source":"baking soda","tool_offset":[0.051,0.009,0.028],"tool_span":0.0275,"tool_back_reach":0.030,"bowl_length":0.0275,"bowl_width":0.0205,"bowl_depth":0.0115}' \
  --skill dump    --params '{"target_container":"b 10 ml water","container_category":"clear plastic cup","tool_offset":[0.051,0.009,0.028],"bowl_length":0.0275,"bowl_width":0.0205,"bowl_depth":0.0115}' \
  --skill place   --params '{"target_location":"scoop"}'
```

- **`object_name: "scoop"` resolves without exemplars.** `SAM_PROMPT_ALIASES`
  maps it to `"white plastic tool"`, the SAM 3 text prompt validated in the
  2026-09-18 sweep. No `--exemplars` flag needed; see the "unloaded" note above
  for why that flag is off by default now.
- **`tool_offset: [0.051, 0.009, 0.028]`** — x/y measured by `pick_up`'s
  `measure_tool_offset` off THIS grasp (it moves if `forward_offset` changes,
  see below); z is the CAD depth, not the measured one. The cameras look down
  and cannot see the bowl floor, so `measure_tool_offset` reports a lower bound
  there (has read as low as 0.011-0.019m depending on grasp) — always take z
  from the printed scoop's CAD (0.028m for a mid-grip grasp), never from the
  printed offset directly.
- **`forward_offset: -0.010` on the pick** moves the grasp off the crank (the
  one spot on the handle where the cross-section changes and the jaws cannot
  seat flat) without going so far it overshoots the flat grip, which is only
  26mm long.
- **`"target_location": "scoop"` on the final `place`** uses the executor's
  pick-site memory (`SkillsExecutor.pick_sites`, added 2026-09-25): a
  successful `pick_up` records where it grasped the object, and `place` resolves
  a matching name to that site instead of asking vision to find a tool that is
  in the gripper and cannot be on the bench for a camera to see. A name the
  executor has not picked passes through unchanged, so this never masks a real
  scene object.
- `container_category: "clear plastic cup"` on `dump` — required because the
  label-category default is now `"white bowl"` (where the reagents live);
  water needs the clear-cup category explicitly.

### The freeze this chain caused before the exemplar stack was turned off

Running this same chain with `--exemplars` on (DINOv2 + a second SAM resident)
froze the whole desktop, not just the grounding process, partway through step
2's `segment_instances` call. Measured cause, in order ruled out:

1. **Not the shared USB controller.** Real and reproducible in `dmesg` (a
   Realtek HID hub dropping with `error -71` under camera load), but a HID drop
   kills the mouse, not the whole desktop, and the timing did not match.
2. **Not per-instance label reads.** Reverting to one whole-frame VLM call per
   camera was tried and made cross-cup labelling WORSE:
   `scene_captures/ab_water_20260921_102436`, `"a 10 ml water"` / `"b 10 ml
   water"` went from 4/4 + 4/4 (per-instance) to 2/4 + 0/4 (whole-frame), with
   invented labels (`"WEAK BASE"`, `"GLITTER ADD"`) not on the bench. Reverted
   back; per-instance stays default (`LabelResolver.per_instance = True`).
3. **Not batching cameras into the /segment call.** Sending 4, 2, or 1 camera(s)
   per HTTP request all measured the identical **8.90GB peak** — the growth is
   a one-time CUDA/workspace allocation on first use, not a per-frame cost, so
   splitting frames across requests cannot lower it.
4. **The actual cause: host RAM.** This machine has 15GB. The grounding
   service with SAM 3 + DINOv2 + a second SAM resident sat at **5.85GB idle,
   8.90GB peak** after the first segmentation of any kind — a permanent step,
   not a spike that comes back down. With Slack and spare editor windows open,
   `available` was ~3GB; the peak exceeded it and the kernel started paging
   rather than OOM-killing anything, which is what a full desktop freeze looks
   like (no `oom-kill` in `dmesg`, 1.3GB already in swap before the run even
   started). `vm.swappiness=60` (the default) does not help here: the 64GB
   swap file was untouched at the time of the freeze, because the resident set
   was live model weights being paged back in on every inference, not idle
   memory swap could usefully evict.

Fix: `--no-exemplars` (now `scripts/grounding.sh`'s default). Confirmed
`/health` idle at **360MB, 0 swap**; loads the 3.3GB SAM 3 weights lazily on
first `/segment` and settles at **3.2GB resident, 5.2GB GPU, 0 swap** after.
The trade is the stirrer and its holder have no validated text prompt yet (see
above) — this chain does not need them, so it is unaffected.

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

## Dump shake (bench, 2026-09-25)

`"b 10 ml water"`, `tool_offset` all zeros, `rim_clearance` 10 mm then 20 mm.
Straight ahead the tip stops at **73°** (cup x≈0.45) then **77.5°** (rim
centre [0.357, 0.043]) of a commanded **90°**. `tip_capped_by_reach` was
false both times — the geometric reach model (`PANDA_REACH` 0.718 m) thought
90° was fine, and sim's MuJoCo IK tracks the planned arc, so sim rotates
further. Those are not the frankapy cartesian controller. With
`tool_offset` all zeros the "bowl" is the TCP, so the tip is a pure wrist
fold; sim's printed scoop has a real bowl offset and the TCP orbits. Neither
run proves 77° is a software cap. Re-commanding that 90° pose for 1.5 s, and
then streaming the shake (66°↔78°) and the return, were all accepted. The
wrist never left the stall: **77.5° → 77.6°** on all four half-swings, and
**77.6°** after the return. Anchoring those streams on the real bowl position
did not change that. The 90° re-command is what seats the joint on the stop;
it is no longer issued.

Third dump, same cup, cameras 2 and 3 agreed within 44 mm, rim centre
[0.449, 0.020], z=0.039. Commanded 90°, stopped at **78.8°**. That angle
empties the scoop; it is now `dump_angle_deg` **79**, and the shake must not
command anything past the angle actually held. The ±12° tool-Y rock at the
stall measured **78.8° → 78.8°** and the skill went to `reset_joints` still
holding the scoop. The shake is now the pour arc run backwards then forwards
again: 79° → 67° → 79°, poses the tip already passed through, not a new
orientation pressed into the stop. Not yet run on the robot. The return
still unwinds in ≤20° steps along that same arc.

A pose that is not actually reached sends the arm to `reset_joints` via
`go_home()`. The gripper stays closed (this is not `--reset`, which opens
the gripper at home). Triggers: carry command fails, bowl arrives short of
the cup, tip command fails, measured tip below `min_tip_deg` (60), shake
backoff moves <3°, return still more than `tip_tol` (10°) from level, or
the lift does not arrive. A tip that clears 60° still counts as dumped; the
home is the recovery when the shake or the unwind cannot leave the stall.
Not bench-tested.

## Stir: top-down by default (2026-09-25)

On the robot, `stir` behaved differently from the sim: it tipped the held
stirrer about 90° onto its side before stirring. The skill's default was
`tool_axis: "x"`, the flat-spoon path. That path rotates 90° about tool Y to
stand a handle up. The sim never took it, because `smoke_test_sim.py` passes
`"tool_axis": "z"`. Nobody passed that on hardware. The agent catalog does not
expose `tool_axis`, so every agent-planned stir on hardware got the tilt too.

Now the default **is** the sim behaviour. `tool_axis` defaults to `"z"`: the
stir is done fingers-down, which is the orientation `pick_up` grasps in. After
the home, the wrist is levelled to exactly straight down with
`tool_down_rotation()`, keeping its yaw so the stirrer does not spin in the
jaws. The hover, descent, circle and lift all command that ideal rotation, not
the measured one. The spoon tilt is still available, opt-in only, with
`"tool_axis": "x"`.

**`tool_length` now defaults to 0.081 on the `"z"` path.** That is the CAD
length: 15 mm from TCP to cube centre plus the 66 mm rod, taken from
`stirrer v1.stl` and the sim's `Prop.tool_length`. It is **not measured on
the real part yet.** Before this change the hardware default was 0.0, which
puts the rod tip 81 mm below the intended depth, i.e. into the cup floor.
**Do not pass pick_up's `suggested_tool_offset` z for the stirrer.** Sim
measured **0.012** against the 0.081 CAD, because the rod is inside the
holder during the pick and no camera sees it. The catalog's `tool_length`
text now says exactly that.

```bash
python scripts/run_experiment.py \
  --skill pick_up --params '{"object_name":"white cube"}' \
  --skill stir    --params '{"target_container":"<cup>","revolutions":2}'
```

Verified in sim and offline only; **not yet run on the robot**:
- `smoke_test_sim.py --only "pick + stir"`: all checks pass.
- A sim pick_up + stir with **no** `tool_axis`/`tool_length`: `tool_axis`
  `z`, `tool_length` 0.081, no tilt, 0.001° from down, 2/2 revolutions.
- `test_skills_offline.py`: 184 pass.
  - The 2 failures are the scoop workspace-floor check. They failed before
    this change too.
  - New test: `test_stir_defaults_to_top_down`.
  - The spoon tests now pass `"tool_axis": "x"`.

Check first on the bench:
- whether 0.081 puts the rod tip where it should be; `stir_depth` is
  measured from the tip
- whether the `"white cube"` grasp centre sits lower on the cube than the
  cube centre, since the cameras mostly see its top face; if it does, the
  real `tool_length` is larger than 0.081

### Full chain: scoop → dump → place scoop → stir → stirrer back (2026-09-25)

These are the seven steps in one `run_experiment.py` call. The pick-site
memory spans the whole chain, so `"scoop"` and `"white cube"` each go back
to their own pick sites. The chain stops at the first failed step. **Not yet
run as one chain.** Steps 1-4 are the robot-validated scoop chain with the
dump offsets used on the bench. Steps 5-7 have not run on the robot in this
form. The depth gate does not affect the scoop: on
`new_scoop_20260918_151846`, `"white plastic tool"` lost 1 of 3,040 points
across the four cameras, and the extents did not change.

```bash
python scripts/run_experiment.py --workspace-min 0.25 -0.40 -0.13 \
  --skill pick_up --params '{"object_name":"scoop","z_offset":0.0,"grasp_force":1.0,"forward_offset":-0.010}' \
  --skill scoop   --params '{"powder_source":"baking soda","tool_offset":[0.051,0.009,0.028],"tool_span":0.0275,"tool_back_reach":0.030,"bowl_length":0.0275,"bowl_width":0.0205,"bowl_depth":0.0115}' \
  --skill dump    --params '{"target_container":"b 10 ml water","container_category":"clear plastic cup","tool_offset":[0.051,0.009,0.028],"bowl_length":0.0275,"bowl_width":0.0205,"bowl_depth":0.0115,"forward_offset":-0.008,"lateral_offset":-0.015}' \
  --skill place   --params '{"target_location":"scoop"}' \
  --skill pick_up --params '{"object_name":"white cube","z_offset":0.0,"grasp_force":1.0}' \
  --skill stir    --params '{"target_container":"b 10 ml water","container_category":"clear plastic cup","tool_length":0.066,"revolutions":2,"stir_depth":0.02,"lateral_offset":-0.012,"forward_offset":-0.010,"stir_radius":0.022,"seconds_per_revolution":2.0}' \
  --skill place   --params '{"target_location":"white cube","vertical_insert":true,"release_clearance":0.0,"stop_force_n":1.0}'
```

### Stirrer back into its holder: vertical insert from memory (2026-09-25)

`place` has a new `vertical_insert` mode. It homes with `reset_joints`
(still holding), levels the wrist to fingers-down, and **crosses in XY at
the home height** (z≈0.49 in sim). It then goes straight down to the hover
and on into the force probe. The plain path moves diagonally from wherever
the last skill left the arm, which would drag the hanging rod through
whatever sits between the cup and the holder.

**It releases with the TCP back at the grasp position, not the centroid.**
The executor now also remembers pick_up's `grasp_pose` translation
(`SkillsExecutor.pick_grasps`). When the place target names a remembered
pick, it passes that as `pick_grasp_tcp`. Putting the TCP back where it held
the seated stirrer re-seats the stirrer. In sim the grasp was 1.7 mm below
the centroid.

**`place`'s `stop_force_n` probe is now relative to a resting baseline**
(5 readings at the probe start), like stir's guard. Before, it compared the
raw Z reading to the threshold. On the real arm that raw reading carries a
bias, which would release in mid-air or never release. Both helpers now use
`BaseSkill.mean_push_up_n`.

```bash
python scripts/run_experiment.py --workspace-min 0.25 -0.40 -0.13 \
  --skill pick_up --params '{"object_name":"white cube","z_offset":0.0,"grasp_force":1.0}' \
  --skill stir    --params '{"target_container":"b 10 ml water","container_category":"clear plastic cup","tool_length":0.066,"revolutions":2,"stir_depth":0.02,"lateral_offset":-0.012,"forward_offset":-0.010,"stir_radius":0.022,"seconds_per_revolution":2.0}' \
  --skill place   --params '{"target_location":"white cube","vertical_insert":true,"release_clearance":0.0,"stop_force_n":1.0}'
```

Sim, the same chain with `"stirring rod"`:
- The order printed as home, XY at z=0.487, down to 0.30, then probing from
  grasp+15 mm.
- Contact at z=0.1045, 1 mm below the grasp, **+2.9 N**.
- Stirrer **0.2 mm** from its seat, **0.00°** tilt, gripper empty.
- `smoke_test_sim.py --only "pick + stir,scoop then stir"` still passes.

**Not yet run on the robot.** The descent is guarded only from grasp+15 mm
down to grasp−20 mm (`probe_above` / `probe_below`). The ~70 mm above that,
where the rod enters the bore, is position-only. That is safe only while
the rod sits in the jaws as it did at the pick. If the stir contact made it
slide or tilt, raise `probe_above` to about 0.08 and `probe_seconds` to
0.4 (about 30 s of probing).

A probe that feels nothing refuses to release and keeps hold. The scoop's
validated `place` back to its pick site is unchanged, since
`vertical_insert` defaults to false.

### Stir circle centre offset and radius (2026-09-25)

After the first top-down stir of `b 10 ml water` on the robot, the circle
needed to sit **1.2 cm to the right** and be wider. `stir` now takes
`forward_offset` / `lateral_offset`, in the same convention as pour, scoop
and dump: +X away from the base, +Y toward the robot's left. The offset
moves the circle centre off the measured rim centre. It is treated as a
correction onto the true cup centre, so it does **not** shrink the
wall-clearance clamp. The radius is still clamped to
`rim_radius − wall_clearance` (12 mm). If a wider `stir_radius` comes back
smaller, the log prints `Clamping stir radius`; that means the cup is the
limit, not the parameter. Command sent for the next bench run, which
assumes the robot's right:

```bash
python scripts/run_experiment.py --workspace-min 0.25 -0.40 -0.13 \
  --skill pick_up --params '{"object_name":"white cube","z_offset":0.0,"grasp_force":1.0}' \
  --skill stir    --params '{"target_container":"b 10 ml water","container_category":"clear plastic cup","tool_length":0.066,"revolutions":2,"stir_depth":0.02,"lateral_offset":-0.012,"stir_radius":0.022}'
```

Not yet run.

### Force-guarded descent into the cup (2026-09-25)

Stir no longer pushes blindly down to the computed depth. It moves quickly to
where the rod tip is 10 mm above the rim, then steps down 2 mm at a time
(`probe_step`, `probe_seconds` 0.4). After each step it reads the vertical
force. The trigger is a **change of more than `stop_force_n` (1.0 N)** from a
baseline, averaged over 5 readings taken at rest before the descent starts.
Raw Franka force estimates carry a pose-dependent bias, so an absolute
threshold is not used. A change in **either** direction stops the descent.
frankapy says a press reads +Z, but only sim has checked that sign; stopping
on either sign means a wrong sign cannot turn into "never stop", and a false
stop only costs depth.

- Contact with the rod tip **at or above the rim**: the rod has landed on the
  rim, not in the cup. Stir lifts back to the hover and fails.
- Contact **inside** the cup: stir backs off 3 mm (`contact_backoff`), stirs
  there, and reports `stopped_by_force`, `contact_force_n`, `contact_z`,
  `contact_tip_below_rim` and `stir_z`.
- No force reading on the arm: stir warns and falls back to a position-only
  descent (`force_guarded: false`). `"stop_force_n": null` turns the guard off.
- The guard covers the **descent only**. The streamed circle has no force
  check.

Sim, pick_up + stir of the plastic beaker with `tool_length` 0.066 (sim's true
value is 0.081, so the rod reaches 15 mm deeper than planned):
- `stir_depth` 0.02: no contact, stirred at the planned z=0.142.
- `stir_depth` 0.09: contact at z=0.102, **+10.5 N**, tip 60 mm below the
  rim; stirred at z=0.105 and completed 2/2 revolutions.

The +10.5 N shows that sim contact is stiff: one 2 mm step past the floor
already reads 10 N. **Not yet run on the robot.** Two things to watch on the
first run:
1. `grasp_force` 1.0 N may let the cube **slide up in the jaws** before
   1 N of reaction reaches the wrist. Sim cannot show this because its grasp
   is a magnet. The width check does not catch it either, since sliding does
   not change the jaw width. If the rod rides up, raise `grasp_force` toward
   the 1.5 N cap, or lower `stop_force_n`.
2. How noisy the resting force baseline is on the real arm. If the stop fires
   in mid-air, raise `stop_force_n`. If it presses hard before stopping,
   use `probe_step` 0.001.

`test_stir_stops_descending_on_contact` covers: floor contact with a −2.5 N
sensor bias, contact on the rim, no contact, and an arm with no force
reading. `test_skills_offline.py`: 195 pass, and the 2 failures are the same
scoop workspace-floor ones as before.

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
  as a second choice since that GPU is also running SAM 3.
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
| `robochem/skills/{place,scoop,stir,dispense}.py` | Rewritten 2026-09-08, not yet bench-tested (stir top-down default 2026-09-25) |
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
