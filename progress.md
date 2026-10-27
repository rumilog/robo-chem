# Progress — robo-chem (Franka + RealSense)

Last updated: 2026-10-27

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
| Live factory intrinsics + board-free cube extrinsic calib | **Recalibrated 2026-10-10** (cube-model + table fit, `--level-table -0.0112 --cube-offset 0.0065 0`): cameras agree to 0-8 mm on flat paper (was up to 40), dish rims within 2.9 mm, table -10.1..-11.4 mm vs measured -11.2, fingertips centred in the clear cups by ruler. The Sep 4 "~3-6 mm held-out" hid a ~18 mm per-camera bias (see end of file) |
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

## Instruction → VLM → sim: how to run it (2026-10-02)

The whole loop runs in sim with a live model. An instruction comes in as a
text task or as a photographed sheet; the model plans skill calls; the sim
executes them. Needs `openai_api_key=` in the repo's `.env`. Verification stays
off in sim, because there is no chemistry.

```bash
source scripts/env_dev.sh
# a typed instruction: the whole text goes to the planner
python scripts/run_experiment.py --sim --workspace-min 0.25 -0.40 -0.13 \
  --task "Add one scoop of citric acid to the white paper cup, then stir the cup with the stirring rod. Put the scoop and the stirring rod back where you found them."
# a photographed instruction sheet; --follow-steps passes its steps on too
python scripts/run_experiment.py --sim --workspace-min 0.25 -0.40 -0.13 \
  --instruction-image instructions/citric_acid_stir.png --follow-steps
```

`instructions/citric_acid_stir.png` is a card written for the sim bench. The
booklet's Magic Beaker (pages 12-13) used to be refused in sim for want of red
cabbage powder, water, clear cups and liquids. Since 2026-10-02 the bench has
them, see "The Magic Beaker kit in sim" below:
`--instruction-image instructions/magic_beaker.png --follow-steps`.

**The typed `--task` above is stale since 2026-10-03: the sim bench has no
white paper cup any more** (see the Magic Beaker kit section). The model does
not refuse it. It substitutes a cup without saying so. Connectivity check on
2026-10-07 after pulling `6b49166`, with `--dry-run --no-viewer`, layout seed
902184 (log `agent_run_20261007_162148.json`, run from the scratchpad, not
kept in `experiments/`):
- Run log `models`: `strong_model gpt-4.1`, `cheap_model gpt-4.1-mini`,
  `base_url None` (plain OpenAI), `api_key_present True`. The scene call saw all
  11 sim objects by their inventory names.
- The plan was 7 sub-tasks, and all 7 resolved to valid skill calls with no
  replan. Dump and stir both went to **`clear cup c`**. The planner's own
  `expecting` line reads "the white paper cup (clear cup c)". Nothing was
  flagged as a substitution.
- So a task that names a container the bench lacks is quietly remapped, not
  blocked. To repeat the 2026-10-02 run, name `clear cup a` (what the card
  names now) in the text.

**Measured on 2026-10-02 (live gpt-4.1, layout seed 829181).** Both commands
above ran all 7 planned steps:
1. spoon, scoop citric acid, dump into the paper cup, spoon back;
2. stirrer, stir the paper cup, stirrer back into its holder.

Neither run touched anything but the grasps, and nothing moved more than 3 mm.
Another seed (31337) was clean on the text task too. Getting there took these
fixes:

- **Image instructions lost their steps.** The orchestrator hands the planner
  the parsed goal alone, which is the research design. Given only "prepare a
  citric acid mixture", the planner dumped into the beaker and never stirred.
  `--follow-steps` (opt-in) appends the sheet's steps.
- **`InstructionParser` was pinned to `gpt-4o`** (API shutdown 2026-10-23). It
  now uses `agents.config.strong_model()`. **Still pinned, still to do before
  10-23:** `vision/label_resolver.py` (the real cell's labelled-cup reads),
  `vision/scene_analyzer.py`, and the legacy `task_planner` / `vlm_orchestrator`.
- **The scene agent called the citric acid dish "empty"** in about 1 run in
  4, and the planner then refused the task. The sim labels are blank paper,
  now mostly hidden under the 102 mm dish. The prompt now says a
  reagent-named inventory entry is that container's label. After the change,
  3 of 3 scene calls on the failing layout reported "citric acid".
- **Putting tools back.**
  - The model put the spoon "beside the citric acid", landed it on the dish
    and pushed the dish 11 cm.
  - It put the stirrer "beside the stirrer holder", which dropped it on the
    bench, and the hand hit the beaker.
  - Now the catalog tells it to name the tool itself. The executor also maps
    a tool's alias or its home ("stirrer holder") to the tool's pick site and
    fills its put-back params from `skills/tool_geometry.py` (the stirrer:
    `vertical_insert`, force-probed seat).
- **Stir got pick_up's measured z (0.017) as `tool_length`,** because the
  orchestrator's note said to pass it to "any that takes tool_length". That
  is 64 mm short for the stirrer. The note no longer says so, and the executor
  raises a `tool_length` under 80% of the tool's CAD length (0.081) to CAD.
- **Twice the stir skill call came back as broken JSON,** despite strict
  schema, and the second time blocked the task. `structured_completion` now
  re-asks once, slightly warm.

**Random layouts: tall containers now keep 18 cm from the powder dishes.**
The scooping hand is about 200 mm across its jaws, and the wrist reaches
toward the base. Centre distances measured in sim: 144 mm hit, 145 mm hit,
161 mm hit (link7), 172 mm clear. **The fixed default layout breaks this
rule:** scooping BAKING SODA there drives the hand into the plastic beaker
(144 mm, 1699 contact steps). Every validated chain scooped citric acid, which
is clear. `randomize_layout` also now checks each draw against every other
prop's current spot. 200 of 200 seeds give a clean layout.

**Where Aliyah's repo lives.** `robomail/` is the rumilog submodule (b7b83a0).
Do not clone over it. Aliyah's repo (aliyahreddingdidit/robomail) belongs in
`robomail_Aliyah/`. That path is a committed gitlink to her 4bee1ac with no
`.gitmodules` entry, so `git submodule update` does not fill it, and a fresh
checkout leaves it empty. Clone her repo there by hand. When it is missing, the
offline tests skip the vocabulary bridge group.

On 2026-10-02 her clone was found over `robomail/`. It was moved to
`robomail_Aliyah/` and the submodule was restored. Results after the move:
- `test_skills_offline.py`: 201 passed, plus the 2 old scoop failures. The
  bridge group passes.
- `smoke_test_sim.py`: all 9 groups pass.

## The Magic Beaker kit in sim: props, liquid, chemistry (2026-10-02)

The sim bench (`sim/bench.py default_bench`) holds what the booklet's Magic
Beaker (pages 12-13, `instructions/magic_beaker.png`) needs and nothing else,
laid out the way the real bench is. **Since 2026-10-03 that means seven
containers.** The user asked to keep only what the experiment needs, so the
white paper cup (a leftover distractor) and the separate 50 ml water cup went.
The beaker now starts with the 50 ml.

| Prop | What it is | Measured? |
| --- | --- | --- |
| citric acid / baking soda / red cabbage powder cup | the same 91/102/29/8 mm dish (`_powder_dish`), 18 mm bed | dish yes; red cabbage colour no |
| clear cup a / b, labels `a 10 ml water` / `b 10 ml water` | 10 ml water each, as on the real bench | **47.3 mm tall, 55.5 mm across the outside, measured by the user 2026-10-03.** Wall 1 mm and base 3 mm are not measured. A straight cylinder (53.5 mm inside, 100 ml), so 10 ml stands 4.4 mm deep, a bit more in the real tapered cup. Until 2026-10-03 the sim had 52 x 66 mm cups read off depth (`scene_captures/b10_water_20260925_153459`), where 10 ml was 3.0 mm deep. |
| clear cup c | empty, for the final mix | the same cup |
| plastic beaker, label `50 ml water` | the indicator's 50 ml, there from the start | beaker as before |
| smaller spoon (`small scoop`) | **placeholder**: spoon.stl at 75%, CAD in `tool_geometry.py` from sim | no part exists |
| larger spoon, now also `big scoop` | unchanged | yes |

Water comes pre-measured because `pour` cannot meter: it tips until the level
reaches the rim. That is what the real bench does too.
`instructions/citric_acid_stir.png` now names clear cup A; it used to name the
white paper cup.

**`robochem/sim/lab.py` keeps the books on contents**. It is on by default in
`build_cell`; `film_sim_skill.py` turns it off and keeps its own tally.
- Every scoop's load is tracked by reagent (`powder.ScoopTally`, now multi-bed).
- Powder lands in whatever opening is under the bowl's lowest rim point.
- Liquid runs over a tilted container's lowest rim point once the level surface
  reaches it, and lands in whatever opening is below. A lattice gives the volume
  kept at each tilt.
- Powders dissolve: tau 45 s still, 2 s while a stirrer's tip moves in the liquid.
  Baking soda is capped at 0.13 g/ml of warm water.
- Citric acid (3 eq) and bicarbonate neutralise each other and foam.
- A rough pH (triprotic titration, flattened) colours the red cabbage indicator.
- None of these constants is measured on the kit.

After every skill it prints `[lab]` lines and adds them to the result as
`sim_lab`, for the log only; the planner never sees them. `run_experiment
--sim` prints the end state.

**The scripted Magic Beaker** (`scripts/sim_magic_beaker.py`, default layout; it prints the contact audit and contents). It was 28 steps on 2026-10-02; with the water already in the beaker it is 25:
- Steps: big scoop citric acid into A, baking soda into B; small scoop red
  cabbage into the beaker (water cup into the beaker, 2026-10-02 only); stir A, B, the beaker
  with `to_floor`; beaker 80 deg into A, 90 into B; A and B into C; everything
  put back.
- Outcome: A pH ~1.7-2.0, **red**; B pH 8.3, **blue**; C fizzes 9.7 meq, pH
  5.8-6.2, **purple/magenta**. The booklet says purple.
- Every pour landed in its target; nothing ran onto the table.
- Leftover citric acid in the big scoop fizzed B a little when the same scoop
  took baking soda. That is real.

*The runs below, through seed 4242, were on the 9-container bench of
2026-10-02, which had a water cup and the paper cup.*

**Live, planner-driven (gpt-4.1), fixed layout, 2026-10-02.** Run with
`--instruction-image instructions/magic_beaker.png --follow-steps` (record
`experiments/agent_run_20261002_*`).
- The plan was 36 sub-tasks. All executed with no replan, and every tool and
  cup went back by its own name.
- Contents: A got 0.96 g citric acid, B 1.20 g baking soda (+0.26 g of the
  scoop's leftover citric acid), the beaker 0.28 g red cabbage + 50 ml.
- Pours: beaker into A 26.1 ml, into B 23.9 ml; A into C 30.9 ml, B into C
  33.9 ml.
- C fizzed 10.2 meq and ended at **pH 6.4, purple**. A kept 5 ml, red.
- The only contacts were the stirrer meeting the floor of A and B while
  `to_floor` felt for it. Nothing on the bench moved more than 5 mm.

**Same, random layout seed 4242** (the dishes and clear cups swapped round):
- 34 sub-tasks, all executed, and again only the `to_floor` floor touches.
- The big scoop took only 0.49 g of baking soda here, against 1.20 g on the
  fixed bench. C fizzed 2.5 meq, the acid was left over, and C ended **pH 4.1,
  pink** instead of purple. How much a scoop collects is the outcome's main
  lever.
- The first try of this seed stopped dead after step 30's pour, with no
  traceback and no OOM line visible. Two sims were rendering with EGL at the
  time. The rerun, alone, went through. Watch for it.

```bash
python scripts/run_experiment.py --sim --workspace-min 0.25 -0.40 -0.13 \
  --instruction-image instructions/magic_beaker.png --follow-steps
```

### 2026-10-03: seven containers, and three things trimming them uncovered

- **The layout was redone for seven containers.** Annealed with 2 cm of slack
  on every rule, then centred: each container has 16-27 mm of slack (it was
  4-26).
  - Acid on the left (dish behind cup A), base on the right (dish behind
    cup B), the beaker in the middle, red cabbage behind it on the left, cup C
    at the front left.
  - 200 of 200 random seeds break no rule, keep every container in 3+
    cameras, and move every container (median 133 mm); before, some could not
    move at all.
- **Segmentation was rendered through multisampling.** The IDs at every edge
  were the average of two IDs, which spells a third geom.
  - Usually such a pixel just landed on some unrelated geom. On this bench one
    spelled 355 against a 316-entry table, and MuJoCo's decode raised
    `IndexError` on the first locate.
  - With more geoms the stray pixels had been assigned silently: a clear cup's
    mask picked up a pixel 0.37 m away and was rejected as "includes the table".
  - `SimVision` now renders depth and segmentation with `offsamples=0` and
    decodes the IDs itself, dropping any ID that names no geom. RGB for the VLM
    keeps antialiasing on a second renderer. The decode matched MuJoCo's own
    pixel for pixel wherever MuJoCo's did not crash.
  - 0 stray pixels per camera now.
- **The scoop's yield hangs on the tool offset's x.**
  - The crisper depth moved the spoon grasp 2.8 mm toward the bowl, and the
    big scoop fell from 1.42 to 0.41 ml at every dish position tried.
  - Passing x by hand at the default dish: 25.7 mm (CAD) gave 0.40 ml; 23.5
    gave 0.68; 21.5 (the truth) 1.24; 19.5 1.83. **2 mm is a factor of 2-3.**
  - pick_up's own x (median of the bowl end) was 29.2, 7.7 mm off, so the
    planner, copying it, scooped 0.05-0.07 g into A.
  - pick_up now also reports `tool_extent_x`, the cloud's [1st, 99th]
    percentile along the tool x.
  - For a known tool the executor sets x = the span's middle +
    `span_mid_to_bowl` (spoon.stl: 21.05 mm; smaller spoon 15.8). y and z
    come from CAD. This is used when no offset is given, or when the given one
    is pick_up's own suggestion copied over; a value passed on purpose (the
    bench's [0.051, ...]) is kept.
  - The far edge alone did not work: the bowl's tip is thinly sampled, and its
    98th percentile fell 5 mm short.
  - Measured: 19.6 vs 21.5 mm (larger), 15.3-16.5 vs 16.4-16.5 (smaller). Every
    scoop filled: 1.83 ml big, 0.77 small. The error is on the deep side, about
    1.5 mm closer to the floor than the 3 mm gap.
  - **On the robot this changes planner-driven scoops whenever pick_up ran in
    the same session.** Check the first one.
- **Dump can't hold the bowl still close in.**
  - A cup at x <= 0.257 m failed or drifted 16-18 mm (cup C's spot at x 0.268:
    6.1 mm). From x 0.276 it held within 0.4 mm, left or right, at every angle
    from -50 to +50.
  - New rule: `LAYOUT_MIN_X` 0.28 for every container.
- **The planner tipped cup A only 80 deg into C**, taking "80 for the first of
  two" from the pour note. That wording was about splitting one container. Now
  the note says: 90 for any pour that should empty the source, about 80 only to
  pour part and keep the rest.

**Validated on the seven-container bench (2026-10-03), final code:**
- `smoke_test_sim.py`: 10/10.
- `test_skills_offline.py`: 220 passed, plus the 2 old scoop failures.
- `test_agents_offline.py`: 61/61.
- `scripts/sim_magic_beaker.py` (25 steps, no floor or offset hints):
  - A got 1.33 g citric acid, B 1.19 g baking soda, the beaker 0.28 g red
    cabbage.
  - C ended at pH 5.1, magenta.
  - The only contacts were the `to_floor` floor touches.
- Scoop into and dump out of every container: every big scoop full
  (1.47-1.83 ml), no contact.
- Live, seed 777: 31 sub-tasks, no replan.
  - Beaker 80 into A, 90 into B; A and B into C.
  - C fizzed 8.5 meq and ended at pH 5.1, magenta.
- Live, fixed layout: everything executed, but **the planner tipped the
  beaker 90 into A for "pour some indicator"**. All 50 ml went to A and B got
  none.
  - The reworded pour note did not settle it. The same sheet gave 80 in the
    seed-777 run and 90 here. `pour` cannot meter, and the model's choice of
    angle varies run to run.
  - **Fixed in the orchestrator, deterministically** (`POUR_KEEP_SOME_DEG`).
    Before a pour runs, it looks down the plan. If the held container pours
    again before any later sub-task puts something down or picks something up,
    `pour_angle` is capped at 80. The change is noted as `[adjusted]` and in the
    step record's `adjusted`.
  - Why 80: at a given tilt a container keeps what lies below its rim whatever
    it held before, ~24 ml for the beaker at 80 (the sim's geometry). The
    sheet gives no amounts ("pour some"), so any split works as long as B gets
    some.
  - A container that pours only once (A or B into C) still goes to 90.
  - On the robot, liquid clinging to the wall leaves a little more behind,
    which is the safe side.
  - `test_agents_offline.py`: 66/66, including this case.
  - First live check, fixed layout, 2 runs:
    - one planned 80 then 90 itself (A 26.1 ml, B 23.9 ml; C pH 5.4), so the
      guard had nothing to do;
    - the other died on a different fault (below).
- **The skill-call agent wrote `","` for `pour_angle`** (twice that day). The
  value went through as a string, and pour died converting it.
  - `coerce_params` now checks every value against the catalog's kind.
    "90" becomes 90.0. An optional value that is not its kind is dropped (the
    skill's default applies) and printed; a required one is a planning failure.
  - On the same run, the replan's pour call came back as cut-off JSON twice
    and blocked the task. `structured_completion` now re-asks twice (at
    temperature 0.3, then 0.6) instead of once.
  - `test_agents_offline.py`: 69/69.
  - With both fixes, two live runs (fixed layout, and seed 4242) went
    through: 31 sub-tasks each, no replan.
    - Beaker 80 into A (26.1 ml), 90 into B (23.9 ml).
    - A at 80 into C (32.4 ml; 3.7 ml left in A, red). B at 90 into C (33.9 ml).
    - C ended at pH 5.4, magenta.
    - The only contacts were the `to_floor` floor touches.
    - The planner chose 80 itself both times, so the guard did not have to
      act. The offline test covers the case where it must.
  - The same run also passed `tool_offset` to pour. It was rejected and the
    replan carried on to the end (9 sub-tasks), as the new replan rule asks.
- **With the measured 47.3 x 55.5 mm cups** (same day):
  - Smoke 10/10. The scripted chain was clean: beaker into A 26.1 ml, into B
    23.9; A into C 36.1, B into C 33.9; C at pH 5.1.
  - The dump drift stayed at 0.1-0.2 mm at the x = 0.28 boundary (5 spots), so
    the rule holds for the shorter cup.
  - The live fixed-layout run **again tipped the beaker 90 into A** ("pour
    some" read as all). A malformed pour parameter (`','` for a number) was
    replanned around. That makes 2 of 3 fixed-layout runs on this sheet; the
    pour-amount question is open.
- **Grasp slip is not tested by the sim.** The default magnet grasp cannot
  slip.
  - With `--sim-grasp-mode physics`, the fingers passed through a clear cup's
    0.8 mm wall: closed to 36 mm on a 66 mm cup and lifted nothing.
  - The beaker, which the robot pours fine, slid out during the tip.
  - That mode is not calibrated. Whether a clear cup can be picked and tipped
    to 90 deg without crushing or slipping is a bench question, still to
    answer.

### Pour keeps the held cup's lip just over the target (2026-10-03)

The user saw cup-to-cup pours as far too high and too far forward. Measured
during the old pour (A into C, the beaker into A):
- the lip (the held cup's lowest rim point, where the liquid leaves) was
  **155-185 mm above the target's rim**, over its centre;
- the cup's body hung out beyond the target, away from the base.

The old site is the fixed one tuned on the robot for beaker into paper cup:
target centroid, +0.05 lean shift, -0.08 forward offset, TCP 0.15 m over the
target's top, +2.5 mm per 10 deg.

**New path** (`PourSkill._pour_over_lip`). It is used whenever the held cup's
shape is known; otherwise pour falls back to the fixed site.
- **Where the shape comes from.** `pick_up` now reports the object's top and
  bottom z (1st/99th percentile) and radius. The executor turns them into
  `source_top_above_tcp`, `source_height` and `source_radius` and fills them
  into pour; the bottom is the table when `table_z` is set.
- **Where the lip goes.** Pour models the held cup as rings in the tool frame.
  At each 10 deg tip step it puts the lip `lip_clearance` (15 mm) over the
  target's measured rim, `lip_inset` (0.4) of the inside radius toward the
  base from its centre.
- **How high the cup rides.** The TCP is raised wherever needed so that every
  point of the cup stays `body_clearance` (10 mm) over the rim height. An
  upright cup's bottom is lower than its lip, so the cup starts higher and
  comes down as it tips.
- **Approach and finish.** The cup arrives 6 cm over the start, goes down,
  tips step by step, holds, then is righted in place 4 cm up before going home.

Measured in sim, after:

| Pour | Lip over the rim | Lip from the target centre | Lowest point of the held cup over the rim |
| --- | --- | --- | --- |
| A into C, to 90 | 57 mm at the start, 19 mm at 87 deg | 11-13 mm toward the base | 11-17 mm |
| beaker into A, to 80 | 104 mm at the start, 35 mm at 77 deg | 11-14 mm toward the base | 11-14 mm |

The beaker's lip stays higher because its bottom edge, 97 mm away, swings
lower than the lip until late in the tip. The same liquid went in as before:
10.0 ml and 26.1 ml.

**The grasp height mattered too.** The bench-validated `z_offset` 0.02 for the
beaker puts the TCP 7 mm ABOVE a 47 mm cup's rim, so the pads, which reach
8.9 mm past the TCP, held its top 2 mm. With pick_up's default (z_offset 0)
the rim is 13 mm above the TCP. `sim_magic_beaker.py` now picks cups on the
defaults, as the planner does.

**On the robot:** after a pick_up in the same session, pour now takes the new
path. It has not run on the robot; the fixed site it replaces was validated
there for beaker into paper cup.

`test_skills_offline.py`: 227 passed, plus the 2 old scoop failures. The new
`pour: lip over the target` group covers the final lip at 15 mm, never under
it, the lip's spot, the descent, and the fallback.

### What the arm actually sweeps, and the layout rules made from it

Measured from the links' collision geometry in sim, in the target's radial
frame. The scratch scripts that did this were not kept. They sampled every arm
link's collision vertices below 0.13 m every 25 steps of a dump, pour or scoop
into each target.

| Motion | Arm below 0.10 m | Arm below 0.13 m |
| --- | --- | --- |
| dump into a 52 mm cup | ahead -0.05..+0.17, aside -0.06..+0.12 | -0.09..+0.27, -0.13..+0.15 (link6/7 reach out) |
| dump into the 95 mm beaker | none | -0.04..+0.14, +-0.03 |
| scoop a dish | -0.08..+0.14, +-0.13 | -0.08..+0.14, +-0.13 |
| pour (any) | none | none |

Nothing comes lower than ~26 mm above the target's rim. The rules in
`bench._conflicts` (`layout_violations`) come from this table:
- `DUMP_SWEEP` keep-out boxes by obstacle height above the target rim; nothing
  within 15 mm of the rim height is constrained.
- Tall things are kept 0.18 m from dishes (the old rule, now confirmed by the
  scoop sweep).
- Bodies are 3 cm apart, labels must not overlap, nothing within 12 cm of the
  holder.
- A tall container is at least 0.38 m from the base. The paper cup at 0.358 m
  could not be reached by dump after a scoop on the far side (stopped 22 mm
  short); at 0.38 m it worked.

The layout was annealed under these rules, then each container was moved to
the middle of the spot it may occupy, with at least 3 cameras seeing it. That
left 4-26 mm of slack per container. The rest of this note does not need the
scratch scripts: `layout_violations` and the shuffle are in `bench.py`.

`randomize_layout` now first **swaps containers of the same body** (the three
dishes, the four clear cups), then jitters each within 6 cm under the same
rules plus a 3-camera check. Jitter alone barely moved anything with the full
kit on the bench.
- 200 of 200 seeds break no rule and keep every container in 3+ cameras.
- Median move is 142 mm.

### Failures that cost time, and what was changed

- **My contact audit was silently off.** `kit_chain.py` wrapped
  `cell.arm._step` and then restarted the lab, which put the class method back.
  That gave "0 contacts" for runs that had them. The audit must patch
  `SimFrankaArm._step` on the class before `build_cell`.
- **Corridor too short.** With the first layout, dumping into cup B drove
  link6 into the seated stirrer, 0.20 m ahead and 0.096 aside. That is outside
  the 0.22 x 0.09 corridor I had guessed, so the sweep was measured instead.
- **A measured tool_offset pointing at the wrong end.** pick_up's
  `measure_tool_offset` took the handle end as the bowl: x was -30 mm against
  CAD +25.7. The grasp sat 16 mm toward the bowl, after the spoon moved to
  (0.62, -0.03). The live VLM run passed it to scoop and dump. The hand flung
  two dishes 2.3 m and hit the beaker and the water cup. `_fill_tool_geometry`
  now discards a measured offset whose x points the other way from the CAD, or
  that is more than 20 mm to one side. The bench-validated [0.051, 0.009, ...]
  is still kept, z raised.
- **`to_floor` stir.** A rim-relative `stir_depth` (0.03) never reaches 10 ml
  in a 52 mm cup: the tip stopped 18 mm above the water. On the real robot,
  `stir_depth` 0.02 in `b 10 ml water` probably did not reach it either.
  `stir(to_floor=true)` feels down to the floor (1 mm steps, 4 mm back-off) and
  is refused without a force reading. Getting it clean took three fixes:
  - the rod went **through** the 3 mm base, because default soft contact let it
    sink past half the thickness; container bases and the rod now use the
    printed parts' stiff `solref`;
  - the fingers (8.9 mm past the TCP) and the stirrer head brushed the rim:
    pick_up holds the head 6.8 mm low, so it hangs ~22 mm under the TCP. Stir
    now keeps the TCP `hand_clearance` 30 mm above the rim, whatever depth was
    asked;
  - the rod (66 mm) cannot reach the floor of the 95 mm beaker with the hand
    clear, so the beaker is stirred above its 50 ml (`hand_limited`). The
    indicator still dissolves unstirred in the sim. Expect the same on the
    robot.
- **The planner set things down "beside" others.** The water cup went onto the
  beaker's spot, and the beaker onto cup B and then into cup A. Cup A went
  over and 34 ml ran onto the table. Fixes:
  - `place` now puts a held object that was picked from the bench back where
    it came from whenever it is asked to go beside something else;
    coordinates and `on_top` pass;
  - the catalog says to name the object itself.
- **A replan ended the run early.** After that failed place, the corrective
  plan was the place alone, and the orchestrator reported success with the
  final pours never run. The replan prompt now says the plan replaces
  everything that has not run, so it must carry on to the goal.
- **The scene agent called water cups empty and dropped objects.** It got the
  inventory as a flat list, with "clear cup a" and "a 10 ml water" as two
  names. `SimVision.inventory_notes` now gives one line per object, with its
  label and other names ("larger spoon -- also called big scoop"). The scene
  agent also now gets two opposing views (cams 2 and 4), not 2 and 3 from the
  same side. Next dry run: "clear cup a | a 10 ml water", and the big scoop was
  used for citric acid and baking soda.
- **The planner tipped the beaker 40 deg to "pour some".** That pours nothing.
  The pour note now says liquid leaves only once the level reaches the rim:
  about 80 for "some", 90 for the rest. The next plan used 80 then 90.
- **InstructionParser misread step 7** as "into the beaker" in 2 of 3 reads.
  At temperature 0, with a line asking to keep each step's containers, it read
  "the empty clear cup" in 3 of 3.
- `MAX_SUBTASKS` 24 -> 40: the plan is 28-34 sub-tasks.
- **The big scoop came up empty in every planner-driven run.** The cause was
  the old open item, the floor estimate. `base_z` (3rd percentile of the cloud)
  reads 3.0 mm for a dish standing on z=0, even with no depth noise. The
  beaker reads 6.1 mm and the cups 2.3-5.3. The bottom edge is only seen at a
  grazing angle, so that percentile lands on the wall; `top_z` is within 1 mm.
  The floor came out 3 mm high, and the larger spoon's mouth stayed above the
  bed: 0.00 ml. The scripted chains never hit this because they pass
  `container_floor_z`.
  - Fix: the executor config takes `table_z`. When it is set and the container
    stands on it (base_z - table_z within -5..+12 mm), scoop takes the inside
    floor as `table_z + floor_thickness`. `build_cell` sets 0.0.
  - Planner-style scoops (no floor hint, a bad measured offset) then carried
    1.42 ml citric acid, 1.43 ml baking soda and a full 0.77 ml small scoop.
  - **On the robot nothing changes until `table_z` is configured.** The cage's
    table sits at about z = -0.013 to -0.017: the ring around the dish and the
    B cup in `scene_captures/b10_water_20260925_153459`, cams 2-5.
- **A DNS hiccup killed two runs**: `openai.APIConnectionError` escaped every
  handler. `llm_client._create` now retries transient errors after 3, 10, 30
  and 60 s, then raises `LLMResponseError`, so the stage reports blocked
  instead of crashing.
- **On random layout seed 4242 the scene agent described 6 of 13 objects.**
  The planner took the citric acid dish and the beaker to be missing and
  refused the task. `comprehend_scene` now appends every inventory object the
  agent left out, with its label as contents, marked as such. The re-run listed
  13 and planned 34 sub-tasks.

### 2026-10-07: whole Magic Beaker run on the robot PC, before going to the arm

This is the user's command, run on the robot PC through `perception_env`. The
user's own computer had already run it clean. This run was on `6b49166` plus
the uncommitted skills edits then in the tree.

```bash
source scripts/env_dev.sh
python scripts/run_experiment.py --sim --workspace-min 0.25 -0.40 -0.13 \
  --instruction-image instructions/magic_beaker.png --follow-steps
```

- **`outcome: success`.** All 31 sub-tasks executed, with 0 replans and 0
  failed steps, in 1214 s. Models were gpt-4.1 and gpt-4.1-mini. This was the
  7-container bench on random layout seed 980455. Record:
  `experiments/agent_run_20261007_165912.json`.
- The plan was:
  1. big scoop: citric acid into A, then stir;
  2. big scoop: baking soda into B, then stir;
  3. small scoop: red cabbage into the beaker, then stir;
  4. beaker into A at 80° and into B at 90°;
  5. A into C at 80° and B into C at 90°.
- **The final cup C was magenta, not the booklet's purple.** The sim's lab
  reported 66.3 ml at pH ~4.9 and "fizzing". A had poured 32.4 ml into C and
  kept 3.7 ml at pH 1.9; B poured all 33.9 ml at pH 8.3. This is a chemistry
  ratio question, not a motion one. Not investigated.
- **The process does not exit after the run.** The viewer stays open until its
  window is closed (`--sim-hold` defaults to forever). Stdout redirected to a
  file is block-buffered, so the final summary does not appear in it until
  exit. Read `experiments/agent_run_*.json` for the outcome instead.
- **The robot venv (Python 3.8, `~/franka`) imports the whole agent stack.**
  - Every `robochem.agents.*`, the orchestrator, every skill and
    `label_resolver` imported. `openai` there is 2.2.0. `compileall robochem`
    was clean.
  - `test_agents_offline.py` passed 69/69 under 3.8.
  - **`test_skills_offline.py` crashes under 3.8**, in
    `test_place_does_not_drop_from_height`. The test swaps in
    `StubRigidTransform` only when `autolab_core` is missing, and its
    `FakePose` subclasses the stub. The robot venv has the real
    `autolab_core`, so the isinstance check misses. This is a harness issue,
    not a skill one: keep running that file from `perception_env`.

**Real-bench labels agreed with the user for the first arm rehearsal (not yet
run).** The rehearsal uses empty containers, no powder and no water, to check
that the arm moves as it does in sim. The labels sit under the containers. The
tools carry no label.
- `CITRIC ACID`, `BAKING SODA` and `RED CABBAGE POWDER` go under the three
  dishes.
- `50 ML WATER` goes under the **plastic beaker**. There is no separate water
  cup.
- `A 10ML WATER` and `B 10ML WATER` go under cups A and B.
- **Cup C must not say water.** The user had `C 10ML WATER`, which should be
  changed to `C EMPTY`. The scene prompt (`agents/scene.py`, label rule) tells
  the model to trust a label over what it sees in a clear cup. Three "10 ml
  water" cups leave no empty cup for the booklet's last step, so the plan
  would stop matching sim's.
- The stirrer starts seated in its holder, and the holder must be on the bench.
- **Risk for the empty rehearsal.** The same rule makes an exception for a
  container the model "can clearly see is empty". Empty white dishes and an
  empty beaker may be reported as empty, and an "empty" citric acid dish has
  made the planner refuse before (2026-10-02). Check it with a real
  `--dry-run` (no motion), and compare its plan with the 31 steps above
  before any motion. If a dish reads empty, put a pinch of any powder in it.

**First real-cell agent dry-run: failed at the scene stage (2026-10-07).**
Command:

```bash
source scripts/env.sh   # grounding was already up: sam3, exemplars []
python scripts/run_experiment.py --dry-run --workspace-min 0.25 -0.40 -0.13 \
  --instruction-image instructions/magic_beaker.png --follow-steps
```

`--dry-run` on the real path never constructs `FrankaArm` (`build_stack`), so
nothing can move. All four cameras captured. Record:
`experiments/agent_run_20261007_170730.json`; frames looked at by eye.
- **The scene agent reported 6 objects: the 3 dishes and "clear cup" x3.**
  The frames show 4 clear containers, 2 scoops lying flat, and the stirrer
  (white cube head) seated in its black holder. `scene_notes` called the
  tools "small white plastic parts and a black cylindrical object ...
  distractors". It also called every cup "empty" and read no A/B/C label.
- **The planner invented a pipette**, failed sub-task 1 ("not listed among
  the visible objects"), then replanned to 0 sub-tasks.
- **Why it differs from sim.** `VisionSystem.known_object_names()` returns
  `[]` on the real cell, so `comprehend_scene` takes its open-vocabulary
  branch: kit vocabulary only, with no inventory and no "trust the label"
  rule. Both of those exist only in the inventory branches that the sim
  feeds. The earlier advice about the C label above assumed the inventory
  prompt. `C EMPTY` is still right, but on the real cell today nothing
  reads it.
- **The real "beaker" is a clear plastic cup** on the `50 ML WATER` label. It
  is not the red translucent `plastic beaker` that the `"plastic beaker"` SAM
  prompt and the validated pick params were tuned on.
- The bench also had some things that were not part of the experiment:
  - a white paper cup and a small round object at the back left (cam 5);
  - a stack of white bowls in the back right corner (cam 2);
  - spilled powder on the table.
  
  The two scoops lay a few cm apart, next to the stirrer holder.
- Next: give the real cell an inventory like the sim's. That means a name per
  object, mapped to how real perception finds it: a label plus a category,
  or a SAM prompt. Telling the two scoops apart needs its own rule, because
  both answer to `"white plastic tool"`.

**Real bench manifest (2026-10-07).** Code exists and offline tests pass. On
the cell, only localisation has run, never motion.
- **`robochem/vision/bench_manifest.py`, `MAGIC_BEAKER`.** It has 10 objects
  under the sim's names. Each one is found in one of two ways:
  - a label plus a SAM category: `"white bowl"` for the dishes,
    `"clear plastic cup"` for A, B, C and the beaker;
  - a swept SAM prompt: `"white plastic tool"` for the scoop, `"white cube"`
    for the stirrer.

  **There is one scoop.** The user took the small one off the bench, so
  `small scoop` and `smaller spoon` are aliases of `larger spoon`. The
  holder is never located: put-back goes to the stirrer's pick site, as
  before.
- **`run_experiment.py --bench magic_beaker`** is the default; `none` is the
  old open vocabulary. With it, `VisionSystem` changes in three ways:
  - it hands the scene agent the sim-style inventory
    (`known_object_names`, `inventory_notes`);
  - `locate()` routes names through the manifest. A labelled container
    whose label is not read fails. It is not retried as a direct prompt;
  - `SkillsExecutor._bench_names` renames aliases and labels to the bench
    name before pick sites and tool CAD are looked up. Without that,
    "smaller spoon" would get the 75% placeholder CAD for the big scoop.
- **The label reader now chooses from the bench's 7 labels instead of
  transcribing.** It runs on `strong_model()` (gpt-4.1), no longer the
  pinned gpt-4o. `snap_to_candidates` maps a partial read that fits one
  label ("EMPTY" → "C EMPTY"), and drops anything else. `normalize_label`
  splits "10ML" into "10 ml".
- **A camera that reads one label on two different cups is no longer used**
  for that request (`pick_matching_instance`). It used to take the higher
  score.

Measured with the scratchpad locate script, which was the first version of
`scripts/check_bench_layout.py`:

| | free transcription, gpt-4o (run 1) | closed set, gpt-4.1 (run 2) |
| --- | --- | --- |
| dishes | 2 of 3 from cam 2 alone, 1 from 4 cams | all 3 from 4 cams |
| invented labels | "A 10% NaOH", "ACETONE", "XOS SCRUB", "B 0.1 M NaOH" | none |
| C | not found ("C 10 ML WATER" on cam 2, "EMPTY" on cam 4) | cam 5 only |
| B | 2 cams | **not found**: seeds cam 2/3 57 mm apart |
| A | 2 cams | **wrong cup**: cam 3 read two cups as A, the pick seeded cam 4, and A came back 39 cm from where it stood |
| scoop, stirrer | 3 and 3 cams | 3 and 4 cams |

Run 2 is why the duplicate rule changed. The cause is the layout: the four
clear cups stood in a tight row, each on the edge of its neighbour's paper.
The sim's default layout has 10-45 cm between them. Laying out the real
bench like the sim is planned next, checked with
`python scripts/check_bench_layout.py`, which reports each object's offset
from `default_bench` in cm, with a hint for which way to move it.

Offline: `test_skills_offline.py` 289 passed, the 2 old scoop failures;
`test_agents_offline.py` 74/74. All new checks also pass under the robot's
Python 3.8.

**Run 3: cups spread out by the user (`scripts/check_bench_layout.py`,
2026-10-07).** Only the cups moved; the dishes, scoop and stirrer stayed put.
- **Nothing was misidentified.** The duplicate rule dropped cams 3/4/5 for A
  and for two dishes instead of letting them guess.
- Found by how many cameras:
  - stirrer 4, red cabbage 4, citric acid 3;
  - baking soda 2, B 2, beaker 2;
  - **A 1, scoop 1**;
  - **C not found**.
- **C failed on depth, not on its label.** Cams 2 and 4 both read
  "C EMPTY". Their 3D centroids were (0.338, 0.203) and (0.273, 0.158),
  80 mm apart against a 50 mm consensus tolerance.
  - Depth passes through a clear cup's wall, so each camera's points are
    biased along its own line of sight. Cams on opposite sides are biased
    opposite ways.
  - B and the beaker each lost cameras to the same disagreement.
  - This also matters for motion: dump and pour aim at the rim centre fitted
    to these points.
- **Measured positions against the sim's layout rules (`layout_violations`):
  11 violations.**
  - All three dishes are 0.60-0.66 m from the base, past `LAYOUT_REACH`'s
    0.56 m.
  - The beaker is at 0.351 m, under 0.38 m for a 95 mm container, and in the
    way of a dump into C.
  - The holder is 0.110 m from the red cabbage dish (needs 0.12 m, and
    0.18 m for scooping there), and in the way of a dump into B.
  - B is 0.092 m from the scoop (needs 0.112 m). The citric acid and baking
    soda dishes are 0.128 m apart (need 0.132 m).
  
  Positions used for this: citric (0.65, -0.13), baking soda (0.57, -0.23),
  red cabbage (0.57, 0.17), A (0.28, -0.21), B (0.47, -0.09), C (0.305,
  0.18, mean of the two cams), beaker (0.35, 0.03), scoop (0.54, -0.03),
  stirrer and holder (0.58, 0.06).

**The real bench is not laid out like the sim, on purpose.** The user arranges
it by hand and moves things between runs. Do not ask for the sim's default
coordinates; the user rejected that on 2026-10-07.
`scripts/check_bench_layout.py` checks the layout as it stands:
- what was found, and by how many cameras;
- the sim's `layout_violations` applied to the found positions.

The robot venv cannot import `robochem.sim` (it needs mujoco), so the script
loads `bench.py` by path.

**Run 4, the user's own layout (2026-10-07).**
- **All 9 located objects were found and none was misidentified.**
  - stirrer 4 cams, baking soda 4, red cabbage 4;
  - scoop 3;
  - A, B, C and the beaker 2 each;
  - **citric acid 1**, at 0.68 m.
  
  Three cameras were dropped as duplicates. C was (0.27, 0.17).
- 9 rule violations remain:
  - citric acid is 0.684 m from the base;
  - B at 0.300 m and C at 0.320 m are inside 0.35 m, and C's x is 0.270
    (dump needs 0.28; C is only poured into);
  - red cabbage is 0.108 m from the holder (scooping beside it needs 0.18);
  - the beaker is in the dump corridor of B;
  - the holder is in the dump corridor of the beaker.
  
  These are reported to the user as risks; the user decides.

**Real agent dry run with the bench: passed (2026-10-07,
`experiments/agent_run_20261007_180619.json`).**
- The scene listed all 10 bench objects.
- The plan was 31 sub-tasks and all 31 resolved to valid skill calls, with
  no replan.
- The plan is the sim's 31 step for step (`agent_run_20261007_165912`). The
  one difference is that step 15 picks the larger spoon for the red cabbage,
  as intended.
- One call carried a note in place of a number: dump into B got
  `tool_offset: "[measured offset from pick_up of larger spoon]"`.
  `coerce_params` never checked `list` params. In a live run the
  executor's `float(given[0])` would have raised outside the skill's error
  handling. `_as_kind` now requires `[x, y, z]` of numbers for a `list` and
  drops anything else, so the CAD offset applies.
  `test_agents_offline.py` 77/77 in both venvs.

**Clear cups are placed by their silhouettes, and every labelled container
is identified by a vote across cameras (2026-10-07).** Only localisation has
run on the cell, still no motion.
- **Depth cannot place a clear cup.** The points of all four cups sat on the
  table, at z -2 to -11 mm for cups 47 mm tall, and leaned away from each
  camera. On one cup, the depth centroids of cams 2/3/4/5 spread 6 cm.
  - Label-free experiment: the ray through each mask's pixel centroid,
    triangulated across cameras.
  - Result: all four cups placed by 3-4 cams, rays within 6-11 mm of the
    point, at 22-31 mm up (half-height: right for cups 47 mm tall and a
    taller beaker).
  - Scratchpad script, not kept: `raycast_test.py`.
- **`VisionSystem._place_rays`.**
  - It takes the largest set of 2+ cameras whose rays meet within 25 mm, at
    a height inside the container. Rays that cross only in mid-air are
    refused: "A" once triangulated 157 mm above the table from cams 2 and 5,
    which were on two different cups.
  - A lone ray is cut at half height.
  - A mask touching the frame edge gives no ray. A at the bottom edge of
    cam 2 pulled a 4-cam fix 46 mm off, with its rays still within 20 mm.
  - The cup handed to the skills is `cylinder_points` of the manifest's
    radius and height. `locate_container` fits its rim, top and base
    exactly (offline test).
- **`_locate_by_vote` covers every bench container that has a label.**
  1. It places every instance of the category: clear cups by silhouette,
     dishes by depth centroid (`group_spots`).
  2. It reads every instance with `LabelResolver.read_all`.
  3. `assign_labels` shares the labels out one container each, to agree
     with the most reads. A tie for the requested label refuses.
  4. `eliminate`: when no camera read a label, it goes to the one container
     left. Every other label must be settled on a container placed by 2+
     cams, and the number of such containers must equal the number of
     labels.

  The per-camera path stays for names that are not on the bench.
- **Manifest sizes.** A and B are measured: 55.5 mm across, 47.3 mm tall.
  **C and the beaker are NOT measured.** C takes A's size; the beaker is
  44 x 60 mm, from triangulated height and the frames. Measure both.

Run 6 on the user's layout (`check_bench_layout.py`):
- The three dishes were found by **4 cams each**. In run 5, per-camera
  matching found no citric acid at all: cam 4 read the baking soda dish as
  citric acid, 155 mm from cam 2's.
- Clear cups and beaker by silhouette and vote:
  - **A**: 3 cams (cam 2 cut off), 23 mm up, at (0.305, -0.208);
  - **B**: 4 cams, at (0.302, -0.057), only 1 of 4 cams reading "B";
  - **beaker**: 4 cams, 4 of 4 reads.
- **C was not found.** It was placed by cams 2 and 3, but both read it as
  "A" and none read "C EMPTY". Elimination was added after this run and is
  not yet run on the cell.
- **The reader leans to "A 10 ML WATER".** Cam 3 read 3 of 4 cups as A.
- **The C label is the cause of the C failure.**
  - The writing is small and thin, and written sideways.
  - The wide, low C cup stands on it: cam 4 sees "EMPTY" only through the
    cup's floor, and no "C".
  - Advice given: write big and bold, keep the writing clear of the cup,
    and write it again the other way up for the cameras on the far side.
- `test_skills_offline.py` 315 passed, the 2 old scoop failures. The new
  checks pass under Python 3.8. `test_agents_offline.py` 77/77.

**Run 7, after the user rewrote the C label bigger and clear of the cup:
all 9 located objects found, none misidentified (2026-10-07).**
- Dishes: 4 cams each, reads 2/4, 3/4 and 3/4.
- A: 3 cams. B: 4 cams. Beaker: 4 cams. Reads 3/3, 2/4 and 3/4.
- **C: 3 cams, and 2 of 3 read "C EMPTY"**, so no elimination was needed.
- Silhouette rays came within 9-11 mm on every clear container.
- Scoop and stirrer: 3 cams each.
- **Cam 3 read all four clear cups as "A 10 ML WATER"**, so its reads carry
  nothing. The vote outweighs it; worth finding out why if reads get thin.
- The user confirmed A, B and C are one size. The beaker is still unmeasured.
- `check_bench_layout.py` now checks the rules with the manifest's sizes
  where it has them. The sim's beaker is 95 x 70 mm; the real one is a
  narrow cup. 4 risks remain:
  - the holder is in the dump corridor of the beaker (+0.274 ahead, 0.030
    aside);
  - citric acid dish to scoop is 0.120 m (needs 0.136);
  - the holder is 0.157 m from the red cabbage dish (needs 0.18 for
    scooping there);
  - B is 0.307 m from the base. That is inside the random-layout bound of
    0.35, but outside the measured dump limit of x 0.28.
- A second real agent dry run on this layout
  (`experiments/agent_run_20261007_193025.json`) gave the same 31 calls as
  `..._180619`, with no prose values.
- **Before motion:** the planner sends `pick_up` only `object_name`. The
  scoop's robot-validated pick (`forward_offset -0.010`, `z_offset 0`,
  `grasp_force 1.0`, which keeps the jaws off the crank) reached no agent
  call; the `pick_up` defaults are 0 and 1.5 N. **Fixed the same day:**
  - `BenchObject.pick` holds the scoop's and the stirrer's validated grasps;
  - `SkillsExecutor._bench_defaults` fills in any key a `pick_up` call
    leaves out.

  Clear cups and the beaker have none, because they have never been picked
  on the robot.

**First motion on the real arm through the bench manifest (2026-10-07):
scoop pick and put-back, both succeeded.**
- The control stack: Franka Desk unlocked, FCI active and the lights **blue**,
  **then** `bash ~/frankapy/bash_scripts/start_control_pc.sh -u franka -i
  franka-Alienware-Area-51-R5`.
  - White lights mean the e-stop is down. A stack started under white
    lights gave `FrankaInterface status not ready for 10s`. Check with
    `rostopic echo -n 1
    /franka_interface_status_publisher_node_1/franka_interface_status`.
    `robot_state.last_motion_errors.cartesian_reflex: True` was left over
    from an earlier session and did not block anything.
- Command: `--skill pick_up --params
  '{"object_name":"larger spoon","z_offset":0.0,"grasp_force":1.0,"forward_offset":-0.010}'
  --skill place --params '{"target_location":"larger spoon"}'`.
- What it measured:
  - "white plastic tool" was found by **4 cams**, the 4 centroids within
    43 mm of each other, confidence 0.94.
  - Centroid (0.606, 0.015, -0.001). The jaws closed 6.9 mm from the
    commanded grasp, and the lift arrived within 11.6 mm.
  - Measured tool offset [0.053, 0.009, 0.013] against [0.051, 0.009] on
    2026-09-25: the same grasp.
- Place went back to the pick site and released 20 mm above the table, as
  designed; the user confirms it was set down, not dropped. The descent move
  prints no line of its own (`place.py` `goto_pose_rigid(target_pos)`).
- The width read 23.1 mm right after the close (`gripper_is_grasped=False`)
  and 8.1 mm at the start of place: the handle. The first reading is taken
  while the jaws are still closing.

**`run_experiment.py --step` (2026-10-07).** Before every skill of a
`--task` or `--instruction-image` run, it prints the call and waits for
Enter; `q` stops the run there with outcome "blocked", without a replan
(`AgentOrchestrator(confirm=...)`).
- For a run with empty containers, `--no-verify` is required. Hardware
  verifies by default, and a colour check on nothing fails and replans,
  which repeats the motions.
- `test_skills_offline.py` 320 passed, the 2 old scoop failures.
  `test_agents_offline.py` 81/81 in both venvs.

**First real pipeline run (2026-10-07,
`experiments/agent_run_20261007_195406.json`).** Flags: `--no-verify --step
--max-retries 0`. The plan was 33 sub-tasks: this time the beaker was picked
twice, once for each pour.
1. **pick_up scoop: ok.** The bench defaults were filled in. 3 cams, cam 5
   dropped at a score of 0.43. The jaws held the handle at 8.0 mm, and the
   measured offset was [0.055, -0.007, 0.018].
2. **scoop citric acid: reported ok, but the stroke was sized wrong.**
   - The vote placed the dish with 4 cams, at (0.510, -0.093).
   - The rim fit was rejected (r 45 mm, 15% inliers). The median fallback
     read the opening as **61 mm**, where the real inside is 45.5 mm.
   - The stroke ran -45..+39 mm from the centre with a 53 mm envelope,
     past the real wall.
   - Fixed: `BenchObject.scoop` gives the dishes `container_radius
     0.0455`, which `_bench_defaults` fills in for `scoop` calls.
3. **dump into A: FAILED.** "Bowl stopped 171mm short of the cup"; the carry
   was 211 mm. `robot_state.last_motion_errors` showed
   **`joint_position_limits_violation: True`**.
   - A was at (0.305, -0.207), placed by silhouette with 3 cams within
     10 mm. That is 0.368 m from the base, 34 deg to the right, and passes
     every sim layout rule.
   - The sim's IK (`SimFrankaArm`) clips to the same vendor limits, but it
     pulls toward HOME in the nullspace. The real Cartesian controller
     resolves the redundancy its own way, so a straight-ahead carry close in
     and to the side can be fine in sim and hit a limit on the robot.
   - **The sim layout rules are not enough for real dump targets.** The one
     real dump that worked was into B at about (0.41, 0.04) on 2026-09-25.
   - Not yet tried: `dump_heading_deg`, or moving A out to about 0.38-0.45
     m and |y| <= 0.15.
   - After the failure the arm went home **still holding the loaded scoop**.
     The next run's `pick_up` opens the gripper first, which drops anything
     held from home height.

**Second real pipeline run (2026-10-08,
`experiments/agent_run_20261008_133157.json`). The cups were moved by the
user.**
- pick_up: ok. The scoop was at (0.599, 0.055); held at 8.0 mm; offset
  [0.051, 0.011, 0.016].
- **scoop: ok, with `dish_radius 0.0455` from `container_radius`.** The
  stroke shrank to -29..+22 mm with a 37.5 mm envelope, inside the wall. The
  rim fallback still read 59 mm, but no longer decides anything.
- **dump into A: failed, "off by 22mm"**, against 171 mm the run before.
  `last_motion_errors` showed
  **`cartesian_motion_generator_joint_velocity_discontinuity: True`**.
  The step's log lines were not captured, so where A stood is not recorded.
- **Cause, as far as the code shows.** frankapy dynamic mode is
  `PassThroughPoseTrajectoryGenerator` under cartesian impedance: setpoints
  go to the arm unsmoothed.
  - `stream_pose_path` resampled the waypoints *evenly*, so the carry
    started at full speed from rest and stopped dead.
  - The speed also jumped wherever a long segment met a short one, since
    time was shared per waypoint.
  - The impedance-tracked arm trails the setpoints and stays where it got
    to when the skill ends.
  - So "22 mm short" is the trailing distance at a dead stop. Which motion
    raised the velocity-discontinuity flag is not certain.
- **Fixed (offline-tested only, not yet on the robot):**
  - `resample_path_eased`: min-jerk timing, starting and ending at rest,
    shared out by arc length (a turn counts 0.1 m/rad). Peak speed is
    1.875x the old constant.
  - The last setpoint is held for `STREAM_SETTLE_SECONDS` = 0.5 s before
    termination.
  
  This covers the dump carry, tip and shake, and the scoop sweep. Stir's
  circles are not resampled; they only get the hold. The sim's
  `follow_pose_path` bypasses all of it, so sim cannot check this change.
  `test_skills_offline.py` 328 passed, the 2 old scoop failures.

**The real scoop moved differently from the sim's. Two causes, both fixed
(2026-10-08).**
- **"It digs too far forward".** The executor placed the bowl from the
  middle of the pick cloud's two ends (`_offset_from_tip`), which was tuned
  in sim. In sim the cloud ran 64 mm against the part's 69.5.
  - **On the robot it ran 111-113 mm**, with 45 mm of it behind the handle;
    spilled powder is the likely cause.
  - The middle put the bowl 26-35 mm from the jaws. `pick_up` measured
    51-55 mm, and 2026-09-25 validated 51 mm.
  - So every scoop pose was 2-3 cm further forward than planned.
  - Fix: `tool_geometry` now carries the part `length`. When the cloud
    differs by more than 15 mm, the ends are not used: the measured x/y are
    kept, with the CAD z. When the call passes no offset and the measurement
    is more than 15 mm from the CAD x, the measured x/y are used too.
  - A cloud of the part's length (sim) still uses the ends, so sim behaviour
    is unchanged.
  - This compensates for where the jaws closed. It does not move the grasp;
    the contaminated cloud still pulls the grasp about 25 mm back along the
    handle. Clean the bench around the scoop.
- **"It comes back too fast".** Sim and robot used different speed profiles.
  - The sim's `goto_pose` and `goto_joints` ran at constant speed with a
    dead stop. frankapy's are min-jerk: over the same duration the middle
    runs 1.875x faster. Home was 5 s min-jerk on the robot against 4 s
    constant speed in sim, 1.5x faster at its peak.
  - Streamed paths were the other way round: min-jerk in sim, constant
    speed on the robot.
  - Fix:
    - `robochem/skills/pacing.py`: `PacedArm` wraps **both** arms in
      `run_experiment.py`. It stretches every point-to-point duration by
      `MOTION_TIME_SCALE` 1.875, so the min-jerk peak equals the old sim's
      constant speed. Home takes `HOME_SECONDS` 7.5.
    - `SimFrankaArm.goto_pose` and `goto_joints` are now min-jerk (`_min_jerk`).
    - `resample_path_eased` reproduces `follow_pose_path`'s timing exactly:
      min-jerk over the waypoint index, equal time per segment. The
      arc-length version of an hour earlier did not match the sim and was
      replaced.
  - Every move now takes longer, on both. A full sim run will take about
    1.9x as long as the 20 min of 2026-10-07.
- Checks:
  - `smoke_test_sim.py`: all 10 groups pass with min-jerk sim moves.
  - `test_skills_offline.py`: 340 passed, the 2 old scoop failures.
  - Not yet run on the robot.

**Third real pipeline run: step 1, the arm did not move (2026-10-08,
`experiments/agent_run_20261008_141113.json`). Cause not yet known.**
- The scoop was at (0.660, 0.047). The hover to (0.650, 0.047, 0.30)
  "ended" at home: "Pose not reached ... err=297 mm". `last_motion_errors`
  read `joint_position_limits_violation`.
- **It is not reach.** I first blamed the distance, since earlier picks were
  at 0.599-0.619 m, and added a 0.62 m tool-reach rule. Both were wrong and
  the rule was taken out again.
  - IK on the sim's arm model (vendor joint limits, tool straight down) holds
    (0.650, 0.047) at z 0.30 and 0.01 with every joint at least 0.97 rad
    inside its limit. It still does with joint 3 locked at 0 as libfranka's
    cartesian control keeps the elbow.
  - **Top-down reach (hover 0.30 m and grasp 0.01 m, every joint ≥ 0.10
    rad inside its limit) is a ring about 0.30-0.78 m from the base** in
    front, and out to x 0.66 at y ±0.40. Scratchpad `reach_map.py`, not
    kept.
- **Likely: commands were being refused.**
  - `move_to_pose` calls frankapy `goto_pose` with its default
    `ignore_errors=True`, so a skill the robot refuses returns silently
    and only the position check notices. That matches an arm that never
    left home.
  - The error flag may be left over from an earlier motion.
  - Untested: re-run the plain pick/place test where the scoop stands, and
    read the franka-interface terminal on the control PC at the failure.

**The run order on 2026-10-08, from `experiments/`, which explains "the scoop
still moves the same".**
- **`agent_run_20261008_135732`** started about 13:40 and loaded the code
  of that moment.
  - The executor's tool-offset fix landed at 13:55 and pacing at 14:00, so
    this run had neither.
  - Its scoops used `tool_offset` x 29.8 and 31.1 mm, the old cloud-middle
    value, where `pick_up` measured 53-54 mm and the extents were 110-112 mm.
  - **This is the scoop motion the user judged "still the same".**
- That run did get furthest so far, **steps 1-9 ok**:
  - scoop pick, citric acid scoop, **dump into A**, scoop back;
  - stirrer pick, stir A, stirrer back into its holder;
  - scoop pick, baking soda scoop.
- Step 10, dump into B, failed: "off by 168mm". Its `[Dump]` lines were not
  captured.
- **`141113` and `141504` are the first runs with the fixes.** Both stopped
  at step 1 with the arm never leaving home, so the fixed scoop has not yet
  run on the robot.
- Both came straight after the step-10 failure. Suspect a refused-command
  state that ignore_errors hides, and restart the control stack before
  blaming the code.
- `move_to_pose` now says so explicitly. When the arm moved under 5 mm and
  missed by more than 50 mm it prints "The arm did not move at all ... the
  robot refused the motion", instead of leaving it to read as "unreachable".

**Scoop-only test with all the fixes in, after a stack restart (2026-10-08,
`--skill` chain, so no run record): it still dug too far forward.**
- The arm moved normally again after the restart.
- Scoop at (0.500, 0.108). The cloud was 84 mm long against 69.5 (cleaner
  than before), with 2 of 4 cams.
- `pick_up` measured x **36 mm**. The executor still used the ends'
  **21.6 mm**: the cloud was 14.5 mm too long, just under the 15 mm length
  gate.
- The jaws closed **4.9 mm behind** the commanded grasp in x. Earlier picks:
  4.7 and 6.3 mm.
- Two errors, both making the bowl look nearer the jaws than it is:
  1. **Offsets were measured from the commanded grasp, not where the jaws
     closed.** The robot stops 5-6 mm short in x every time; the sim does
     not. `pick_up` now measures `suggested_tool_offset` and
     `tool_extent_x` from `held_frame`: the commanded rotation at
     `grasp_tcp_reached`.
  2. **The ends were trusted when they disagreed with the bowl-end median.**
     In sim the two agree within 8-10 mm (6.2, 8.1, 7.0, and 9.6 / 8.2 from
     the run records). On the robot they were 14-29 mm apart every time,
     the ends always the nearer. `_offset_from_tip` now drops the ends when
     they are more than **11 mm** from `pick_up`'s measured x, and the
     measured x/y is used with the CAD z.
  
  The sim margin is thin: 9.6 against 11.
- Estimated true offset for this run: about 36 + 5 = 41 mm, where 21.6 mm
  was used. The bowl was about 19 mm further forward than planned; the
  stroke's front edge would reach about 55 mm against a 45.5 mm wall.
- **Still unknown: how close the measured bowl-end median is on the robot.**
  In sim it read 7.7 mm long. Get a ruler measurement, with the scoop held,
  of jaws centre to bowl centre along the gripper's forward axis, and
  compare it with the printed "Measured tool offset".
- `test_skills_offline.py` 343 passed, the 2 old scoop failures.

**Ruler check of the measured tool offset (2026-10-08): it is right.** With
the scoop held after a `pick_up`, `pick_up` printed `[0.053, -0.011, 0.026]`
(measured from where the jaws closed). The user's ruler, jaws centre to bowl
centre, read **52.9 mm**.
- `MAGIC_BEAKER` therefore has `measured_tool_offset=True`.
- `VisionSystem.trust_measured_tool_offset()` reports it.
- The executor then uses the measured x/y with the CAD z for any scoop or
  dump call that passes no offset or passes pick_up's copy. An offset passed
  on purpose is still kept.
- The sim has no bench, so it keeps the cloud-ends logic its own picks
  validated.

**Scoop, sim against robot, input by input (2026-10-08).** The skill is one
file, `robochem/skills/scoop.py`, for both. What differed was what it was
given and how the arm ran it.
- The sim log is step 2 of `sim_full_run`. The robot log is the user's
  scoop-only test above.

| | sim | robot before | now |
| --- | --- | --- | --- |
| tool offset x | 19.6 mm (ends; right in sim) | 21.6 mm (ends; true 41-53) | measured, ruler-checked |
| dish inside radius | 45 mm, circle fit | 45.5, `container_radius` | same |
| inside floor | `table_z` 0 + 8 mm (cell config) | base_z -0.1 mm + 8 mm | **`table_z` 0.0 now in the real config** (`REAL_TABLE_Z`, `run_experiment.py`) |
| rim height | 31.6 mm read | **16.7 mm read** (real 29) | **`container_height` 0.029** from the manifest |
| stroke | -28..+21 mm | -29..+22 mm | same |
| timing | min-jerk, 5 s | min-jerk, 5 s (+0.5 s hold) | same |
| sweep end, tool long axis | 90.0 deg | **71.1 deg** | to re-check |

- **`container_height` is a new optional scoop param.**
  - When given, the rim is `table_z` (or base_z) plus the height, rather
    than the cloud's top.
  - It sets the handle's clearance over the rim during the sweep and the
    "lift straight out" height.
  - The sim never passes it, so the sim is unchanged; offline test
    `test_scoop_takes_a_known_rim_height`.
- **The 19 deg roll shortfall at the sweep end is probably a symptom, not a
  stiffness problem.** With the bowl actually ~19 mm further forward than
  planned, its front edge (about 55 mm out) met the 45.5 mm wall and stopped
  the roll. Re-check "finishing at +0 deg (long axis ...)" with the right
  offset before touching frankapy's dynamic-mode impedance, which defaults
  to [600]*3 N/m and [50]*3 Nm/rad.
- `test_skills_offline.py` 349 passed, the 2 old scoop failures.

**Next scoop on the robot: "the descent is not deep enough, too high above"
(2026-10-08; no log captured).**
- **It is not the plan.** I replayed scoop offline with the robot's numbers:
  offset 53 mm, `container_height` 29 mm, `table_z` 0, back reach 30
  against 2.8 mm. In every case the dig plans identically to the sim:
  descent to z 0.0229, head 3 mm over the inside floor, `rim_lift_max` 0.
  Only the lift-out height changed.
  - The handle-reach suspicion was wrong for the dig. Holding at 53 mm
    leaves only about 2.8 mm of handle behind the jaws, against the CAD's
    30, but `rim_lift_max` stayed 0 either way.
- **It is execution.** Scoop's descent (`goto_pose_rigid(...,
  use_impedance=True)`) and the streamed sweep (frankapy dynamic mode,
  cartesian impedance, [600]*3 N/m) both stop short. Same signature as
  every robot move under impedance: the pick closed 6.8 mm above its
  command, and lifts arrived 11 mm short. Powder resistance adds to it.
  - The descent's arrival check uses `contact_tol` 0.05 m, so 10 mm high
    passes silently.
  - Estimate: the bowl rode 7-10 mm high, so its tip sat 10-13 mm over the
    floor of an 8.8 mm bed.
  - The sim's servos track the command and its powder has no resistance.
  - Earlier robot scoops hid this: the bowl was 19 mm too far forward,
    against the wall.
- **Fix (offline-tested; not yet on the robot).** After the compliant
  descent scoop reads where the TCP stopped. If it is more than 2 mm above
  the command:
  - the descent is re-commanded that much lower, at most
    `sag_compensation_max` 0.010;
  - every sweep setpoint is shifted down by the same amount.
  
  It stays under impedance, so the bowl is still compliant against a floor.
  The result reports `descent_sag` and `sag_compensation`. The sweep's end
  now prints "TCP ±x mm from the last setpoint".
- Checks:
  - Sim smoke "pick + scoop", "pick + scoop + dump" and "scoop then stir"
    all pass, with no correction triggered; the sweep ended +0.1 mm from
    its setpoint at 90.0 deg.
  - `test_skills_offline.py` 354 passed, the 2 old scoop failures.
  - Next robot run: read the "compliant descent stopped", "now ... from the
    planned height" and "from the last setpoint" lines.

**Next scoop on the robot (2026-10-08): the bowl hit the far wall and pushed
the citric acid dish at least 10 cm across the bench.**
- Inputs were right and matched the sim:
  - offset 40.8/-5.2 mm, measured;
  - floor `table_z` + 8 mm; rim from `container_height`;
  - stroke -29..+22 mm, the sim's.
- The descent stopped 3.9 mm high; after compensation it was +0.7 mm.
- The sweep ended with **long axis 71.3 deg** (level is 90) and the
  **TCP 13.4 mm below its last setpoint**. The run before it also showed
  71.1 deg, so the 19 deg was not caused by the bad offset, as I had
  guessed. The sim ends at 90.0 deg and +0.1 mm.
- **The commands do not go there.** Replaying scoop offline with the
  robot's numbers, the commanded TCP moves only +6 mm in x over the whole
  sweep (0.5854 -> 0.5910) and the bowl ends 22 mm past the dish centre.
  - A roll that lags (still nose-down) leaves the bowl *behind* plan, about
    12 mm at 19 deg, not ahead.
  - So the dish was moved by the arm leaving its commanded path.
  - Suspect: the TCP sinking (13.4 mm is about the hand's ~7 N weight on
    frankapy's 600 N/m, if the end-effector load is not fully
    compensated), which put the bowl into the floor or the wall while the
    impedance pulled it along. **Not established.**
- Changes (offline-tested; not yet on the robot):
  - `STREAM_TRACKING_IMPEDANCES` [1500]*3 N/m + [150]*3 Nm/rad, passed as
    frankapy's `cartesian_impedances` by scoop's sweep and dump's
    carry/tip (`stream_pose_path(stiffness=...)`,
    `dynamic_start_kwargs`). frankapy's defaults are [600]*3 + [50]*3.
  - `stream_pose_path` samples the arm every 5 ticks and prints "tracking:
    worst X mm / Y deg behind the setpoints ... at the end ... (dx dy
    dz)". Robot path only.
  - Scoop's sweep-end line now prints the TCP error in x, y and z.
  - **New scoop param `air_offset`**: plans the dish that much higher, so
    the same scoop runs in the air touching nothing (offline test: every
    dig command +100 mm, same x/y). Use it on the robot before a dish is
    put at risk again.
- `test_skills_offline.py` 361 passed, the 2 old scoop failures.

**FAILED, reverted: stiffer streaming made the arm shake heavily
(2026-10-08).**
- The air rehearsal (`air_offset` 0.10) ran with two changes that had never
  been on the robot:
  1. `STREAM_TRACKING_IMPEDANCES` [1500]*3 N/m + [150]*3 Nm/rad as frankapy
     `cartesian_impedances` for scoop's sweep and dump's carry/tip;
  2. a `get_current_pose()` (a ROS service call) every 5th tick inside the
     50 Hz setpoint loop.
- The user: "completely go wrong! shake heavily!".
- frankapy dynamic mode is `PassThroughPoseTrajectoryGenerator`, so 50 Hz
  setpoints reach the controller as steps. A stiffer spring answers each
  step harder, and a blocking read in the loop makes the steps uneven.
  Which of the two caused it, or both, was not separated.
- **Reverted both.** No skill passes a stiffness; the constant stays only as
  a warning. Nothing runs between setpoints. The end-of-path error is read
  once, after the skill has ended.
- The run before, with the same min-jerk timing and 0.5 s hold, frankapy's
  default stiffness and no in-loop read, streamed 275 setpoints without
  shaking.
- **Lesson:** do not change impedance on the robot blind. Measure tracking
  first, at frankapy's defaults (the air rehearsal now prints the
  end-of-path error). Check the end-effector mass in Franka Desk before
  suspecting gravity sag.
- `air_offset` itself only changes the plan, not the controller, and stays.

**Air rehearsal at frankapy's default stiffness (2026-10-08,
`air_offset` 0.10). The user: "the motion is correct this time".** It
touched nothing, so these are pure tracking errors:
- The descent stopped 2.2 mm high, and +0.7 mm after compensation.
- Sweep end: long axis **86.9 deg** (3 deg short; the wrist tracks in
  free air).
- **End of path 12.1 mm / 3.5 deg from the last setpoint: dx +11.9,
  dy +0.7, dz -2.3 mm.** That is after the 0.5 s hold, so it is a static
  error, not lag.
- **This explains the dish pushed 10 cm.**
  - The planned stroke leaves the bowl's front edge 8 mm off the far wall;
    12 mm of forward overshoot puts it about 4 mm into the wall.
  - Cartesian impedance then keeps pushing, about 7 N at 600 N/m.
  - The contact explains the dish run's other numbers too: 19 deg of roll
    lag and the TCP 13.4 mm low.
- A static error of 12 mm in free air means an unmodelled steady force of
  about 7 N. Its direction depends on the arm's configuration: dz on the
  dish run, dx here, and lifts that always arrive about 11 mm short.
- **Ruled out: the end-effector load configured in Franka Desk.** The user
  checked: it is Franka Hand. The static error is the cartesian impedance
  controller's own behaviour on this arm, at frankapy's default stiffness;
  joint friction against a soft spring is the usual cause. It is not a
  configuration mistake.
- Fallback if Desk is right: run the sweep position-controlled (blocking
  `goto_pose(use_impedance=False)` waypoints). It is accurate but moves in
  steps, unlike the sim. Ask the user first.

**Scoop `sweep_mode` (2026-10-08). The user chose position control.**
- **`"stream"`** (the default) is unchanged and is what the sim runs:
  - smoke "pick + scoop" and "pick + scoop + dump" pass;
  - the sweep ends +0.1 mm and 90.0 deg.
- **`"position"`** runs the sweep's 12 planned waypoints as blocking
  `goto_pose_rigid(..., use_impedance=False)`. That is frankapy's
  CartesianPoseSkill, the controller pick, place and the lifts use.
  - Same geometry as the stream; it pauses briefly between waypoints.
  - It does not inherit the descent's impedance-sag correction, since
    position control goes where it is sent.
- The real bench's dishes carry `("sweep_mode", "position")` in
  `MAGIC_BEAKER`, filled in by `_bench_defaults`, so every robot scoop
  uses it and the sim never does.
- Dump's carry and tip still stream under impedance; they will likely need
  the same treatment.
- `test_skills_offline.py` 365 passed, the 2 old scoop failures.
- Not yet on the robot. Air-rehearse first.

**Position sweep on the robot (2026-10-08).**
- Air rehearsal: the user judged it "good". Its log was not captured.
- In the dish: "not low enough, still too high". The dish was not pushed.
- A ruler, scoop held level: fingertip centre to the bowl's lowest point
  **31.5 mm**; the CAD says 29.4.
  - By that number the bowl already ran about 1 mm over the floor (8 mm
    base), not the planned 3 mm.
  - I proposed using 31.5 and adding powder: the sim's bed is 23.6 mm,
    where the robot dish read 8.1 mm and the bowl's mouth stayed 6.4 mm
    above the surface.
  - **The user rejected that: the powder is enough, scoop lower.** The 31.5
    is recorded here, not used.
- **New scoop param `z_offset`** (metres, + up / - down) moves the whole
  dig: descent, sweep and lift-out. It shares `air_offset`'s mechanism,
  planning the dish that much higher or lower.
  - The real bench's dishes carry **`DISH_SCOOP_Z_OFFSET` -0.003** in
    `bench_manifest.py`. A run can override it with `"z_offset"` in scoop's
    `--params`.
  - Offline test: every dig command exactly 3 mm lower, same x/y.
  - `test_skills_offline.py` 366 passed, the 2 old scoop failures.
- If it goes lower still, watch for the bowl meeting the dish floor; the
  ruler says it is already close. Go down in small steps.

**Correction, same day.**
- The edit that put the ruler's 31.5 mm into the code
  (`BenchManifest.tool_depths`, `VisionSystem.bench_tool_depth`) **had run**
  although the user rejected it; I did not check the files afterwards.
- So the first `z_offset` -3 mm robot run used z 31.5. The log line read
  "z 31.5mm by ruler", and the net change was only about 0.9 mm lower.
- **Now fully reverted.** The executor uses the CAD 29.4 for z, so the next
  run digs 2.1 mm deeper than that one.
- That run is still useful: **the position-controlled sweep tracked
  essentially as the sim does.**
  - Sweep end: TCP -0.4 mm from its last setpoint (x -0.3, y +0.2), long
    axis **89.9 deg**. The sim gives +0.1 mm and 90.0 deg.
  - The streamed sweep in the air had given +11.9 mm.
  - The descent stopped 2.8 mm high and was +1.5 mm after compensation.
- **`DISH_SCOOP_Z_OFFSET` is now -0.015, the user's call.**
  - By the plan, -3 mm already put the bowl's lowest point at about the
    dish's inside floor (z 0.008 on an 8 mm base). I said so, and asked
    whether -0.025 (the user's first figure) meant 25 or 2.5 mm; the user
    chose 15 mm down.
  - **Run on the robot: the user calls -15 mm "perfect"** (2026-10-08). The
    citric acid scoop ran with position sweep, measured x/y, CAD z 29.4 and
    `z_offset` -0.015. **This is the validated real scoop.**

**Whole-pipeline test, first dump (2026-10-08): "the dumping is too
forward".**
- The user asked for a default 10 mm back. **`DUMP_FORWARD_OFFSET` is
  -0.010** in `bench_manifest.py`. It is dump's existing `forward_offset`:
  base frame, + away from the arm.
- It is applied through the new `BenchManifest.skill_defaults`
  (`{"dump": {...}}`, for every dump on the bench, whatever the target).
  `_bench_defaults` now also covers dump, keyed on `target_container`.
- An offset a call passes is kept. Pour is not touched. The sim has no
  bench, so it is unchanged.
- Dump's carry and tip still stream under cartesian impedance. The scoop's
  stream overshot forward 11.9 mm in free air, and this offset compensates
  the same kind of error.
- `test_skills_offline.py` 371 passed, the 2 old scoop failures.
  `test_agents_offline.py` 81/81.

**The user's read: "the position of each item is not so accurate, maybe the
point cloud" (2026-10-08).** The logs agree. The scoop's -15 mm and the
dump's -10 mm are likely compensating the same thing.
- **Opaque dish centres lean toward each camera.** The citric acid dish's
  per-camera depth centroids, x: cam 2 0.5714, cam 3 0.5481, cam 4 0.5427,
  cam 5 0.5707. That is 29 mm of spread, cams 2/5 against 3/4, from
  opposite sides of the cage.
- The dish rim circle fit is rejected on nearly every robot scan (13-18%
  inliers), so the rim centre comes from the median fallback, which leans
  toward the cameras.
- Clear-cup silhouette rays met within 6-11 mm, which bounds how well the
  four extrinsics agree. Progress says ~3-6 mm held-out from the cube
  calibration.
- **`scripts/point_at.py` (new, not yet run).**
  - It locates each named object as the skills do: `locate_container`'s rim
    centre, and the bench's known height for clearance.
  - It parks the closed fingertips straight down 5 cm over that point,
    position-controlled via `SAFE_Z` 0.25 m, and waits for Enter while the
    user reads the real offset.
  - Several objects across the bench separate one global shift (fix the
    calibration or apply one correction, and drop the per-skill offsets)
    from per-object error (fix the centre estimate: a known-radius fit, or
    silhouettes for the dishes too).

**Pipeline test, stir (2026-10-08): "the stir motion is too little ... like
a circle in the bowl".**
- The skill's 15 mm circle was clamped to 14.75 mm: the 26.75 mm opening of
  the clear cups minus `wall_clearance` 12 mm.
- The circle is streamed under cartesian impedance, which shrinks a small
  circle further; the scoop's stream showed about 1 cm of error.
- **Bench defaults for stir: `STIR_RADIUS` 0.020, `STIR_WALL_CLEARANCE`
  0.006** (`skill_defaults["stir"]`; `_bench_defaults` now keys stir on
  `target_container` too).
  - With the 5 mm-radius rod, the rod's edge comes about 1.75 mm from the
    wall at the circle's edge, if the centre is right. With the ~1 cm
    centre error the logs suggest, it may touch.
  - The beaker clamps to 15 mm: radius 22, wall 1, clearance 6.
- `test_skills_offline.py` 373 passed, the 2 old scoop failures.
- Not yet run.

**Whole pipeline on the robot, run of 16:24 (2026-10-08,
`experiments/agent_run_20261008_164945.json`): sub-tasks 1-16 all ok.**
- Covered: citric acid and baking soda each scooped, dumped (A at
  (0.412, -0.159), B at (0.434, -0.011)) and stirred; the stirrer went
  back twice; red cabbage was scooped.
- This run started before the dump offset and the stir radius were added,
  and loaded the code of 16:24, so it had neither: dump `forward_offset`
  0.0, stir at the skill's defaults. **The user's "dump too forward" and
  "stir too little" came from this same run.**
- **Sub-task 17, dump into the beaker, "failed": "Tip stalled at 59.9 deg
  of 64 deg, short of min_tip_deg=60".** The user watched it empty the
  scoop well. **`DUMP_MIN_TIP_DEG` 55** is now a bench default for dump.
- **Resume: `run_experiment.py --resume RECORD [--from-step N]`.**
  - It takes the record's goal and last plan and skips the sub-tasks before
    N (default: the failed one).
  - It replays their recorded results into
    `SkillsExecutor._remember_pick_site` / `_track_held` and the loop's
    held object and history, without moving.
  - Offline test `test_a_run_resumes_from_its_record`;
    `test_agents_offline.py` 88/88 in both venvs.
  - A dry run from this record restored 16 steps ("holding larger spoon;
    pick sites known for larger spoon, stirrer") and resolved sub-tasks
    17-31 to valid calls.
  - **The bench must be as the record left it.** For 17, the loaded scoop
    must still be in the jaws; if it was taken out, resume from 15.

**Resumed at 17 (2026-10-08, `experiments/agent_run_20261008_181820.json`):
sub-tasks 17-22 ok.**
- 17: dump into the beaker. Rim (0.556, 0.025) at z 0.06; `forward_offset`
  -0.01 applied; tip capped by reach at 66 deg, reached 62.0 deg; bowl
  drift up to 21 mm while tipped.
- 18: scoop back. 19: stirrer picked. 20: beaker stirred. 21: stirrer back.
- 22: **the clear beaker picked: the first clear container picked on the
  robot.**
- **23, pour into A, failed: "Cannot locate 'clear cup a'".**
  - With the beaker in the jaws, the vote still counted "50 ML WATER" as a
    container on the bench, and the beaker's label paper still lay by cup A.
  - Reads: A's cup A x3 + "50 ML WATER" x1; B's cup "A" x2 and never "B";
    C x3.
  - Two share-outs scored 6 (A on A's cup; or A on B's cup and the beaker on
    A's cup), so A was refused.
- Fixes:
  - `SkillsExecutor._track_held` now calls `vision.set_held(...)`, and
    `_locate_by_vote` leaves the held container out of the vote. The
    offline test replays this exact case: refused without the hint, A found
    with it.
  - A resumed run's record now carries the steps before it
    (`resumed_steps`), so a resume of a resume restores everything.
- Dry run of `--resume ..._181820.json --from-step 23`: restored 6 steps
  ("holding plastic beaker; pick sites known for plastic beaker, stirrer");
  23-31 resolved.
- `test_skills_offline.py` 378 passed, the 2 old scoop failures.
  `test_agents_offline.py` 90/90 in both venvs.
  - If 15 mm works, the model is about 1 cm out somewhere: table z, dish
    base thickness, or the tool's depth below the TCP. That would matter
    for dump, pour and stir heights too.

**Resumed at 23 (2026-10-08, `experiments/agent_run_20261008_182937.json`):
23 and 24 both poured into cup B; 24 left the arm stuck inside
franka-interface's virtual floor; 25 failed.**

- **23, "pour into A", poured into B.** Both pours went to lip_xy
  (0.420, -0.021). B's cup is at (0.431, -0.021), A's at (0.415, -0.140).
  - Cause: label reads. On the 18:18 look, B's cup was read
    "A 10 ML WATER" by 2 cameras and "B" by none, and A's cup "A" by 3.
  - The vote gives A to whichever "A" cup collects more reads. At step 23,
    B's cup did.
  - Fix, `misread_twin` in `_locate_by_vote`: refuse instead of guessing
    when both hold:
    - a look-alike label (one word apart, as A and B are) is read on no
      container;
    - the requested label is read on two or more distinct cups (each placed
      by 2+ cameras, at least one cup diameter apart).

    It also refuses when the cup the vote picks for A is the only cup B
    was read on. Example: B's cup read "A" 3 times and "B" once, A's cup
    read "A" once. The most reads would then put A on B's cup.

    The message names the label to rewrite. The offline test replays the
    18:18 reads: A is refused. With one "B" read on B's cup, A and B are
    both found.
  - **The real fix is the label.** Rewrite B (and A) big and bold, as was
    done for C: C went from 0 reads to 3 of 3.
- **24, pour into B at 90 deg: franka-interface aborted the last tip step
  and refused every motion after it.**
  - ROS log (`~/.ros/log/latest/rosout.log`): skill 545 (the 80->90 step)
    started 18:29:22, and "franka_interface status is not ready" came at
    :27. The righting (546), go-home (547) and place hover (548) were all
    refused.
  - Pour still reported success: frankapy's `ignore_errors` hides
    refusals.
  - Place's "unreachable (off by 271mm)" is the distance from the 90 deg
    pose (TCP 0.406, -0.021, 0.084) to the hover. The arm never moved.
  - **The franka-interface terminal** (the window running
    `./franka_interface --robot_ip ...`; the ROS-interface window only shows
    protobuf noise) said:
    - `Frame 7is in collision with wall 3with distance 0.0978709`
    - `Robot is in collision with virtual walls!`
  - This is franka-interface's own virtual floor, nothing physical. The
    user watched it: nothing was touched.
    - The plane is at z ~ -0.015. Frame 7 is the flange. At 90 deg the hand
      lies flat and the flange was 82.5 mm up.
    - Step 23 at 80 deg ran with the flange 105 mm up. The threshold is
      therefore 0.098-0.120 m from the plane.
    - robot_state also had
      `last_motion_errors.cartesian_motion_generator_joint_acceleration_discontinuity`,
      from the abort.
  - **An arm parked inside the wall cannot be moved by software.** Every
    motion is checked from its start pose. Restarting the control stack
    did not help: three reset_joints, 19:03-19:05, were all refused.
    Recovery: guide it out by hand (white light, wrist buttons), then blue
    light and reset_joints.
  - Fix, pour's `flange_min_z` 0.11 (base frame): `_lip_pose` raises the
    TCP so the flange stays at or above it.
    - Over cup B with the beaker: lip 20 -> 25 mm over the rim at 80 deg,
      15 -> 41 mm at 90 deg. 70 deg and below are unchanged.
    - Sim and robot plan the same.
    - **Not yet run on the robot.**
  - Also in this run: step 23's 60 deg step was aborted at 18:26:53 (not
    ready for under a second; it recovered by itself), so the beaker went
    from 50 to 70 in one step. Its cause was not captured.
- **Refusals are now caught where they happen:**
  - `go_home()` checks the joints reach frankapy's HOME_JOINTS (0.1 rad).
    When they do not, pour fails with "Poured, but the arm did not come
    back: it is still tipped N deg over X".
  - `goto_pose_rigid` prints "[Safety] The arm did not move at all" when
    the arm moved under 5 mm and is more than 50 mm off.
  - `BaseSkill.robot_fault()` appends franka-interface's readiness and
    libfranka's current/last-motion error flags. They come from the raw
    robot_state; frankapy's state dict leaves them out.
  - Place will not start with the held object tipped more than 20 deg from
    the pick's tool axis. The executor fills `pick_tool_z` from the pick's
    grasp pose.
- Tests: `test_skills_offline.py` 394 passed, the 2 old scoop failures.
  `test_agents_offline.py` 90/90 in both venvs.
- Next run (given to the user): reset_joints, rewrite the B label, then
  `--resume experiments/agent_run_20261008_182937.json --from-step 23
  --no-verify --step --max-retries 0`. The beaker is still in the jaws.

**Resumed at 23 again, B relabelled (2026-10-08,
`experiments/agent_run_20261008_192626.json`): 23-25 ok, 26 failed.**
- 23 poured into A: lip (0.389, -0.160). 24 tipped the full 90 deg into B
  with the flange floor in place: lip (0.391, -0.049), tip 90.0. The user
  had moved B while relabelling. **This is the first robot run of the
  flange floor, and it worked.** 25 put the beaker back.
- **26, pick up clear cup A, failed:** "Object lost during the lift: gripper
  went from 16.9mm to 0.2mm". The cup is 55.3 mm across.
  - The user saw the jaws go in askew: "the mouth of cups A, B and C is too
    wide; the gripper did not line up". 55.5 mm of rim in the 80 mm opening
    leaves about 12 mm a side.
  - Before the user said what they saw, I guessed a soft cup crushed at its
    rim and planned a lower grasp (z_offset). That was not the problem, and
    the change was never made.
- The user read `scripts/point_at.py` over A, B and C (its first use):
  A's perceived centre was 5 mm toward B; B and C were right.
- Fix: bench pick default for cup A, `lateral_offset`
  `CUP_A_LATERAL_OFFSET = -0.005`, 5 mm away from B. A stood at about
  (0.40, -0.16) and B at (0.40, -0.05). It belongs to that spot; if A
  moves, check it again with point_at.py.
- Not fixed, and offered to the user: pick_up's final descent runs under
  impedance and stops 5-6 mm short in x (measured on the spoon). Position
  control for that descent would remove the error, but it is a control-mode
  change and waits for the user's yes.
- Next: resume `experiments/agent_run_20261008_192626.json --from-step 26`.
  Nothing is held after step 25.

**Resumed at 26 (2026-10-08, `experiments/agent_run_20261008_195105.json`):
refused, not mis-picked.**
- `misread_twin` stopped it. B's cup at (0.395, -0.066) was read
  "A 10 ML WATER" x3, A's cup at (0.392, -0.178) x2, and "B" nowhere,
  although the user had rewritten B. Counting reads would have picked B's
  cup as A.
- **Cause: the label crop, not the label.**
  - In cam 2's and cam 5's frames "B 10ML WATER" is plain to read. The
    crops sent to the reader were about 100 px and cut the "B" off, so the
    model saw "10ML WATE" and answered "10 ML WATER" (no single label).
    Cams 3 and 4 answered "A".
  - `_paper_sheet_box` found no sheet for B's cup on any camera, and none
    for A's on 3 of 4. The sheets lie close together with powder spilt
    between them. The white mask (S<70, V>145, 9x9 dilated twice) joined
    each sheet to its neighbours into one blob of 66-80% of the search
    window, which is refused (cap 65%).
- Fix: a strict pass first (S<50, V>190, 5x5 dilated once), then the old
  pass as the fallback for dim light.
  - On the saved frames of that bench, a sheet was found for all 16
    containers.
  - gpt-4.1 reads on the same frames, by cup:

    | Cup | Before | After |
    | --- | --- | --- |
    | A | A x2-3, rest unread | A x3-4 (one run: one "C") |
    | B | A x2-3, never B | B x2, A x1, CITRIC ACID x1 |
    | C | C x3, CITRIC ACID x1 | C x4 |
    | beaker | 50 ML x1, "C EMPTY" x2 | 50 ML x3 |

  - Both runs of the reads vote A and B to the right cups.
  - Offline test: the cup's own sheet is found among powder and a
    neighbour's sheet; the old mask alone finds none; one dim sheet is
    still found.
- The bigger crops do show a neighbour's paper at times: cam 5's crop of
  A shows B's and C's papers. The reads above were right regardless.
  Watch for a label read on the wrong cup.
- How to look at what the reader sees: the scratch scripts
  `look_labels.py` / `save_frames.py` / `read_new.py` (session scratchpad,
  not kept). They capture once, save each crop, and read them.
- Tests: `test_skills_offline.py` 398 passed, the 2 old scoop failures.
  `test_agents_offline.py` 90/90 in both venvs.

**Resumed at 26 (`experiments/agent_run_20261008_200142.json`): A was
found, and the grasp from above failed again.**
- "Object lost during the lift: gripper went from 22.5mm to 2.7mm".
- **The user's procedure for picking cups A, B and C, now the bench
  default:**
  - pick_up `slide_in` 0.06: the jaws open fully, closing across base Y;
    they go down 60 mm behind the cup (toward the base), slide forward onto
    it, and close.
  - `z_offset` -0.012: "it does not go down enough, need to go down 12 mm
    more". The rim is now 25 mm above the TCP instead of 13.
- **Place is the reverse.** The executor turns the pick's `slide_in` into
  place's `slide_out`. Place sets the cup down where it was held (the TCP
  back to `pick_grasp_tcp`, `set_down_clearance` 3 mm up), opens, slides
  the open jaws 60 mm back (-X), and only then rises.
- The 60 mm keeps the fingers about 20 mm off the cup's back wall on the
  way down (rim 27.75 mm from the centre, finger pads ±10 mm).
- The slide passes the open fingers (±40 mm) either side of the cup. At the
  lower grasp the cup is narrower than its 55.5 mm rim.
- The slide moves run on pick_up's usual `move_to_pose` (impedance), so no
  control mode changed.
- Cup A keeps its -5 mm `lateral_offset`.
- Offline: `test_cups_are_taken_from_behind` checks the move order, that
  nothing comes down over the cup before the slide, the reversed place, and
  that the executor carries the slide over.
- Tests: `test_skills_offline.py` 409 passed, the 2 old scoop failures.
  `test_agents_offline.py` 90/90 (py3.8).
- **On the robot (run started 20:10):** the slide-in took cup A. The user:
  "the grasp force is too big, the cup is squeezed heavily". The jaws held
  at 37.0 mm (`/franka_gripper_1/joint_states`).
  - The user measured the cups: **73.2 mm across the outside at the rim**,
    their widest, and **about 61 mm where the jaws close**.
  - The 55.5 mm in the manifest since 2026-10-03 was not the rim.
    `CLEAR_CUP_RADIUS` is now 0.0366. The vote's "one cup placed twice"
    distance grows with it. The rim radius pour and stir read off the
    cylinder model grows too: pour's lip is 14 mm from the target's centre
    instead of 11. Most of the old miss when coming straight down was
    likely this: a 73 mm rim in the 80 mm opening leaves under 4 mm a side.
- Fix: pick_up `grip_width`, a measured width to close to and stop at (a
  position move, `grasp=False`). The force-limited 1.5 N close keeps
  squeezing a soft cup. Bench default for A, B, C: `CLEAR_CUP_GRIP_WIDTH`
  0.058 (61 - 3 mm): wider if it still squeezes, narrower if the cup slips.
- Caveats:
  - A position close cannot tell a cup that slips out during the lift: the
    jaws stay at 58 mm either way. Only the force-limited close shrinks to
    "lost". Watch the lift.
  - The finger bodies above the pads may still meet the wider part of the
    tapered cup.
- Tests after `grip_width` and the new cup size: `test_skills_offline.py`
  410 passed, the 2 old scoop failures. `test_agents_offline.py` 90/90
  (py3.8).

**Resumed at 26 (`experiments/agent_run_20261008_202116.json`): 26-31 all
succeeded.** The steps: pick A, pour into C, place A back, pick B, pour into
C, place B back. That used the slide-in pick and the 58 mm `grip_width`.
- **With this, all 31 sub-tasks have run on the robot**, across the 16:24
  run and its resumes (17, 23, 26).
- The record says "every sub-task executed". The user's own view of the
  last steps was not given beyond running on.
- Full run again (as the 16:24 one, plus `--step`):
  `python scripts/run_experiment.py --workspace-min 0.25 -0.40 -0.13
  --instruction-image instructions/magic_beaker.png --follow-steps
  --no-verify --max-retries 0 --step`
  - The planner re-plans from the sheet each run, so the count may differ
    from 31.
  - Reset the bench first: scoop at its spot, stirrer in its holder, cups
    and beaker on their labels, nothing in the jaws.

**Full run again, 20:27 and 20:40 (2026-10-08).**
- 20:27 (`agent_run_20261008_202700.json`): refused at sub-task 1, "holding
  something unidentified".
  - The jaws stood at 58.2 mm, cup B's `grip_width`.
  - The orchestrator counts 1-75 mm as holding.
  - Fix: open the gripper and start again. The check is right to refuse:
    it keeps a held object from being dropped at the start.
- 20:40 (`agent_run_20261008_204018.json`): sub-tasks 1-8 ok; 9 (scoop
  baking soda) failed: "Failed to clear the cameras before scanning".
  - ROS log: step 8's pick lift (skill 241) ended 20:40:15. Step 9's
    reset_joints (242) came at 20:40:17 and was refused, "franka_interface
    status is not ready".
  - robot_state: `last_motion_errors.cartesian_motion_generator_joint_velocity_discontinuity`.
  - franka-interface recovers from such an error on its own within about a
    second, and a command sent inside that window is refused. Step 23's
    18:26:53 abort was the same kind.
- Fix: `go_home` retries once, after `retry_wait` (2 s), when the refused
  reset left the arm where it was (< 0.01 rad). A move aborted part-way is
  not retried. Offline: refused once, then home on the retry; always
  refused, tried twice and reported. `test_skills_offline.py` 412 passed,
  the 2 old scoop failures.
- Next: `--resume experiments/agent_run_20261008_204018.json` (from the
  failed step, 9). The arm was at home holding the spoon (jaws 8.3 mm).

**Scoop put-back drift, and the scoop too shallow (2026-10-08, 20:40-20:50).**
- **Put-back drift.**
  - Place-backs released with the TCP over the object's centroid. The scoop
    is taken 10 mm behind its centroid and the arm stops ~5 mm short, so
    every put-back moved it ~15 mm forward.
  - 20:40 run: picked with the jaws at x 0.6319, released at 0.6473, and
    the next pick found the centroid 27 mm on.
  - Fix: with `pick_grasp_tcp` (any executor put-back), place releases with
    the TCP at its xy. The release height still comes from the site.
  - Offline test `test_place_back_returns_the_tcp_to_the_grasp`, not yet
    run: the user stopped the test run to get to the robot.
- **20:50 run** (`agent_run_20261008_205043.json`, stopped by the user before
  sub-task 3): "the scoop depth is too shallow! add at least 15 mm, does the
  offset not work?"
  - The offset was applied: `floor_z` -0.007 in every run since 16:24
    (table 0 shifted -15 mm, + 8 mm floor).
  - What differed was the per-pick jaws-to-bowl x offset: 51.1 mm, against
    40.0 / 39.9 mm in the runs -15 mm suited. The jaws close on a different
    spot of the handle each pick.
  - At the 60 deg dig, an x offset 11 mm too large puts the real bowl
    ~10 mm high.
  - `DISH_SCOOP_Z_OFFSET` became -0.030, as asked; then -0.025 at the
    user's word ("put the scoop offset z to 25 mm, not 30").
  - Risk, told to the user: with an offset measured near 40 mm again,
    -30 mm may press the bowl into the dish floor.
  - The real cure is a repeatable grasp point on the handle or a better
    offset measurement. Not done.

**20:52 run (`agent_run_20261008_210905.json`): 1-13 ok, 14 failed.**
- The run started before `DISH_SCOOP_Z_OFFSET` became -0.030 (21:05), so
  its scoops ran at -15 mm (`floor_z` -0.007).
- 14 (stirrer back into its holder): "Failed to reset_joints before
  inserting".
  - The trip home (skill 502) started 21:09:02, three seconds after the
    stir's lift (501) ended, and was terminated at 21:09:05 "not ready".
  - robot_state again had
    `last_motion_errors.cartesian_motion_generator_joint_velocity_discontinuity`.
  - No retry followed. The run had the retry (base_skill.py 20:41 < run
    20:52), so the arm had moved, and a part-way abort is not retried by
    design.
  - The console and franka-interface lines were asked for, to tell why;
    not received yet.
  - Afterwards the arm was at home, still holding the stirrer (jaws
    29.9 mm).
- **Stir's descent to the cup floor was too slow (the user).**
  - `to_floor` felt down 1 mm per step from 10 mm above the rim: ~60
    separate motions, each with a force read. Sub-task 13 took 162 s.
  - Now `coarse_step` 5 mm through the open cup and `probe_step` (1 mm) only
    in the last `fine_zone` 15 mm above the deepest target. Offline: under
    60% of the steps.
  - A contact within one step below the rim now counts as landing on the
    rim: a coarse step can carry the commanded tip up to 5 mm past it, and
    that rim landing had been read as "in the cup" (offline test caught it).
  - Not yet run on the robot.
- **The beaker's grasp was too high (the user, a resume of that run).**
  `BEAKER_GRASP_Z_OFFSET` -0.012 is now the beaker's bench pick default.
  The user gave no number, so 12 mm was taken from cups A-C.
  - A grasp moved down holds the object higher in the jaws. Its put-back
    must release that much lower, or the object falls the difference.
  - pick_up now reports its `z_offset`. The executor keeps it per object
    (`pick_z_offsets`, restored from records on resume). A put-back of an
    object picked lower gets `release_clearance` 20 mm + z_offset (8 mm for
    the beaker). The scoop and stirrer are unchanged.
- **go_home also retries a trip home cut short by a timing fault only.**
  - Sub-task 14 showed the first retry rule (only when the arm had not
    moved) was too narrow.
  - The retryable flags (`RETRYABLE_ERRORS`) are any `*discontinuity` and
    `communication_constraints_violation`, read from the raw robot_state by
    `BaseSkill._error_flags`.
  - Any other flag (reflex, safety, limits) is never retried.
  - Offline: cut short by a velocity discontinuity, home on the retry; cut
    short by `cartesian_reflex`, tried once and reported.
- The put-back fix on the robot (20:52 run): the scoop was released where
  the jaws had closed (4: 0.6308 = pick 1's 0.6308; 11: 0.6508 = pick 8's).
  Pick 8 still saw the centroid 20 mm on, so that remaining difference is
  perception (or the spoon settling), not the put-back.

**Before the next full run (given to the user):**
- Reset the bench: scoop at its spot, stirrer in its holder, A/B/C and the
  beaker on their labels, nothing in the jaws.
- Run with `--reset`: on the robot it homes and OPENS the gripper before
  starting. That removes the "holding something unidentified" refusal, and
  drops anything still held.
- Command: `python scripts/run_experiment.py --workspace-min 0.25 -0.40
  -0.13 --instruction-image instructions/magic_beaker.png --follow-steps
  --no-verify --max-retries 0 --reset`
- Not yet seen on the robot:
  - scoop at -25 mm;
  - the faster stir descent;
  - beaker -12 mm and its lower put-back;
  - the timing-fault retry.
- Tests with all of the above: `test_skills_offline.py` 421 passed, the 2
  old scoop failures. `test_agents_offline.py` 90/90 (py3.8).
- **The beaker's first pour (into A, sub-task 23) now tips to 60, not 80**
  (the user: "turn that 80 into 60").
  - `POUR_KEEP_SOME_DEG` is 60. It caps a pour whose container pours again
    before it is put down; B's pour, the beaker's last, stays 90. (Widened
    2026-10-15 to "pours again anywhere later in the plan"; see the end of
    this file.)
  - The catalog's cap sentence says 60. Its general advice ("about 80" for a
    partial pour) is unchanged. A version saying "about 60" lasted an hour.
    It was reverted because it would also have steered the model's free
    choices: sub-task 27, A into C, which it planned at 80.
  - The real beaker (r 22 mm, ~60 mm, both unmeasured) with 50 ml spills
    from ~51 deg and keeps ~33 ml at 60. The sim's wider beaker keeps
    everything below ~70, so in sim this pour now gives A little.
  - First I misread "the first dump into A" as the scoop dump and set
    `dump_angle_deg` to 60. Reverted the same hour: dump is back to 79,
    min_tip 60 (the bench default 55 still applies).
  - `test_agents_offline.py` 90/90 (py3.8), with the cap test at 60. Dump
    tests pass.
- Next: `--resume experiments/agent_run_20261008_200142.json --from-step 26`.


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
  --skill stir    --params '{"target_container":"white paper cup","revolutions":2}' \
  --skill place   --params '{"target_location":"stirring rod","vertical_insert":true,"release_clearance":0.0,"stop_force_n":1.0}'
```

Re-run on 2026-09-27 after merging `5c56826`: stir now uses its own
defaults (`tool_axis` "z", `tool_length` 0.081), and the stirrer goes back
through `place`'s `vertical_insert` to its remembered pick site. All 7 steps
succeed. The stirrer seated at z=0.1043 (5.85 N) and nothing moved more than
3 mm. The dump tipped to 69° and came back to within 9°.

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

### Random container layouts in sim (2026-09-27)

*Superseded on 2026-10-02 for the full-kit bench (shuffle within a size class,
then jitter, under measured sweep rules); see "The Magic Beaker kit in sim".
The flags below are unchanged.*

`run_experiment.py --sim` now places the cups and dishes at random on every
run. The seed is printed, so a run can be repeated.
- `--sim-layout-seed N` repeats a layout.
- `--sim-fixed-layout` gives back the default bench.
- `smoke_test_sim.py` and `film_sim_skill.py` keep the fixed layout, as does
  `build_cell(layout_seed=None)`.

What `bench.randomize_layout` moves and why:
- **Only containers move:** the beaker, the paper cup and the two powder
  dishes. The spoon, the stirrer and its holder stay put, because the skills
  pick those up and seat them by memory.
- **Each container is drawn from a 6 cm disc around its usual spot**, not
  anywhere on the table. That keeps the camera coverage and the reach the
  default bench was validated with.
- **Draws are rejected unless** the container is 0.35-0.56 m from the base
  (past about 0.56 m a straight-ahead dump cannot reach its 60° minimum), is
  3 cm clear of every other footprint (labels included), and is 12 cm clear of
  the stirrer holder, which dump's wrist swept into from 16 cm.

Validated with the 7-step sim chain above on seeds 1, 2, 3, 42, 777 and 2024:
7/7 steps on every seed, no contact other than the grasps and the stirrer in
its bore, and nothing moved except the spoon nudged 13.6 mm on release
(seed 1). The dump tipped to 65-79°, depending on how far out the paper cup
landed.

### Sim gripper: jaws clipping through held parts (2026-09-27)

This showed on screen as the jaws passing into the stirrer's cube, and it
affected every magnet grasp. It was two bugs in `SimFrankaArm`, both
sim-only:

- **The force-limited hold command was off by a factor of 2.** The actuator
  drives the `split` tendon, whose length is half the jaw width (0.5 per
  finger). The hold was `(kb*w - limit)/kg` where it should be
  `(kb*w/2 - limit)/kg`, which made it `2w - limit/kg`.
- **Once a part is attached it no longer collides with the hand**
  (`_exclude_from_hand`), so nothing stopped that command:

  | Part | Width | Jaws ended at |
  | --- | --- | --- |
  | stirrer cube | 30 mm | 42.4 mm with `grasp_force` 1.0 (open, not touching); 31.5 mm with 1.5 |
  | spoon handle | 8 mm | **0.0 mm, straight through it** |

  `_try_grasp` also snapped the fingers to `grasp_width`, which put the pads
  about 1 mm into the cube, because the pad faces sit inside the joint gap.

Both are fixed. The hold formula is corrected. At attach, the jaws stay where
the part stopped them, and `ctrl` is set to hold them there. Measured after
the fix: the stirrer at 30.0 mm and the spoon at 7.8 mm, both steady after
1 s. The smoke test passes 9/9 and the 7-step sim chain above runs clean.

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

**Deeper scoop variant (2026-10-05).** Add `"floor_thickness":0.001` to the
scoop params (the default is 0.004) to put the bowl tip 3 mm deeper.
`tip_floor_gap` (3 mm) is then held above an assumed floor that is 3 mm
lower. Prefer this knob over `tool_offset` z, which dump also uses. The
log's `[Scoop] Inside floor z=` line shows the floor it assumed. Not yet run.

### First robot run of the stirrer put-back: it pressed and never let go (2026-10-05)

**On this arm, pressing down reads NEGATIVE Z.** frankapy's docs and the sim
both said positive. The `place` step-7 log, baseline −2.61 N at z=0.1069,
change from baseline on each 2 mm step down:

```
z=0.1049 -5.99N   z=0.1029 -17.11N   z=0.1009 -34.74N   z=0.0989 -60.94N
z=0.0969 -76.13N  z=0.0949 -31.32N   z=0.0929 -3.57N    ...   z=0.0789 -95.71N
```

`place` only stopped on `push > stop_force_n` (positive), so it never
fired. It stepped 28 mm past first contact, up to **−96 N**, with the
stirrer already seated, then reported "never reached 1.0N" and kept hold.
The drops back toward 0 (−76 → −3.6 N, −96 → −0.2 N) look like the jaws
slipping down the cube: the pick closes by width, not by force, so it
held well beyond 1 N before slipping. Stir's guard was already
sign-agnostic, which is why stir behaved.

Fixes:
- `place` stops on **|change|** > `stop_force_n`, the same rule as stir.
- The sim's `get_ee_force_torque` now returns the reaction, so pressing
  reads negative there too. A positive-only test can no longer pass in sim
  and fail on the robot.
- The `ee_wrench` / `ee_push_up_n` docstrings now say negative, and that
  contact must be detected by magnitude.
- Offline `PressArm` reads negative. New test
  `test_place_releases_on_a_negative_press` replays this log: it releases at
  z=0.1029, one step after the seat.
- Sim pick → stir → insert: contact −2.78 N, stirrer 0.2 mm from its seat,
  0° tilt, released. `smoke_test_sim.py --only "pick + stir,scoop then
  stir,force guard"` passes.

**First contact came at z=0.1049, 13 mm ABOVE the remembered grasp
(0.0919).** The memory stored the *commanded* grasp, and `move_to_pose`
accepts a grasp move up to 30 mm off. The likely cause is that the arm
closed about 13 mm high, e.g. the fingertips met the holder top. Unconfirmed:
step 5 would have printed `Reached within … mm of target`. The other
possibility is that the stirrer sat lower in the jaws after the lift, because
of rod-in-bore friction. pick_up now records `grasp_tcp_reached` (the TCP
with the jaws closed) and prints `Closed at …` when it is more than 5 mm off
the command. The put-back aims at the reached position. The probe window
(+15/−20 mm) had only 2 mm to spare on this run.

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

**First full run start to finish on the robot (2026-10-08 21:33-22:10,
`experiments/agent_run_20261008_221001.json`): 31/31 sub-tasks, 2177 s, no
replans.**
- It started before the -25 mm scoop (21:52) and the 60 deg first pour
  (22:05), so it ran -30 mm (`floor_z` -0.022) and 80 into A. The next run
  has both.
- On the robot, as changed today:
  - **Cups A and B:** slid in from 60 mm behind, 12 mm lower (TCP z
    0.0211), closed to 58.1 mm. Both held and were poured into C (A at 80,
    the model's choice; B at 90).
  - **Beaker:** picked 12 mm lower (TCP z 0.0348, jaws 38.0 mm). Into A at
    80, into B at 90 with the flange floor.
  - **Stir:** with the coarse descent, 110 s per stir (162 s before).
    `stopped_by_force` in A and B, not in the beaker (rod shorter than the
    beaker is deep).
- **Scoop:** the jaws-to-bowl x offset measured 54.4 / 43.3 / 56.1 mm for
  the three scoops. It still varies per pick; the user said nothing about
  the depth this time.
- The user moved the scoop to y -0.14 between runs; it was found and picked
  there.

**Seeing the point cloud (2026-10-10): `scripts/view_pointcloud.py`.**
- Read-only. One frame per camera, every 2nd pixel deprojected in colour
  and moved to the base frame with the skills' extrinsics, cropped to the
  bench box.
- Writes `diag_out/pointcloud/pointcloud_<time>.ply` and a `.png`, then opens an Open3D
  window. Needs the robot venv (py3.8 has open3d 0.19; perception_env does
  not).
- Options:
  - `--objects NAME ...` paints what `vision.locate` returns for each, in its
    own colour;
  - `--per-camera` colours by camera, for a calibration check;
  - `--video` renders a 120-frame turntable .mp4;
  - `--no-show` skips the window.
- First capture: about 270k bench points from the four cameras. Dishes,
  papers, stirrer holder, cage posts and the hand at home are clear. **The
  clear cups barely appear: depth passes through them**, which is why they
  are placed by silhouette. The scoop, highlighted, sat where it lies.
- Open3D's `ViewControl.rotate` drifted the turntable off level. The video
  now re-aims the camera round the vertical axis each frame.
- **What the viewer shows is not what locates objects.** Objects are found
  from 2D masks (SAM, label reads). Only the masked pixels are deprojected,
  each camera separately, and cameras that disagree are rejected before
  fusing. Clear cups are triangulated from silhouettes, with no depth. The
  viewer overlays every raw pixel of all four cameras, unfiltered. The user
  found it "terrible, all messed up together".
- **Measured from the same capture, 2026-10-10 (scratch `cam_agree.py`):**
  - Bare-table patch (x 0.30-0.80, |y| < 0.35):
    - cams 2/3/4/5 put the table at -12.7 / -11.5 / -11.9 / -13.2 mm:
      agreeing within 2 mm, but **~12 mm below the `table_z` 0 the code
      assumes**;
    - tilts dz/dx 11.5 / 6.8 / 5.5 / **27.1** mm/m, dz/dy 4.9 / -6.6 /
      -14.5 / -12.6 mm/m.
  - White dishes, centroid of each camera's own masked depth: the same dish
    lies **34-49 mm apart** across cameras in xy. Part is visibility (each
    camera sees the near wall), part calibration.
  - This is the likely common cause behind several bench offsets: scoop
    z_offset -15..-25 mm (the table is 12 mm low), cup A -5 mm, dump 10 mm
    back.
  - The root fix is a recalibration (`scripts/calibrate_cameras.py`,
    `validate_calibration.py`). Offered to the user, not done: every tuned
    offset would need rechecking after it.

**Why the fused cloud looks like a mess (2026-10-10, measured offline from
`pointcloud_20261010_153342.ply` (`--per-camera`) and `..._153049.ply`
(colour)).**
- **Not the table, not flying pixels.** Each camera's own table plane:
  - resid std 1.4-1.5 mm, 10-17 mm below z 0;
  - the four planes agree within 1.6 mm at (0.55, 0) and ≤ 8 mm at the
    corners;
  - points with < 4 neighbours in 8 mm: 0.3-0.5 % per camera.
  - The red/green/blue/yellow speckle on the table in `--per-camera` is that
    1.5 mm noise interleaving. It is normal, not a calibration sign.
- **Objects are shifted between cameras by centimetres.** Two independent
  checks, same answer:
  - ICP on object points, same-side pairs (fitness 0.91-0.98): 3→2
    (+37, +3, -3) mm; 5→4 (-33, +13, +5) mm.
  - **Flat white paper on the table** (no height step, no occlusion, no edge
    veil, so visibility cannot explain it), best-IoU shift: 3→2 (+36, +4),
    5→4 (-36, +4), 4→2 (+40, +20), 5→3 (-32, +16), 5→2 (+4, +20),
    4→3 (+4, +4) mm.
  - Pattern: **every camera puts things farther from itself**, about 18 mm
    horizontally. Cameras opposite in x (2/3, 4/5) disagree by ~36 mm.
    Every object therefore shows up as two to four offset copies.
- How the colour PLY was split back into cameras: each camera's block is in
  image raster order. So y/z in that camera's own frame never decreases;
  the first decrease marks the next camera. This recovered blocks of
  67210 / 70058 / 70642 / 64994 points.
- **Likely root cause: the cube calibration bakes each camera's view of the
  cube into its extrinsics.**
  - `calibrate_cameras.py` pairs the TCP with the median of the cube's
    *visible* surface.
  - Each camera sees the 4 cm cube's top and the faces toward it. Its
    "centre" therefore sits toward that camera and up, and Kabsch absorbs
    this.
  - The result: that camera's whole cloud is pushed away from it and down.
  - Modelled (scratch `cube_bias.py`): 40 mm cube over the default sweep,
    ±y faces hidden by the finger pads, real camera positions. It predicts:
    - pushes of (+12, 0, -9) / (-9, +1, -11) / (-11, -1, -10) /
      (+10, -1, -10) mm for cams 2/3/4/5;
    - pairwise 3→2 +21, 5→4 -21, 5→2 +1, 4→3 +2 mm in x.
  - **Sign and pattern match all six measured pairs, and the z push matches
    the low table. The model gives ~60 % of the measured magnitude.**
  - Probably the rest: the sweep only spans x 0.42-0.62, |y| ≤ 0.20,
    **z 0.10-0.25**, so the table is extrapolated. Also the report's max
    residuals of 16-21 mm on cams 2/4/5.
  - Inferred, not proven on the robot.
- What would fix it (not done, nothing changed):
  - recalibrate with a view-independent centre: a ball with a sphere fit, a
    tag, or the cube's centroid corrected by half a side along the ray;
  - extend the sweep down to ~z 0.03 and wider in x/y;
  - then recheck every bench offset that has been compensating for this.
- Other things that add to the look but are secondary:
  - depth passes through the clear cups;
  - each camera sees only its own side of an object;
  - exposure differs: the same table has median brightness 0.18-0.19 in
    cams 2/3 against 0.28-0.34 in cams 4/5, so the colour view is a
    patchwork.

**Recalibration with a cube model (2026-10-10, code ready, not yet run on the
robot).** The user chose to recalibrate and then reset the per-skill offsets.
Only the green cube is available, **39.8 mm** (user's measurement).
- Re-running the old solve would not have helped. The Sept 4 calibration was
  fresh and still had the bias; the bias is in the method.
- `robochem/vision/cube_calibration.py` (new): each camera's raw cube points
  are fitted to a 39.8 mm cube centred on the TCP, jointly over all cameras
  plus the cube's tilt in the grip. The old median + Kabsch solve is the
  starting point.
  - The cube-to-TCP offset cannot be solved from these data: shifting every
    camera with it fits equally well. It moves all cameras together, so it
    only shows as the table height.
- `scripts/calibrate_cameras.py`:
  - `--centre model` is the default; the median solve is always printed for
    comparison.
  - **Cameras are now captured one at a time** (`VisionSystem(cameras={})`).
    The script used to stream all four together, which wedged USB on
    2026-09-25.
  - Default sweep is 48 poses: x 0.38-0.66, y ±0.25, **z 0.07-0.25** (was
    27 poses, 0.42-0.62, ±0.20, 0.10-0.25). **The bench under it must be
    clear.**
  - Every sweep saves `calibration_out/observations_<time>.npy`.
    `--from-observations` re-solves it with no robot.
  - The report adds:
    - per-camera table z (installed / median / new), from the last capture;
    - the cube centre each camera implies at every pose vs the TCP;
    - the per-pose gap between cameras.
  - `--level-table Z` shifts all cameras together in z. Use it only if the
    table really is at Z in the base frame. `REAL_TABLE_Z` 0 is an
    assumption, never measured.
- `scripts/test_cube_calibration.py` (new, offline, PASS). It renders the
  cube as the four installed cameras would see it: finger pads hiding the
  ±y faces, z-buffered per pixel, 1.5 mm noise, 5 % edge pixels.
  - Table-point error per camera, median solve: **12-14 mm in xy, -3.5 to
    -5.5 mm in z**. This reproduces the bench pattern.
  - Cube model: **0.2-0.4 mm**. A 4° tilt in the grip comes back as 3.9°.
- End-to-end through `--from-observations` on synthetic data, with the cube
  10 mm above the TCP:
  - the table read -9.5 mm on all four cameras;
  - `--level-table 0` recovered the truth to 0.2-0.4 mm.
- An earlier implied-centre diagnostic showed 30-50 mm outliers, from edge
  pixels on corner views. Gating to points within 10 mm of the cube first
  fixed it (max 6 mm). The calibration fit itself was never affected.
- **First live attempt (2026-10-10, 16:12-16:15).**
  - `--check` passed: all four cameras saw the cube at (0.513, 0.004,
    0.158). Masks were 577-1456 px, with 302-1050 points, spread 10-14 mm.
  - The `--skip-grasp` sweep then **segfaulted in librealsense** (dmesg:
    `python: segfault ... in libc-2.31.so`, 16:15:25) about a minute in.
    Nothing was saved: stdout through `tee` was block-buffered, and the
    observations were only written at the end.
  - The `usb 2-x: Failed to query (GET_CUR) UVC control 1 on unit 3: -32`
    spam (~700 per camera) appears on **all four** cameras from 16:12:47,
    including during the successful `--check`. It is noise from opening a
    pipeline, not one bad camera.
  - The four D415s sit on bus 2, ports 3/4/5/6, all at 5000 Mb/s.
  - Changes made:
    - observations are saved after every pose, and `--resume <npy>`
      continues from there (mock-tested: crash at pose 4 of 8, resume
      finished 4-8);
    - `keep_cameras_alive` holds every opened camera object and pauses
      0.3 s after each stop. This is a guess at the cause, not a proven fix.
    - Run with `python -u` so the log keeps up.
- **The table is 11.2 mm below the robot's base frame (user measured,
  2026-10-10).** The bottom of the white Panda base, the link0 origin, sits
  11.2 mm above the tabletop. So the real table is at **z = -0.0112**, not
  the `REAL_TABLE_Z` 0.0 / `VisionSystem.table_z` 0.0 the code assumes.
  - The old calibration's -10..-17 mm table was therefore mostly right in z.
    Its error was the 36 mm between cameras.
  - The skills aiming 11 mm high on the table fits the bench offsets they
    needed: scoop z_offset -15..-25 mm against a floor taken as
    `table_z` + 8 mm, and cup/beaker grasp z_offset -12 mm.
  - Plan:
    - install the recalibration with `--level-table -0.0112`;
    - set `REAL_TABLE_Z` / `table_z` to -0.0112 before re-tuning any offset
      (not done yet).
- **Second live sweep (2026-10-10 16:22-16:35, `python -u ... --skip-grasp`,
  `calibration_out/observations_20261010_162214.npy`,
  log `sweep_20261010.log`).** 48 poses, no segfault.
  - Usable observations: cam 2 **32**, cam 3 **42**, cam 4 **44**, cam 5 **2**.
  - **Cam 5 stopped delivering frames from pose 12 on**: 37 ×
    `Frame didn't arrive within 5000`. It was still dead after the sweep.
    `pyrealsense2 hardware_reset()` on serial 151322066932 brought it back
    (it re-enumerated). Before it died, at the z 0.07 layer, cam 5's masks
    were only 12-354 px. At `--check`'s z 0.16 it had 577 px.
  - Cams 2/3/4, cube model, with the cube 2.1° off square in the grip:
    - implied centre vs TCP: mean 2.4-2.8 mm, max 5.7-6.9 mm;
    - surface rms 1.4-1.7 mm;
    - gap between cameras per pose: median 4.3, max 8.7 mm.
  - Old median solve on the same data: held-out mean 4.2 / 5.6 / 9.5 mm.
  - Change from installed: cam 2 21.5 mm / 1.00°, cam 3 9.3 mm / 0.35°,
    cam 4 8.8 mm / 0.51°.
  - **Offline check on the 15:30 colour capture**, re-expressed with the new
    extrinsics (scratch `eval_new.py`):
    - flat paper shift 3→2 drops from **(+36, 0) to (+4, 0) mm**; 4→2 from
      (+40, +20) to (+4, +4); 4→3 from (+4, +12) to (+4, 0). 4 mm is the
      grid cell.
    - Paper IoU as-is 0.44 → 0.65, which equals the best after shifting.
    - New table z: -18.7 / -15.6 / -17.0 mm. Against the measured -11.2,
      the cube centre sits ~6 mm above the TCP.
  - Not installed yet: cam 5 still needs observations.
  - Additions to `calibrate_cameras.py`:
    - with a single `--cameras N`, that camera streams for the whole run
      (never stopped), and 5 frames are drained before each capture;
    - `--from-observations A.npy B.npy` merges sweeps. Valid only with the
      cube in the same grip.
  - Plan:
    1. `--skip-grasp --cameras 5 --check`, then the cam-5 sweep, same grip;
    2. merge, then `--level-table -0.0112 --apply`.
  - My Bash sandbox cannot open the RealSense devices: enumeration hangs.
    Running outside it was denied by the permission classifier, so the user
    runs camera commands.
- **Cam 5 re-sweep (16:50-16:55, `--skip-grasp --cameras 5`, same grip,
  persistent stream, `observations_20261010_165047.npy`,
  `sweep_cam5_20261010.log`).**
  - 0 dropped frames. 41 usable observations, 7 masks too small.
  - The first solve was garbage: median residual 124 mm, cube "23.8° off
    square". **Two misdetections were the cause**: pose 31 had 43 points
    0.59 m off; pose 44 was 1.9 m off. The other 39 sat within 5-19 mm of
    where the old calibration put the cube. Frame lag was ruled out: pairing
    with pose i-1 or i-2 fits worse.
  - The median solve now drops observations above max(30 mm, 4× median
    residual) and refits, up to 3 rounds. It dropped exactly those two.
- **Table in the fit.** With `--level-table Z`, each camera's bare-table
  points (from the sweep's last capture, 3000 per camera) are held at Z in
  the joint fit. The cube's height above the TCP becomes a fitted parameter:
  **+5.0 mm**.
  - Independent check on the 15:30 colour capture, which was not in the fit:

    | | table z, cams 2/3/4/5 (mm) | spread | paper shifts |
    | --- | --- | --- | --- |
    | installed (Sep 4) | -12.6 / -12.0 / -13.3 / -13.6 | 1.6 | up to (+40, +20) mm |
    | cube only, then leveled | -13.6 / -8.0 / -9.1 / -13.1 | 5.6 | 0-4 mm |
    | **cube + table (chosen)** | **-10.8 / -11.3 / -11.6 / -9.9** | **1.7** | 0-8 mm (one pair 8, rest ≤ 4) |

  - Tilts with cube + table: ≤ 8.5 mm/m, against up to 19 mm/m installed.
  - The cube fit itself loosens slightly: implied-centre mean 2.3-3.4 mm
    (was 1.9-2.7); per-pose gap median 5.4 mm (was 4.2).
- **Staged, not yet installed** (`calibration_out/`,
  `solve_merged_20261010.log`). Change from installed: cam 2 22.7 mm /
  0.68°, cam 3 9.8 / 0.28, cam 4 10.6 / 0.65, cam 5 21.6 / 0.67.
  Reproduce with:
  `python scripts/calibrate_cameras.py --from-observations calibration_out/observations_20261010_162214.npy calibration_out/observations_20261010_165047.npy --level-table -0.0112`

**Installed and verified (2026-10-10 17:05).**
- `--install-only` installed the staged cube + table solve. The previous
  extrinsics are in `calib/Past_Calibrations/backup_20261010_170502`.
- Fresh capture `diag_out/pointcloud/pointcloud_20261010_170527.ply`, bench
  partly restored:
  - table z: -10.8 / -11.0 / -11.4 / -10.1 mm (cams 2/3/4/5), tilts
    ≤ 6.7 mm/m;
  - flat-paper shifts 0-8 mm. 4→3 showed (0, -20), but its IoU was flat
    (0.40 as-is = 0.40 best), so that pair saw too little shared paper to
    judge;
  - **dish rim circle per camera (rim-top band z 10-24 mm): the four
    centres within 2.9 mm of their mean**, 65-85 % inliers. The same check
    with the old calibration on the 15:30 capture: 15.1 and 17.6 mm on two
    dishes, cams 2/5 against 3/4 about 28 mm apart in x, 21-38 % inliers.
  - Point-to-point ICP between opposite-side cameras on object points
    drifted to (+14, +72, -40) mm with fitness 0.94. It contradicts the
    1.3 mm table agreement: opposite cameras see different walls of each
    dish. **Do not use cross-side ICP as a calibration check.**
- **Table height is now -0.0112 everywhere on the real cell.**
  - `robochem.vision.REAL_TABLE_Z` feeds `VisionSystem.table_z`, which is
    where clear cups stand when rebuilt.
  - `run_experiment.py` now imports it; `point_at.py` passes
    `vision.table_z`.
  - Why it matters now: scoop's floor and pick's held-container bottom
    accept the table only when base_z - table_z is within -5..+12 mm. With
    the new calibration, base_z reads about -8..-5 mm. Against table_z 0
    that falls outside the window, so the floor would have fallen back to
    base_z.
- **Bench offsets reset to 0** for re-tuning (`bench_manifest.py`; the old
  values are kept in its comments):

  | constant | was |
  | --- | --- |
  | `DISH_SCOOP_Z_OFFSET` | -0.025 |
  | `DUMP_FORWARD_OFFSET` | -0.010 |
  | `CUP_A_LATERAL_OFFSET` | -0.005 |
  | `CLEAR_CUP_GRASP_Z_OFFSET` | -0.012 |
  | `BEAKER_GRASP_Z_OFFSET` | -0.012 |

  - The cups' -12 mm is most likely exactly the old table_z error: the cup
    model stood on z 0, while the table is at -11.2.
  - **Kept**, because they are not calibration:
    - the scoop pick's `forward_offset` -0.010 (crank geometry);
    - `STIR_RADIUS` / `STIR_WALL_CLEARANCE` and the impedance shrink;
    - `DUMP_MIN_TIP_DEG`, `sweep_mode`, cup `slide_in` / `grip_width`.
  - The pipeline commands' `lateral_offset` / `forward_offset` values for
    dump and stir, earlier in this file, were tuned on the old calibration.
    Drop them when re-testing.
  - Dump's forward offset may partly come back: the streamed carry
    overshoots 11.9 mm in free air, which no calibration fixes.
- Tests:
  - `test_agents_offline.py` 90/90.
  - `test_skills_offline.py` **must run in perception_env**:
    `perception_env/bin/python scripts/test_skills_offline.py`.
    - There it gives 421 passed, 2 failed: the 2 long-known scoop
      workspace-floor checks.
    - In the robot venv, 39 tests raise `TypeError ... 'FakePose'`. The test
      doubles subclass a stub RigidTransform, but that venv has the real
      autolab_core, so `to_rigid_transform`'s isinstance check fails.
    - An earlier note here called the suite "broken at HEAD"; that was this
      environment mix-up.
- **First bench check with `point_at.py` after the recalibration
  (2026-10-10).**
  - On every object the user checked, **the fingertips stopped 3-5 mm toward
    the base** from the true centre. That is one shift everywhere:
    perception reads x ~4 mm low.
  - Likely cause: the cube sat ~4 mm forward of the TCP along the pads. The
    sweep cannot see that offset (assumed 0 in xy), and it moves every
    camera alike.
  - Not excluded: the arm stopping short. `point_at` only warned above 5 mm.
    It now always prints the reached-minus-commanded offset in x/y/z.
  - Fix: `calibrate_cameras.py --cube-offset DX DY` moves every camera by
    (DX, DY) after the fit. The cube + table solve was re-staged with
    `--cube-offset 0.004 0` (`solve_merged_20261010_offset.log`). Its only
    difference from the installed one is +4.0 mm in x on all four cameras,
    rotations identical.
  - The terminal output of that point_at run was not kept.
- **Second point_at check, after installing `--cube-offset 0.004 0`
  (17:37, backup `backup_20261010_173751`).**
  - The user read the fingertips "still ~5 mm toward the base".
  - `arm reached` was within ±0.3 mm of the command in x on every move, so
    the arm is not the cause.
  - The reading did not move with the 4 mm shift: 3-5 mm before, ~5 after.
    So the reading itself is suspect. Hovering 5 cm above the object,
    **a 5° off-vertical view shifts the apparent position ~4 mm**. Viewed
    from the bench front, the fingertips appear displaced toward the base.
  - Next measurement, no parallax: fingertips 5 mm *below* the rim inside
    clear cups A/B/C (`--above -0.005`), then ruler the gap to the wall on
    the base side and the far side; offset = (far - base) / 2.
  - **Bug fixed in `point_at.py`**: it took the bench height (47.3 mm cup,
    29 mm dish) as an absolute z, ignoring `table_z`. Every hover sat 11 mm
    higher than `--above` said. It now uses `vision.table_z + height`.
- **Inside-the-cup measurement (2026-10-10, `--above -0.005`, +4 mm
  installed).** Cup A: wall gap 26.8 mm on the base side, 32.1 mm on the far
  side. So the fingertips sit **2.65 mm toward the base**, and the total x
  correction would be ~+6.5 mm.
  - Waiting for cups B and C before changing anything. Cup A's centre has
    been odd before (it needed its own -5 mm lateral offset on 2026-10-08).
- Cups B and C read the same as A, ~2.7 mm toward the base. That is one
  global offset. Re-staged with **`--cube-offset 0.0065 0`**
  (`solve_merged_20261010_offset65.log`): +2.5 mm in x on every camera
  over the installed +4 mm, rotations unchanged. Next: install, then one
  inside-the-cup check, expecting equal wall gaps (~29.5 mm each side on
  A).
  - Implied: during the sweep the cube centre sat ~6.5 mm forward of the
    TCP along the pads.

**Calibration done (2026-10-10 17:50).** Installed: cube + table solve with
`--cube-offset 0.0065 0` (backup of the +4 mm version:
`backup_20261010_175026`). Inside cup B at `--above -0.005`, the user read
the fingertips **at the very centre**. Final command, reproducible with no
robot:

```bash
python scripts/calibrate_cameras.py \
  --from-observations calibration_out/observations_20261010_162214.npy calibration_out/observations_20261010_165047.npy \
  --level-table -0.0112 --cube-offset 0.0065 0 --apply
```

- **To check absolute x/y on the bench, put the fingertips inside a clear
  cup below the rim and ruler the wall gaps.**
  - Reading by eye from 5 cm above was 3-5 mm off by parallax. It also did
    not move when the calibration moved 4 mm.
  - The cube-to-TCP offset in xy is invisible to the sweep. Expect to set
    it this way after any future recalibration, unless the cube is placed
    reproducibly.
- y was not measured with the ruler, so `--cube-offset` y stays 0. The
  earlier hover reads gave no left/right complaint.
- Next: re-tune skills one at a time with the bench offsets at 0. Drop the
  old pipeline `lateral_offset` / `forward_offset` params.
- **All bench offsets now 0 (user's request, 2026-10-10).**
  - Also the larger spoon's pick `forward_offset`, which was -0.010 and was
    there to keep the jaws off the crank. If they land on the crank again,
    put it back.
  - Not offsets, so unchanged: `STIR_RADIUS`, `STIR_WALL_CLEARANCE`,
    `DUMP_MIN_TIP_DEG`, `sweep_mode`, cup `slide_in` / `grip_width`, and
    the tool geometry. tool_offset is measured at each pick.
  - Skill test commands given to the user mirror the params of the 31/31 run
    (`agent_run_20261008_221001.json`): names only, plus `to_floor` for
    stir and `pour_angle` for pour. Everything else comes from the bench
    manifest and the executor.
  - `--step` only pauses agent runs (`--task` / `--instruction-image`). A
    `--skill` chain runs straight through and stops at the first failure.
- **First scoop test after the reset (2026-10-10, `--skill` chain, no log
  saved).**
  - The user: the scoop is still too high and needs ~25 mm more, at
    `z_offset` 0.
  - Not put back blindly. The table_z fix already moved the assumed floor
    11 mm down, so -25 today digs 11 mm deeper than -25 on 2026-10-08.
    Picks at a 40 mm jaws-to-bowl offset could press the bowl into the dish
    floor under the position sweep.
  - The depth error is mostly the per-pick jaws-to-bowl x offset: 40-56 mm
    on 2026-10-08, ~1 mm of height per mm at the 60° dig. Camera
    calibration does not fix it.
  - The spoon pick's `forward_offset` moving -10 → 0 also moved the grasp
    spot.
  - Asked for the `[PickUp] Measured tool offset` and `[Scoop] Inside floor`
    / `z_offset` / `Descending` lines before choosing a value.
  - Real cure, still open: a repeatable grasp point on the handle.
- **Scoop log from that run** (user-pasted):
  - The cloud read the dish at `top z=0.0170, base z=0.0028`. The real rim
    is -11.2 + 29 = 17.8 mm, so the top was right; the base was 14 mm above
    the table.
  - base_z - table_z = 14 mm is outside scoop's -5..+12 mm "on the table"
    window, so the floor fell back to base_z + 8 mm = **0.0108**. The real
    floor is about **-0.0032**: the whole dig ran 14 mm high.
  - With the old calibration and table_z 0, the window passed (base_z read
    -0.1..-1.5 mm), but the floor was then assumed 11 mm high. That explains
    much of the -15..-25 mm the scoop needed before.
  - The measured jaws-to-bowl offset was 34.6 mm (forward_offset now 0; the
    ruler gave 52.9 at -10 mm).
- **Fix (`scoop.py`):** with a known `container_height`, a container whose
  measured top lies within 8 mm of `table_z + height` counts as on the
  table, and the floor is `table_z + floor_thickness`. The base_z window
  still works on its own.
  - New test: `test_scoop_floor_from_the_table_when_the_base_reads_high`.
  - Run in perception_env: the two new checks pass. The unchanged-path
    check needed a 1 mm tolerance, since the synthetic cloud's 3rd
    percentile is 3.3 mm, not 2.8 mm. Not re-run after the user asked to go
    straight to the arm.
- **`DISH_SCOOP_Z_OFFSET` = -0.011.** The user asked for 25 mm deeper:
  14 mm comes from the floor fix and 11 mm from this offset. Not yet run on
  the arm.
- **Scoop run 18:19 (2026-10-10, `experiments/test_scoop.log`): the user
  calls it "perfect".**
  - Settings: `z_offset` -0.011 and the floor from the table "by its top".
    The log line: `Inside floor z=-0.0142 (table z -0.0222 ... on the table
    by its top) + 8 mm`. The table and floor are shown shifted by the
    offset; the real floor is -0.0032.
  - Measured jaws-to-bowl offset 35.7 / 5.3 mm (forward_offset 0).
  - **The bowl reached the floor.** The compliant descent stopped 6.5 mm
    above its command (2.7 mm the run before), and was +3.6 mm after
    compensation. The position sweep ended 6.0 mm from its last setpoint
    (x +4.3, y -1.2, ~4 mm in z).
  - So -11 mm is the limit: do not go deeper. If the dish moves or scrapes,
    use -0.008.
  - Validated scoop on the new calibration:
    `--skill pick_up --params '{"object_name":"larger spoon"}' --skill scoop
    --params '{"powder_source":"baking soda cup"}'`, bench defaults as in
    `bench_manifest.py`.
- Dump in the same chain, before the user's verdict on it:
  - carry end 11.1 mm from setpoint (dx -2.7, dy -6.0, dz +9.0);
  - tip end 16.9 mm off (dx -16.5), "bowl 14.8 mm from over the target",
    tip 76.1° (the streamed impedance path lags);
  - `DUMP_FORWARD_OFFSET` is 0.
- **Stir after the reset (2026-10-10, no log saved).** The user: the rod
  "touches the bottom of the cup and moves the cup".
  - Cause traced from the code and the 2026-10-08 record
    (`agent_run_20261008_221001.json`).
  - On 2026-10-08, stir A/B felt the floor at TCP z **0.0803** (-1.65 /
    -1.07 N against the 1.0 N threshold) and stirred at 0.0843. The 1 mm
    probe steps started at `stir_z + fine_zone` = 0.0773 + 0.015 =
    **0.0923**, about 12 mm above the floor.
  - With table_z -0.0112 the cup model's base, the to_floor aim (5 mm under
    it) and the fine zone all moved 11.2 mm down. The fine zone then
    started at ~**0.0798**, below where the floor is felt, so the rod met
    the floor at the end of a 5 mm coarse step.
  - **Fix: bench stir default `fine_zone` 0.026** (`STIR_FINE_ZONE`; 15 + 11.2
    mm). The 1 mm steps start at ~0.0908 again. Not yet run on the arm.
  - Side note from the same record: the floor felt at TCP 0.0803 puts the
    CAD rod tip (`tool_length` 0.081) at -0.7 mm. The cup's inside floor
    should be ~-8 mm (table -11.2 + ~3 mm base), so the real
    `tool_length` is likely ~7 mm longer than the CAD. This does not matter
    for to_floor, which stops on force.
- **Stir with `fine_zone` 0.026 (18:26, `experiments/test_stir.log`).**
  - Probing now works as on 2026-10-08: coarse steps to ~0.092, then
    1 mm steps, contact at TCP z **0.0821** (-2.20 N), stir at 0.0861
    (4 mm back off).
  - **The rod still touched the bottom and moved the cup.** The user: "when
    it stirs the height is not the same".
  - The circle is streamed at 50 Hz under cartesian impedance. Each
    revolution ended 5.7 mm off its setpoint (dx -5.1, dy -2.6,
    dz -0.1). The log only shows the end of each revolution, not the z
    excursion mid-circle.
  - Today's streamed dump carry ended dz +9.0 mm off, so mm-level z swings
    on a streamed path are expected. With the tip ~3-4 mm above the felt
    floor, they reach it.
  - Options put to the user, since this is a control-mode change:
    - `smooth: false`: the existing blocking-waypoint path, position
      controlled (`goto_pose_rigid` defaults to `use_impedance=False`),
      12 points per revolution at 0.4 s. Constant height, slightly
      stepped. The scoop's position sweep tracked to 0.3 mm.
    - Keep streaming and raise `contact_backoff` 4 → 7 mm. The tip may
      then sit above a 4.4 mm-deep 10 ml.
  - **The user chose: keep streaming, 3 mm higher.** Bench stir default
    `contact_backoff` 0.007 (`STIR_CONTACT_BACKOFF`). to_floor applies
    max(param, 0.004), so 7 mm holds. Not yet run.
- **Stir at 7 mm, streamed (18:36): still touched the bottom.**
  - Contact at TCP z 0.0821 (-2.05 N), stirred at 0.0891.
  - Revolutions ended 6.3-6.8 mm off (dx -5.6..-6.1, dy -2.5..-2.7,
    dz +1.0..+1.2).
  - Side note: the CAD tip (0.081) read 35 mm below the rim at contact,
    while the inside floor is ~44 mm below it. So the real rod reach is
    ~9 mm longer than the CAD again.
  - **The user asked for position control: bench stir default
    `smooth` False** (`STIR_SMOOTH`): 12 blocking `goto_pose_rigid`
    waypoints per revolution, `use_impedance=False`, 0.4 s each.
  - Back off kept at 7 mm for this run to isolate the cause. If position
    control stops the touching, the wander was the cause and the back off
    can go down toward the liquid. If it still touches at 7 mm, look at the
    cup floor flexing under the 2 N contact, or a tilted rod.
- **Stir, position-controlled waypoints, 7 mm back off (18:37-18:40):** no
  longer touched the bottom (the user then asked for 4 mm again). So the
  touching was the streamed circle's height, not the floor or the rod.
  - The streamed circle is cartesian impedance. frankapy `goto_pose(dynamic=True)`
    forces `use_impedance=True` and uses `CartesianImpedanceFeedbackController`
    (`/home/rumi/frankapy/frankapy/franka_arm.py`), at
    `FC.DEFAULT_TRANSLATIONAL_STIFFNESSES` 600 N/m unless
    `cartesian_impedances` is given. The `use_impedance: False` in
    `dynamic_start_kwargs` has no effect there.
- **User: the descent too slow; the stir too slow, wanted faster and
  smoother. Bench stir defaults now:**
  - `coarse_step` 0.020 and `fine_zone` 0.020: 7 descent steps instead of
    ~17 (simulated for cup B: 0.1071, 0.0871, then 1 mm to 0.0821);
  - `contact_backoff` 0.004;
  - `smooth` False with `waypoints_per_rev` 16 and `seconds_per_waypoint`
    0.25 (was 12 at 0.4).
  - Not yet run.
- **User: keep the stream, fix its height drift (2026-10-10).**
  - Diagnosis: at 600 N/m, a few N of unmodelled joint friction is several
    mm of z. The friction changes direction as the circle goes round, so
    the height wanders; 7 mm of back off was not enough.
  - Change:
    - new stir param `stream_stiffness` passed to `stream_pose_path` →
      `cartesian_impedances`;
    - bench `STIR_STREAM_STIFFNESS` = [600, 600, **1500**, 50, 50, 50];
    - `STIR_SMOOTH` True again; back off stays 4 mm.
  - Why z alone: the 2026-10-08 all-axis 1500/150 shook because each 20 ms
    x/y setpoint step became a jolt. z's setpoint is constant on a circle,
    so a stiffer z has nothing to jolt on. Expected drift: ~1/2.5 of
    before.
  - frankapy confirmed to honour custom impedances in dynamic mode
    (`skill.set_cartesian_impedances`).
  - Not yet run. **If it shakes, e-stop and set `STIR_SMOOTH` False**
    (position waypoints, 16 x 0.25 s, still configured).
- **Streamed stir, z 1500 N/m, back off 4 mm (18:54,
  `experiments/test_stir.log`): no shaking, "轻微碰底" (light touching).**
  - Contact at TCP z 0.0801 (-1.79 N; 2 mm lower than the earlier 0.0821,
    a different cup spot), stirred at 0.0841.
  - Revolutions ended 5.6-5.7 mm off (dx -4.9..-5.0, dy -2.7..-2.8,
    **dz +0.4..+0.5**, against +1.0..+1.2 at 600 N/m).
  - The contact reads only after the rod presses in past 1 N, so the tip
    sits ~2-3 mm above the real floor at a 4 mm back off.
  - **Now z 2000 N/m and `contact_backoff` 0.005.** Not yet run.

**Point cloud presentation rewrite (2026-10-10): `scripts/view_pointcloud.py`.**
User asked for a denser, clearer cloud, each object outlined and labelled in
its own colour, ending in an interactive 3D view.
- **Capture:** cameras one at a time, 15 warm-up + 8 frames each.
  - Depth is the per-pixel median.
  - A pixel is dropped if it had depth in < 60 % of frames, or flickered
    more than 3 mm + 0.5 % of range (edge veils).
- **Cloud:**
  - stride 1;
  - flying pixels dropped where a 4-neighbour jumps more than 4 mm + 1.2 %
    of range;
  - exposure balanced per camera by median ratio over 1 cm table cells all
    cameras see. A plain per-view median was skewed by white paper:
    gains 0.5-2.0, against 0.95-1.08 on shared cells;
  - 1.5 mm voxel merge, statistical outlier removal, normals oriented up;
  - rendered lit.
- **Objects:** `vision.locate` for every bench object (or `--objects`).
  - Containers get a cylinder outline of the manifest's radius and height,
    drawn 3 mm outside and 2 mm above, or the points hide it.
    - Dishes: inside radius + 2 mm, 29 mm tall, centred by a rim-top
      circle fit.
    - Clear cups: silhouette centroid, plus a translucent shell.
  - Tools get an oriented box.
  - Scene points in or near each object take its colour; the rest are
    dimmed.
  - Labels are drawn on the PNGs by projecting anchors through the offscreen
    camera, with leader lines, no overlaps, plus a legend (name, label,
    size).
- **Outputs:**
  - `_raw.npy` (re-render with `--from-raw`, no cameras);
  - `.ply`;
  - perspective `.png` and `_top.png`;
  - `.mp4` with `--video`;
  - then an O3DVisualizer window, in a fresh process (the offscreen
    renderer leaves Open3D headless), with 3D labels. It falls back to
    `draw_geometries` without labels if the GUI fails.
- **Offline-tested only**, on a capture rebuilt from
  `pointcloud_20261010_170527.ply` (half resolution) with synthetic objects
  from clusters: renders correctly. The interactive window and the live
  capture/locate path are **not yet run** (my sandbox cannot open the
  cameras, and the window would open on the user's display).
- **First live run (19:09, `pointcloud_20261010_190900_*`): the user found
  the objects' shapes still bad.** The labelled O3DVisualizer did open.
  Analysis of the saved raw capture:
  - **Bug:** the flicker mask never applied, due to operator precedence:
    `seen >= N & (spread <= ...)` binds as `seen >= (N & ...)`. The log said
    "407040 steady pixels of 407040" on every camera, so the edge veils
    stayed in. Fixed with parentheses. NaN medians are zeroed.
  - The cameras run at **848x480** (robomail default). The D415's best depth
    is 1280x720. Capture now asks for 1280x720 (`--size`) and falls back to
    848x480 if refused.
  - **Points are the wrong representation here.** Four noisy views (±1-3 mm
    at 0.6-0.9 m, stereo-smoothed edges, mm calibration residuals) stacked
    as points give fuzzy shells and blobs.
    - **TSDF fusion** of the same raw (2 mm voxel, 8 mm truncation,
      `ScalableTSDFVolume`, 2.8 s) gave solid dishes with walls and rims,
      the black stirrer holder as one column, and a continuous table with
      sharp paper edges.
    - Now the default: `--mode mesh`. The `.ply` is the surface's vertices
      (330k); `_mesh.ply` is the surface.
  - **Clear cups and beaker cannot be measured:** they are absent from the
    TSDF too, except faint footprints. They now own no scene points (the
    "cup" blobs were table and paper seen through them). They are drawn as
    a translucent model shell (alpha 0.38) plus outline at their silhouette
    position.
  - The spoon stays a blob: small, white, flat, at 0.7 m. The stirrer
    holder was "not found" by locate on that run.
  - Fixes for the top view: labels sit on the objects (map style), the
    legend goes bottom-left, and label placement tries free spots in order
    instead of drifting.
  - Exposure gains for the mesh are taken at stride 2 (stride 4 left only 3
    shared cells).
  - Re-rendered offline from the 19:09 raw: clean. **A fresh capture with
    the flicker fix and 1280x720 is not yet run.**
- Further options, not done:
  - CAD meshes (the sim's spoon / stirrer STLs) placed at the located pose,
    for a true digital twin;
  - a close-range scan with the wrist camera (cam 1, `realsense_ee.tf`),
    if it is mounted.
- **Second live run (19:22, 1280x720, `pointcloud_20261010_192220_*`): "the
  transparent part is completely missing, even worse".**
  - **Cause, a new bug:** `get_live_intrinsics` cached by camera id only.
    The 1280x720 capture cached its intrinsics, and every later `locate`
    in the same process (848x480 captures) deprojected with them.
  - Result: 6 of 10 objects "not found"; the citric acid dish was placed at
    (0.438, -0.101), the beaker from one camera only. No clear cup was
    located, so none was drawn.
  - The TSDF itself was fine at 1280x720: dishes and holder solid.
  - **Fix (`object_localizer.py`):** the cache key is now
    `(cam_id, cam.W, cam.H)`. Tested with fake cameras (1280x720 and
    848x480 kept apart). Skills always run 848x480, so their behaviour is
    unchanged.
  - Clear cups are now drawn as **frosted open cups**: a wall and floor
    mesh, the object colour lightened, alpha 0.7, 4 outline verticals.
    The legend marks them "(model)" with one explaining line, and lists
    objects that were not found.
  - Re-rendered offline from the 19:09 raw: all four clear containers read
    as cups. **A fresh live run with the fix is not yet done.**

**Scoop rerun failed (2026-10-15 17:06, `experiments/test_scoop.log`): "Scoop
did not lift clear (off by 101mm)".**
- **Cause: the spoon pick, with `forward_offset` 0 (reset 2026-10-10).**
  - The jaws first held at **27.1 mm** (`gripper_is_grasped=False`), closed
    13.9 mm from the commanded grasp, and read **8.2 mm** by the start of
    the scoop. So the spoon slid or turned in the jaws after the lift.
  - This is the crank failure the manifest warned about: "The default
    forward_offset 0 lands on the crank, where the handle's section changes
    and they cannot seat".
  - On 2026-10-10 a pick at 0 happened to seat (8.2 mm) and passed.
- With the spoon moved in the grip, the tool offset measured at the pick
  (37.1 / 5.2 mm) no longer described it:
  - the compliant descent stopped 7.2 mm high (+4.5 after compensation);
  - the position sweep jammed at once: long axis 31.5° at the end, where
    it should be ~90°, TCP 29.6 mm off (x -22.0);
  - libfranka aborted with `cartesian_motion_generator_joint_acceleration_discontinuity`;
  - the lift and roll were refused ("did not move at all").
- Afterwards (read over ROS): `is_ready` True, no current errors, EE at home
  (0.307, 0.000, 0.487), the spoon still held at 8.2 mm.
- Proposed to the user, not yet changed: restore the spoon pick's
  `forward_offset` -0.010. `DISH_SCOOP_Z_OFFSET` -0.011 stays: it is at the
  limit, but it was not the cause.
- **User agreed: the spoon pick's `forward_offset` is -0.010 again**
  (`bench_manifest.py`). The offline bench-defaults test now expects
  -0.010. Not yet rerun on the arm.
- **Rerun with `forward_offset` -0.010 (2026-10-15 17:15,
  `experiments/test_scoop.log`): all four steps succeeded.** pick → scoop
  baking soda → dump into B → place back.
  - Pick: held at 19.6 mm (`gripper_is_grasped=False`), measured offset
    [0.045, 0.004, 0.021], about 10 mm more than at 0 as expected.
    Scoop start: 8.2 mm. The width still changes after the pick; it was
    harmless this time.
  - Scoop: descent stopped 6.2 mm high, then the position sweep completed
    (long axis 90.7°, TCP 6.0 mm off), as in the 2026-10-10 "perfect" run.
- Next: the whole pipeline with `--step`, given to the user. Not re-verified
  on the arm since the recalibration and offset reset: beaker pick
  (z_offset 0, was -12 mm), cup pick + pour, the dump landing
  (forward_offset 0, was -10 mm), and stir at back off 5 mm / z 2000 N/m.

**Whole pipeline with `--step` (2026-10-15 17:2x-17:31,
`experiments/agent_run_20261015_173140.json`, `pipeline_20261015.log`).**
- Steps 1-5 ok: pick spoon, scoop citric acid, dump into A, spoon back, pick
  stirrer.
- **Step 6, stir A, failed with "landing on the rim".**
  - The **first 20 mm coarse step** (0.1271 → 0.1071 in 0.4 s, in free air
    over A's opening) read **+1.00 N** against the 1.0 N threshold
    (baseline -1.58 N). That was a false trigger: the rod tip was 10 mm
    below the rim at the centre of a 36 mm-radius opening.
  - With a 20 mm `this_step`, the rim check (contact within one step below
    the rim) then called it a rim hit.
  - Afterwards: arm back over A at z 0.217, ready, no errors, stirrer held
    (31 mm).
- Fixes:
  - **`STIR_COARSE_STEP` 0.010**: 9 descent steps to a floor at 0.0821,
    against 7 at 20 mm and ~17 originally;
  - `_feel_down` waits 0.15 s after a coarse step before reading the
    force.
- **Found while testing:** the 2026-10-10 `stream_stiffness` change passed
  `stiffness=None` to `stream_pose_path` every time, which broke the
  offline test double (`fake_stream` has no such keyword). It was never
  caught because the suite was not rerun in perception_env. stir now passes
  it only when set.
  - `test_skills_offline.py` (perception_env): **424 passed**, the 2 old
    scoop workspace-floor failures.
- Resume from step 6 with the stirrer still in the jaws, **without
  `--reset`** (it would open the gripper).
- **Resumed at 6 (17:39-17:45, `agent_run_20261015_174505.json`): 6-8 ok,
  9 failed.**
  - **Stir fix confirmed on the arm:** 10 mm coarse steps plus the settle
    found A's floor at TCP z 0.0851 (-1.89 N) and stirred at 0.0901. 7
    (stirrer back, seated at 0.1061, -3.35 N) and 8 (spoon pick, held
    17.5 → 8.2 mm, offset [0.043, -0.000, 0.023]) were ok.
  - **9, scoop baking soda, jammed the same way as 17:06**, this time with
    the -10 mm grasp:
    - descent stopped 6.6 mm high;
    - sweep ended with the long axis at 34.9° (should be ~90°), TCP
      33.6 mm off;
    - `joint_velocity_discontinuity` abort, then "did not lift clear (off
      by 91 mm)".
  - Scoops at -11 mm on 2026-10-15: 17:06 jam (bad grasp), 17:15 ok, 17:2x
    (citric) ok, 17:4x jam. **-11 mm is past the margin.**
  - Proposed `DISH_SCOOP_Z_OFFSET` -0.008, pending the user's word.

**Experiment report written by the model after each run (2026-10-15).**
- `robochem/agents/reporter.py`:
  - `load_run_chain(record)`: a resumed record names its predecessor only
    by `resumed_from.started` (`log_path` is None), so predecessors are
    found by that timestamp among the records in the same folder.
  - `build_facts`: goal, parsed instruction sheet, bench, plan, and every
    executed step across sessions, with params, outcome, failure, seconds
    and `measurements(result)` (scalars, short strings, ≤3-vectors; poses
    and point sets dropped).
  - `write_report`: one `structured_completion` (strong tier, gpt-4.1,
    strict schema) with the facts plus the final photos, all four cameras.
  - The model writes the title, summary, objective, materials, procedure,
    observations (source photos/record), result vs expected, issues,
    conclusion and next steps.
  - **The header, the step table and the measured-values appendix are
    rendered from the record, not by the model.**
  - Output: `experiments/report_<time>/report.md`, `final_cam<N>.jpg`,
    `facts.json`.
  - Prompt rules: nothing beyond the facts and photos; colours only from
    photos ("not visible" otherwise); failure messages quoted, not
    reinterpreted; operator notes attributed.
- `run_experiment.py`: after every `--task` / `--instruction-image` /
  `--resume` run (not `--dry-run`), it photographs the bench
  (`vision.capture_scene()`) and writes the report. `--no-report` skips it;
  `--report-language Chinese` changes the prose language. A report error
  never fails the run.
- `scripts/write_report.py [RECORD] [--capture] [--language L]
  [--note TEXT ...]`: report on any finished run (default the newest).
  `--capture` photographs the bench now.
- Tests: `test_agents_offline.py` **101/101** in both venvs (new
  `test_the_report_covers_every_session`, which uses the FakeLLM double).
- Real model on today's two sessions (no photos, two operator notes):
  `experiments/report_20261015_175040/report.md`. It is factual and covers
  both sessions and both failures.
  - The first version paraphrased "did not lift clear" as "failed to lift
    the powder", hence the quote rule.

**Scoop jams: cause and fix (2026-10-15, after the user hand-recovered the
arm from the dish).** The arm had stood in the baking-soda dish from 17:45,
TCP (0.570, -0.190, 0.052), spoon held, interface ready.
- **Cause, from all five scoops at z_offset -11 mm:**

  | run | descent stopped high | sweep end | result |
  | --- | --- | --- | --- |
  | 10-10 18:19 | 6.5 mm | TCP **+6.0 mm** above the last setpoint, 90.7° | ok |
  | 10-15 17:06 | 7.2 | long axis 31.5°, 29.6 mm off | jam |
  | 10-15 17:15 | 6.2 | **+6.0 mm**, 90.7° | ok |
  | 10-15 citric (step 2) | 7.3 | **+5.7 mm**, 90.7° | ok |
  | 10-15 baking soda (step 9) | 6.6 | 34.9°, 33.6 mm off | jam |

  - Every finished sweep ended ~6 mm above its last setpoint: the plan ran
    ~6 mm below the dish floor, and position control pressed the bowl
    into it. The descent's 6.2-7.3 mm stop, against 2.7 mm free-air sag,
    says the same at the entry.
  - Whether the tip slid or pinned at the 60° entry decided ok or jam: 2
    of 5 jammed.
  - The user's "25 mm deeper" on 2026-10-10 was about 6 mm more than the
    floor allows.
- **Two bugs made a jam fatal:**
  - `goto_pose_rigid` returns True for a command sent, so the sweep logged
    "Swept 12/12" over refused waypoints;
  - the lift went out in franka-interface's recovery window, was refused,
    and the skill returned with the arm in the dish.
- **Fix:**
  - **`DISH_SCOOP_Z_OFFSET` -0.005**: the sweep ends on the floor, not
    6 mm into it. The bowl cannot collect anything below the floor, so the
    depth is unchanged in effect.
  - scoop: each position-sweep waypoint must end within `sweep_jam_tol`
    15 mm (jams were 30-34 mm off, finished sweeps 6), else it is a JAM.
    The scoop is then lifted straight out (`BaseSkill.lift_out`: waits for
    `is_ready`, up to 3 tries) and dug again `jam_raise` 4 mm higher,
    `jam_retries` 1 time. If it still jams, it fails with the arm out of
    the dish. The result carries `jams` and `dig_raised`.
  - Stage and rise lifts use `BaseSkill.goto_with_retry` (one retry after
    waiting for ready when the arm did not move), then `lift_out` before
    failing.
  - New `BaseSkill.wait_until_ready` (polls franka_interface_status; True
    off the robot).
  - Test `test_scoop_lifts_out_of_a_jam_and_digs_again` (JammingArm: pins
    after the descent, refuses while stuck). `test_skills_offline.py`
    (perception_env) **429 passed**, the 2 old scoop workspace-floor
    failures.
- Not yet run on the arm. Next: resume the 17:39 record from step 8 (spoon
  pick); the stirrer is back in its holder and the spoon on its spot.
- **Reports are written only on request (user, 2026-10-15).**
  `run_experiment.py` no longer photographs the bench or writes a report;
  `--no-report` and `--report-language` are gone.
  - `python scripts/write_report.py` finds everything itself:
    - `latest_record()`: the newest `agent_run_*.json` that moved the
      robot (dry runs and step-less records skipped);
    - the `--resume` chain;
    - `find_console_logs()`: every `experiments/*.log` modified since the
      experiment started;
    - photos taken now by default (`--no-photos` to skip). It warns when
      the run ended over 30 min ago, and the report states when the photos
      were taken (`facts.photos_taken`).
  - `console_excerpts()`: for each failed step, the skill's own console
    lines (`[Scoop]`, `[Stir]`, `[Safety]` ...) from the last block under
    `[step N] <sub-task>` that contains a failure. The `--step` prompt line
    is handled. They are given to the model and quoted in an appendix.
    On today's log it pulled 1.6 (the +1.00 N false contact) and 2.9 (the
    jam: 34.9°, 33.6 mm, the refused rise).
  - Options: `--language`, `--note`, `--no-console`, `--experiments`, or a
    record path.
  - `test_agents_offline.py` **103/103** in both venvs.
- **Resumed at 8 (2026-10-15 ~18:06-18:11, `agent_run_20261015_181120.json`;
  the user then deleted `experiments/`. It is in
  `~/.local/share/Trash/files/experiments/`, `pipeline_20261015.log`
  included): 9 jammed again at z_offset -5 mm.**
  - The descent stopped 5.1 mm high and was compensated to -0.6 mm.
  - The position sweep jammed. It was detected only at the +20° waypoint
    (18 mm off): the wrist had never left the 60° entry (tool z-axis
    (-0.87, -0.02, -0.49) when read afterwards), and the error grew
    gradually as the waypoints moved away. Detection is too late.
  - **All three `lift_out` tries were refused outright** ("did not move at
    all (0.0 mm)") while franka-interface reported ready.
    `robot_fault()`: robot_mode 1, last_motion_errors
    `joint_velocity_discontinuity`, no "not ready".
  - So it was not just the recovery window. A persistent refusal is the
    virtual-walls signature, or an error state that waiting does not
    clear. Asked the user what the franka-interface window said and
    whether the software lift worked during recovery. Not yet known.
  - The run wrote a report automatically because it was started before
    the auto-report was removed (18:08:02); Python had already loaded the
    old `run_experiment.py`.
  - Proposed, awaiting the user (a control change): the sweep and the
    lift-out under cartesian impedance, so floor contact is a bounded
    spring force rather than a libfranka abort.
- **The user's diagnosis (2026-10-15): the baking-soda dish was outside the
  arm's reachable area.** They hand-guided the arm out, and with the dish
  moved, the scoop works.
  - This fits the 18:11 log better than floor contact:
    - the wrist never left the 60° entry;
    - `lift_out` was refused with 0.0 mm three times while
      franka-interface was ready;
    - that is the arm at a joint/reach limit, where any motion needing that
      joint further is refused.
  - All three jams of the day were the baking-soda dish at y ≈ -0.19
    (the citric dish at y -0.08 never jammed). The same spot also
    succeeded twice, so it is the edge of the reach, not beyond it.
  - Consequences:
    - impedance-sweep proposal withdrawn;
    - `DISH_SCOOP_Z_OFFSET` -0.005 kept, since the +6 mm end error at -11
      was real floor pressure. -0.008 if the user finds the scoop too
      shallow;
    - jam detection and `lift_out` kept. `lift_out` cannot help at a joint
      limit: there, only hand-guiding works.
  - **Keep the powder dishes nearer the middle than y -0.19.** The
    validated reachable position is not recorded yet; take it from the
    next scoop log.

**Report writer looks for itself, always in English (user, 2026-10-15).**
- `llm_client.tool_completion`: an OpenAI tool-calling loop with strict
  `json_schema` output.
  - Tool results go back as text; their images follow in a user message,
    since a tool message can only hold text.
  - After `max_rounds` the tools are withdrawn and the answer is required.
- `reporter.BenchCamera(vision)`:
  - `photos(cameras)`: one capture per camera;
  - `close_up(name)`: `vision.locate` → project the object's points box into
    every camera → the two views with the largest whole box, padded 25 %
    and upscaled to 512 px.
- `write_report(records, camera=...)`:
  - starts the model with four overview photos and offers `look_at(object)`
    and `photograph_bench(cameras)` (8 rounds, 24 photos at most);
  - every photo it looked at is saved (`overview_cam*`, `closeup_NN_*`,
    `photo_NN_*`) and shown with a caption;
  - the calls are listed in "what the writer looked at" and in `facts.json`;
  - no language option: the prompt says English;
  - observations sourced "photos" must not mix in record inferences.
- `scripts/write_report.py`: cameras on by default (`--no-photos` writes
  from the records alone). `--language` is gone.
- Tests: `test_agents_offline.py` **107/107** in both venvs.
  - `ToolFakeLLM` snapshots each request; the caller appends to one message
    list.
  - `FakeBenchCamera`.
  - The model looks at cup a, then fresh views, then writes; the tool
    rounds are cut off; the no-camera path still works.
- **Real-API format check:** gpt-4.1 on the three trashed 10-15 records,
  with a stand-in camera returning point-cloud renders. It called
  `look_at` for cups a, b, c and the beaker, then wrote the report, so
  tools plus a strict schema plus image follow-ups are accepted.
  - Not yet run with the real cameras: needs the user's terminal; my
    sandbox cannot open them.

**Full pipeline 2026-10-15 18:19-20:27 (`experiments/agent_run_20261015_202701.json`,
the folder recreated after the user deleted the old one; no console log
kept): sub-tasks 1-16 ok, 17 failed.**
- **The first time steps 1-16 all ran in one session on the new
  calibration:** citric acid → A, stir; baking soda → B, stir; red cabbage
  scooped.
  - Dumps into A and B: 79° commanded, 75.8 / 75.9° reached.
  - Scoops at z_offset -0.005 with no jam: the citric descent sagged
    2.8 mm, i.e. no floor contact.
- **17, dump into the beaker: tip stalled at 54.6° of 59°, below
  `min_tip_deg` 55.**
  - The planner passed no `dump_angle_deg`, but that does not matter: the
    beaker, 0.573 m out, is reach-capped (`_wrist_reach`, PANDA_REACH 0.718).
    On 2026-10-08 it was capped at 64 / 66 / 70 / 67°. Today the cap was 59,
    since the beaker reads 6.5 mm further out and its rim 11 mm lower after
    the recalibration and table_z. The streamed tip then ends ~4° short.
- **The user: "54.6 is okay, don't set it too strict, keep the experiment
  going is the most important." `DUMP_MIN_TIP_DEG` is now 45** (the test
  reads the constant). Skills suite 429 passed, the 2 old failures.
- `dump_heading_deg` (sideways dump, said to reach 90° anywhere) is still
  untried on the robot; not used.
- After the failure the arm was home and ready, still holding the spoon
  (8.2 mm). Resume from 18 (spoon back), since the user judged the 54.6°
  dump good.

**Pour angles, the needless beaker put-down, and edits at the `--step`
prompt (2026-10-15, after the resumed run from 18).** Code and tests only;
not yet run on the robot.
- **What happened at sub-task 23:** the beaker poured into A at the
  planner's 80, not 60. The user had typed that it should be 60.
  - The planner had written 22 pick beaker, 23 pour A at 80, **24 place
    beaker back, 25 pick beaker**, 26 pour B. Its reason for 24: "the beaker
    must be put down before pouring into the next cup". No rule says that.
  - The old cap stopped looking at the first put-down, so it never saw the
    pour at 26.
  - Typed text at the `--step` prompt was ignored: anything but q ran the
    step unchanged.
  - 26 got `pour_angle=','` from the skill-call agent again; it was dropped
    and the default 90 applied, which is the right value for the last pour.
- **The user's rule, now in `_keep_some_for_later`:** "if the container needs
  to pour again, the first time 60, the second 90".
  - `_pours_from(plan, source)` follows the plan's pick-ups and put-downs
    (and "pour from X into ..." wording) to list every pour out of the held
    container.
  - With 2+ pours, every one but the last is set to 60 (`POUR_KEEP_SOME_DEG`)
    and the last to 90 (`POUR_LAST_DEG`), whatever the model asked for. A
    container that pours once is still left to the model; A into C was
    planned at 80 and is untouched.
  - The note names the later pour, e.g. `pour_angle 80 -> 60: plastic beaker
    pours again at sub-task 24, so this pour must not empty it`.
  - On the 18:19 plan it finds beaker pours [23, 26], so a `--resume` of
    that record from 22 or 23 also gets 60 then 90.
- **Needless put-down then pick-up, now removed (`drop_needless_regrasps`):**
  - When a "place X" is followed straight away by "pick up X", both are
    dropped and the plan is renumbered. Logged as `[plan] dropped sub-tasks
    ...` and kept in the record as `plan_edits`.
  - A put-down that takes X somewhere else ("at the centre of the bench")
    is kept, and so is one made to free the gripper for another tool.
  - It runs only on a fresh plan, never on `--resume`, because
    `--from-step` counts in the stored plan.
  - On the 18:19 plan it drops 24-25: 33 sub-tasks become 31, and the beaker
    pours at 23 then 24.
  - The pair moved nothing and cost a scan plus a grasp. The re-pick at 25
    closed 6.6 mm from the commanded grasp and reported
    `gripper_is_grasped=False` while holding at 37.7 mm.
  - The pour skill already goes home holding the full beaker before every
    scan and comes back upright, so two pours in a row are physically
    nothing new.
  - New chaining rule in `skill_catalog.CHAINING_RULES`: "a container that
    pours into more than one target stays in the gripper between its pours".
    The code guard does not depend on the model following it.
- **`--step` prompt (`scripts/run_experiment.py:confirm_step`):**
  - Enter runs the step; q stops.
  - `key=value` changes the call's parameters (`pour_angle=60`,
    `to_floor=false`, `target_container='clear cup b'`), prints old -> new,
    and asks again. Values are read as JSON.
  - Other text is "not understood" and asks again; it no longer runs the
    step.
  - The operator's change comes after the automatic cap, so it wins. It is
    recorded in the step's `params` and in `adjusted` ("changed by the
    operator at the step prompt: {...}").
- Tests: `test_agents_offline.py` **124/124** in both venvs. New:
  - put down for the stirrer between pours, still 60 then 90;
  - the put-down then pick-up pair dropped, plan renumbered 1-4;
  - a relocating put-down kept;
  - `parse_step_overrides`, and an operator change ran and was recorded.
  - A piped-stdin check of `confirm_step`: "60 degrees" re-asked,
    "pour_angle=60" changed 80 -> 60, Enter ran.
- **Same rule for a cup poured into twice (the user, after watching 29
  run: "in step 29 the degree should also be 60").**
  - In the live run, 29 (A into C) poured at the planner's 80. The run was
    already at 31, so this applies from the next run.
  - `_pours_into(plan, target)` lists the pours into a container, from the
    sub-task's `target_container`, else "into X" in its text.
  - When a target is poured into more than once, every pour into it but the
    last goes at 60 and the last at 90: A into C 60, B into C 90.
  - The source rule is checked first.
  - A pour that is alone on both counts is still the model's choice.
  - Catalog: the pour note now describes both caps. The "about 80 for a
    partial pour" advice for a lone pour is unchanged.
  - Replay of the 18:19 plan through `_keep_some_for_later`: 23 -> 60,
    26 90, 29 -> 60, 32 90.
  - After `drop_needless_regrasps` the numbers become 23 / 24 / 27 / 30
    with the same angles. The step numbers in the next run are 2 lower from
    24 on.
  - `test_agents_offline.py` **127/127** in both venvs, with A and B into C
    planned at 80/80 coming out 60/90.
- **How much of A and B should go into C (estimate only, nothing measured;
  the user asked, 2026-10-15):**
  - Retention by tilt, ideal liquid with no wall cling:
    - clear cup (47.3 tall, rim r 36.6, ~r 25 at the base by the measured
      taper) keeps up to 25 / 11 / 2.5 / 0 ml at 50 / 60 / 70 / 80 deg;
    - the beaker (r 22 x 60 mm cylinder, unmeasured) keeps 51 / 35 / 19 /
      7 / 0 ml at 50 / 60 / 70 / 80 / 90.
  - So in the next run: A gets ~15 ml of indicator (~25 ml total) and B
    ~35 ml (~45 ml). A at 60 puts ~14 ml in C and keeps ~11 ml.
  - In today's run the beaker went at 80: A got ~43 ml and B ~7 ml.
  - Chemistry model: charge balance with dissolved CO2 held at 15 mM, B
    poured whole. With the sim's scoop masses (citric 1.33 g, soda 1.19 g):
    - all the bicarbonate reacts at ~68% of A;
    - 56% of A (the 60 pour): pH 6.9, ~240 ml CO2;
    - 70-80%: pH 6.3-6.6, ~290-300 ml;
    - all of A: pH 5.6, ~320 ml.
  - With denser soda in the same scoop (citric ~1.2 g, soda ~1.6 g), all of
    A is about right (pH 6.7).
  - What decides it is the citric : soda mass ratio of the real scoops,
    ideally ~0.76 : 1 (3 NaHCO3 per citric acid). They have never been
    weighed. Weighing one big scoop of each would fix A's fraction.
- **A into C is now 65, the rest unchanged (the user: "change the A -> C
  degree to 65, dont change the rest").**
  - New `POUR_FIRST_INTO_SHARED_DEG = 65`. It is used only for the first pour
    into a target that is poured into again.
  - The beaker's first pour stays `POUR_KEEP_SOME_DEG` 60, and every last
    pour stays 90. The catalog's pour note says the same.
  - Replay of the 18:19 plan (after the drop): 23 beaker->A 60, 24
    beaker->B 90, 27 A->C 65, 30 B->C 90.
  - `test_agents_offline.py` 127/127 in both venvs.
  - From the estimate above, 65 should put ~19 of A's ~25 ml (~75%) into
    C. Not yet seen on the robot.

**Photos after every pour, for the report (2026-10-15).** Code and tests
only; not yet run on the robot.
- **Why:** the report said A turned pink-red, but it could not say B had
  turned blue. When it photographed the bench, B had already been poured
  into C and was empty. The only photos were taken at report time.
- **Run side:**
  - `AgentOrchestrator(snapshot=...)`: after every successful step whose
    skill is in `SNAPSHOT_AFTER` (`("pour",)`), it calls
    `snapshot(subtask, call)`.
  - The paths go in the step record as `photos`.
  - A snapshot that throws is printed (`[photos] none taken: ...`) and the
    run goes on.
- **`run_experiment.py`:**
  - `make_snapshot(vision, folder)` saves one JPEG per camera from
    `object_localizer.capture_pointclouds()` as
    `experiments/photos_<start time>/s<NN>_pour_cam<k>.jpg`.
  - On by default; `--no-step-photos` turns it off. Dry runs take none.
  - Each pour ends with `reset_joints` home, so the arm is clear of the
    cameras when the photo is taken.
  - The photo after the beaker's second pour (into B) is the bench just
    before A goes into C, with A red and B blue. That is why it is "after
    each pour" and not "before A into C": before a pour, the held cup hangs
    over its pick site in view.
- **Report side (`reporter.write_report`):**
  - The model is given FACTS, then RUN PHOTOS grouped by the step they
    follow ("after sub-task 24 (session 1): pour ..."), then today's
    overview.
  - Run photos are capped by `max_run_photos` (24, thinned evenly per step)
    and copied into the report folder as `run_sNN_*.jpg`.
  - Missing files are skipped, also looked for next to the records.
  - The prompt says a cup later poured out is empty now and its colour at
    each stage is in the run photos.
  - `facts.steps[].photos_after` gives the count.
- Tests: `test_agents_offline.py` **133/133** in both venvs. New:
  - photos only after pours; the paths are recorded; a failing snapshot does
    not stop the run;
  - run photos reach the model before today's views, and a missing file is
    skipped;
  - `make_snapshot` with a fake localizer wrote `s24_pour_cam2.jpg` and
    `s24_pour_cam3.jpg`.
- Not seen on the robot: how long the extra capture takes per pour. The
  skills' own scans open the four cameras in a few seconds.

**Robot runs 2026-10-19 (read back from the records and
`experiments/pipeline_20261015.log` on 2026-10-20).**
- **16:22 (`agent_run_20261019_163229.json`, 24-step plan):** 1-2 ok, then
  blocked at sub-task 3's skill call. The error was "SkillCallAgent: could
  not reach the model after 4 retries: Connection error."
  - Not the key: on 2026-10-20 the same key answered a gpt-4.1 call in
    4.2 s.
  - A missing key raises `MissingAPIKeyError`, and a bad key a 401; neither
    is retried.
  - "Connection error" is the network, and the old retries gave up after
    103 s.
- **16:35 (`agent_run_20261019_165344.json`, 31-step plan, no needless
  regrasp to drop):** 1-14 ok, then 15 (pick the larger spoon) failed with
  "Failed to lift object".
  - It closed 12.6 mm from the commanded grasp, at 16.7 mm
    (`gripper_is_grasped=False`).
  - The lift stalled at z 0.0756 against 0.1564 commanded (81 mm short,
    tol 35).
  - The spoon was at (0.725, -0.128), the farthest object on the bench, at
    the edge of the reach. The same pick at (0.728, -0.123) worked on the
    resume.
- **Resume from 15, about 16:54, pointed at the 10-15 record:**
  - The log says "the plan of 2026-10-15 18:19:28 ... running 15-33", so it
    ran that day's 33-step plan (with the needless put-down at 24-25), not
    the 31-step plan of the run it continued.
  - Sub-tasks 15-32 all ran.
  - Pours: 60 / 90 / 65 / 90 with `[adjusted]` notes, and 16 photos in
    `photos_20261019_165442/` (s23, s26, s29, s32 x cams 2-5).
  - **It ended at the step 33 prompt and wrote no record**, probably on
    Ctrl-C. So the report cannot see 15-32 or the photos.
- **Fixes (2026-10-20):**
  - `llm_client._BACKOFF` is 3/10/30/60/120/180/300 s, about 12 min in all,
    before a stage is reported blocked.
  - Ctrl-C at the `--step` prompt is a stop like q, so the record is
    written.
  - `--resume latest` takes the newest record in `--log-dir` that moved the
    robot (`reporter.latest_record`). It prints the record and the
    sub-tasks done there.
  - `test_agents_offline.py` 133/133 in both venvs. A patched `input`
    raising KeyboardInterrupt made `confirm_step` return False.
- **Full run, as of 2026-10-20** (agent pipeline; the scene agent reads the
  bench photos and the sheet with gpt-4.1):
  `python scripts/run_experiment.py --workspace-min 0.25 -0.40 -0.13
  --instruction-image instructions/magic_beaker.png --follow-steps
  --no-verify --max-retries 0 --reset --step 2>&1 | tee -a
  experiments/pipeline_<date>.log`
  - After a stop: the same command without `--reset`, plus `--resume latest
    --from-step N`.
  - `--legacy-planner` (the single-prompt `VLMOrchestrator`) is not the
    path that has run the Magic Beaker on the robot.

**OpenAI models available to this key (probed 2026-10-20).** Tiers come
from `.env` / env vars: `ROBOCHEM_MODEL_STRONG` (planner, skill-call agent,
report writer; default `gpt-4.1`) and `ROBOCHEM_MODEL_CHEAP` (scene agent;
default `gpt-4.1-mini`). No code change is needed to switch.
- `models.list()` returns 135 ids. The chat models include gpt-4.1/-mini/
  -nano, gpt-4o (API shutdown 2026-10-23), gpt-5, gpt-5-mini, 5.1, 5.2, 5.4
  (and -mini), 5.5, 5.6-sol/-terra/-luna, gpt-6-sol/-luna/-astra,
  gpt-6.1-sol, o3, and o4-mini.
- Probe (scratchpad `probe_models.py`): the sheet image plus a strict
  json_schema, then one strict-schema call with a function tool. Times are
  for the sheet call; "steps" is how many the model read (the sheet has 7).

  | model | temperature | time | steps | tools in chat.completions |
  |---|---|---|---|---|
  | gpt-4.1 | ok | 4.8 s | 7 | ok |
  | gpt-5.4 | ok | 5.6 s | 7 | ok |
  | gpt-5.5 | rejected | 11.9 s | 7 | ok |
  | gpt-6-astra | rejected | 14.7 s | **8** | **400: use /v1/responses** |
  | gpt-6.1-sol | rejected | 11.0 s | 8 | 400 |

  - gpt-5.6-* and gpt-6-* all refuse function tools on chat.completions.
    `reasoning_effort` 'none' and 'minimal' are not accepted (only
    low..xhigh), and 'low' is still refused.
  - The Responses API does return a `function_call` for gpt-6.1-sol.
  - A rejected temperature is harmless: `llm_client` drops it and retries.
- **gpt-6-astra (the user asked about it):**
  - Usable for the planner and skill calls (structured output, no tools):
    `ROBOCHEM_MODEL_STRONG=gpt-6-astra` in `.env`.
  - **`write_report.py` will fail on it** (tool loop on chat.completions).
    Until `llm_client.tool_completion` is ported to `/v1/responses`, run
    `ROBOCHEM_MODEL_STRONG=gpt-4.1 python scripts/write_report.py`.
  - About 3x slower per call than gpt-4.1, so roughly +5 min over ~33
    calls.
  - Plan quality vs gpt-4.1 is **untested**. A real-planner comparison
    script was written but not run (the user stopped it).
  - Suggested check before a robot run: the full command with `--dry-run`
    instead of `--step` (plans every step, no motion; still opens the
    cameras).

**gpt-6-astra real-cell dry run (2026-10-20): planned nothing.**
`ROBOCHEM_MODEL_STRONG=gpt-6-astra python scripts/run_experiment.py
--workspace-min 0.25 -0.40 -0.13 --instruction-image
instructions/magic_beaker.png --follow-steps --no-verify --dry-run`
- Run 1 (`agent_run_20261020_114231.json`): blocked at the goal stage.
  `InstructionParser` called chat.completions directly with `temperature=0`
  and got a 400; the `llm_client` temperature fallback never saw it.
  - Fix: `llm_client.chat_create(client, **request)` drops a rejected
    `temperature` or `max_tokens` and retries. `instruction_parser`,
    `label_resolver` (both calls) and `scene_analyzer` (both calls) use it
    now. The token cap is dropped rather than renamed, since on a
    reasoning model it also counts hidden reasoning. 133/133 offline.
- Run 2 (`agent_run_20261020_114405.json`): goal read (all 7 sheet steps),
  scene found all 10 objects, then **the planner returned 0 sub-tasks**,
  outcome `failed`. Its reason: no small scoop, no water warming, no speech
  skill, and "one large scoop each ... does not establish neutralizing
  proportions", so it would not claim the purple endpoint.
  - gpt-4.1 plans the same goal as 31-33 steps and has run it on the
    robot. gpt-6-astra is too literal for `--follow-steps` as the prompt
    stands. Stay on gpt-4.1 unless the planner prompt is told to
    approximate unmet details (doses, warming, speech) instead of refusing.
- **Planner prompt relaxed (2026-10-20, per the user):** a new autonomy
  rule in `planner._PLAN_PROMPT` — "approximate rather than refuse". It
  drops non-manipulation steps (speaking, the audience, presentation),
  assumes bench liquids are already at the sheet's temperature (the water
  counts as warm), uses the closest tool for a missing one or dose (one
  scoop with the larger spoon, big or small), and does not refuse because
  the amounts might not give the stated result. `feasible=false` only when
  an object the procedure needs is absent. 133/133 offline.
  - **Not yet tested on a model.** The rerun
    (`agent_run_20261020_115314.json`) hit a 429 `credit_balance_exhausted`
    at the instruction parser: the OpenAI account is out of credits. Rerun
    the astra dry run above once billing is topped up, and check that
    gpt-4.1 still plans the full sheet. (Done the same day, below.)
- **Run 3, credits added (`agent_run_20261020_120603.json`):** the relaxed
  prompt worked: astra planned 31 steps. Then sub-task 2 (scoop) came back
  INFEASIBLE: "the preceding pickup did not report suggested_tool_offset".
  A dry run never grasps, so nothing is measured. It replanned into 12
  sub-tasks with several skills each ("Run place ... Then run pick_up ..."),
  and the skill-call agent took one skill from each, so the calls were
  wrong (e.g. `place target_location 'larger spoon'` while the stirrer was in hand).
  - Fix: in a dry run `_with_offset` says "DRY RUN: nothing was grasped,
    no tool_offset measured, leave it out; on the robot the pick_up
    supplies it". The real path is unchanged.
- **Run 4 (`agent_run_20261020_121117.json`): success, 31 steps, 0
  replans.** 3x scoop/dump/stir and 4 pours, the same shape as gpt-4.1.
  Pours were adjusted by the existing logic (23: 80 -> 60, 27: 90 -> 65).
  It expects red in cup a, blue in b, and purple + fizz in c.
- **gpt-4.1 with the new prompt (`agent_run_20261020_121256.json`):**
  success, 31 steps, 0 replans. Skills: 9 pick_up, 9 place, 3 scoop,
  3 dump, 3 stir, 4 pour. So the prompt change did not hurt it.
- Neither model has run this on the robot with the new prompt yet. To try
  astra for real: `ROBOCHEM_MODEL_STRONG=gpt-6-astra` plus the full-run
  command with `--step`.
- Open items for the next session:
  - port `tool_completion` to the Responses API if gpt-6 models are kept;
  - first robot run with the 2026-10-15/20 changes (60/90/65/90 pours, the
    needless put-down dropped, photos after pours, `--resume latest`);
  - consider moving the spoon 3-5 cm toward the base (it stalled lifting
    at x 0.725).

**2026-10-20, robot run `agent_run_20261020_123353.json` (gpt-4.1, 27-step plan).**
- 1-4 ok (citric acid into A, then scoop baking soda). 5, dump into
  clear cup B: "Cannot locate 'clear cup b'" from the scan with the arm over
  the bench. A camera-only check from home a few minutes later found all
  three cups: B at (0.458, +0.125), read B by 3 of 4 cameras. The plastic
  beaker at (0.573, +0.018) was read "A 10 ML WATER" by 3 cameras but stayed
  unassigned.
- The user's first resume pointed `--resume` at the 10-15 record. That
  plan's step 5 is "pick up the stirrer" while the spoon was still in hand.
  Flagged to the user to stop it at the `--step` prompt. Use `--resume latest` or the record of the run being continued.
- `scripts/check_bench_layout.py` (camera-only identification, no arm) now
  loads `.env` itself. Without `source scripts/env.sh` it had died in
  `OpenAI()` with no API key.
- **Water cups grasped 6 mm further forward (user, 2026-10-20):**
  `bench_manifest.WATER_CUP_FORWARD_OFFSET = 0.006` is pick_up's
  `forward_offset` for clear cups A and B (C and the beaker unchanged).
  Checked in the manifest: A and B carry `forward_offset 0.006`. Not yet
  seen on the robot. `test_agents_offline.py` 133/133.
  `test_skills_offline.py` crashes on `FakePose` (`float()` of a fake pose)
  with or without this change, from earlier work. Not investigated.
- **Step 31 (pour B into C) refused twice:** "Cannot locate 'clear cup c'"
  (`agent_run_20261020_142834.json`, then the resume
  `agent_run_20261020_143135.json`). Scanned from home: cup A at
  (0.379, -0.111) read [A x1, C EMPTY x1, 50 ML x1]; cup C at
  (0.427, -0.024) read [A x1, C EMPTY x1]. Two share-outs tie, so the vote
  refused. C held red liquid by then.
  - Fix, `VisionSystem.last_seen` (bench name -> xy): set on every resolved
    vote. On a tie, take the tied candidate **closest** to the last-seen
    spot (the user's choice: closest, no distance threshold). With no
    last-seen spot, it still refuses. A resume seeds it from the record:
    pour `lip_xy` -> target, place `place_position` -> the placed object.
    Replaying `_143135` gives C (0.412, -0.025): 15 mm from the C candidate
    and ~100 mm from A's. 133/133 offline. Not yet seen on the robot.
  - Gap: a live (non-resume) run puts only vote results in `last_seen`,
    not place positions.

**`pick_pipette` skill (2026-10-21, `robochem/skills/pick_pipette.py`). Not yet run on the robot.**
- One skill: grasp the clear transfer pipette by its bulb (standing in the
  stirrer's black holder), 1 N touch to measure the bulb's relaxed width,
  squeeze flat at `squeeze_force` 20 N, lift 15 cm straight out of the
  holder, home, scan the source, feel down to its floor (stir's
  force-guarded descent; or `dip_depth` below the rim), back off 3 mm, open
  the jaws `release_amount` (5 mm) but never past relaxed width −
  `hold_margin` (3 mm), wait `draw_seconds`, lift clear. `dispense`
  (mode "dispense") then squeezes drops relative to that width.
- **Perception, camera-only (`scene_captures/pipette_20261021_164147`,
  overlays `diag_out/pipette_prompts/`):** SAM 3 `"pipette"` 4/4 cams (0.77),
  `"dropper"` 4/4 (0.82), `"plastic dropper"` 0.78, `"straw"` 0.64; `"bulb"`,
  `"pipette bulb"`, `"rubber bulb"` found nothing. Masks are the bulb +
  exposed stem, stopping at the holder mouth. **Clear plastic: cams 2 and 5
  read the table through it** (z ≈ −0.005, 200+ mm off); `VisionSystem.locate`
  consensus rejects them and keeps cams 3+4 (7 mm apart). Centroid
  (0.441, −0.013, 0.111), top z 0.137. Width profile: stem 2-7 mm at z
  0.100-0.110, bulb 9-15 mm from 0.112 to 0.137 — so the grasp is
  `top − grip_below_top` (15 mm), not the centroid (which is stem).
- **`tool_length` 0.11 is a guess** (top 0.137 standing on a bore floor near
  the table, TCP 15 mm under the top). Measure the pipette with a ruler.
  `fine_zone` is 35 mm so a longer real pipette still meets the floor during
  2 mm steps.
- Once released the jaws are in position mode, so a pipette that slid out
  is NOT detectable by width (`volume_measured` false). While squeezed (force
  mode) a drop shows as jaws shutting and is caught.
- Executor: pick site / held / pick_grasps recorded under `"pipette"`, so
  `place {"target_location":"pipette","vertical_insert":true}` should aim
  back at the holder (untested). Catalog entry added.
- Offline: `test_pick_pipette_squeezes_dips_and_releases_a_little` 11/11.
  Full suites not re-run (user: go to the robot). `test_skills_offline.py`
  currently dies in `test_transparent_cups_are_placed_by_silhouette`:
  `VisionSystem` has no `last_seen` (from the 2026-10-20 work), unrelated.

```bash
python scripts/run_experiment.py \
  --skill pick_pipette --params '{"source_container":"b 10 ml water","container_category":"clear plastic cup"}'
```
- **First robot run (2026-10-21), `pick_pipette` into `"a 10 ml water"`:
  success.** Cams 2+5 kept this time (3 rejected see-through, 4 score 0.29).
  Top z 0.1416, grasp 0.1266 (reached within 0.1 mm). Bulb **16.4 mm** at 1 N,
  squeezed to **1.9 mm** at 20 N. Cup A rim z 0.036; contact at TCP 0.1161
  (−4.35 N vs −3.15 N baseline), released 1.9 → 6.9 mm, held 6.8 mm.
  - User: it **lifted before the bulb had finished drawing**. Fix: poll the
    jaws until they reach `release_width` (≤5 s), then `draw_seconds` 3.0
    (was 1.5). Not yet re-run.
- **`dispense` `squeeze_out` (2026-10-21):** squeezes the bulb flat
  (force-limited, `squeeze_force` 20 N) over the target, tip
  `dispense_height` above the rim, and lifts still squeezed so nothing is
  drawn back. `container_category` added to dispense. The executor passes
  pick_pipette's `tool_length` to dispense (whose default 0.0 would put the
  tip at the jaws, i.e. 11 cm into the cup). Not yet run.

```bash
python -u scripts/run_experiment.py \
  --skill pick_pipette --params '{"source_container":"a 10 ml water","container_category":"clear plastic cup"}' \
  --skill dispense     --params '{"target_container":"b 10 ml water","container_category":"clear plastic cup","squeeze_out":true}'
```

**Jellyfish sheet (2026-10-21, `instructions/jellyfish.jpg` = `IMG_5545.jpg`, `--bench jellyfish`).**
- New manifest `JELLYFISH` in `bench_manifest.py` from
  `scene_captures/jellyfish_20261021_173701`: dishes baking soda / red
  cabbage powder, clear cups "2/3 cup of water" (`first clear cup`) and
  "20 ml water" (`other clear cup`), `beaker` on "vegetable oil" (dims NOT
  measured, Magic Beaker's), `pipette` standing through the lid of a white
  paper cup (`pipette cup`), larger spoon (also "medium scoop"), stirrer +
  holder. Skill defaults shared with Magic Beaker.
- Camera-only: `"pipette"` in the lidded cup found on **4/4** cameras (the
  white cup behind it gives real depth), centroid (0.658, 0.074), top z
  0.141; cup lid top z ≈ 0.085, so the grasp (TCP ~0.126) is 41 mm above the
  lid.
- Dry run (`experiments/jellyfish_dryrun_20261021.log`, gpt-4.1): 21
  sub-tasks; 18 = `pick_pipette` from `other clear cup`, 19-21 = `dispense`
  3 drops each into `first clear cup`. Failed at 20 only because a dry run
  never marks the pipette held (the scene still shows it in its cup). Not a
  real-run blocker.
- **First real run planned the oil first (`agent_run_20261021_173954.json`,
  gpt-4.1):** 1-3 pick beaker / pour into first clear cup / place, THEN the
  baking soda. The goal text it read listed the steps in the right order.
  Two causes: (a) the planner prompt said "reason about which reagents, and
  in which order" and "nobody will second-guess your chemistry", which
  outranks `--follow-steps`' "in order" inside the goal; (b) nothing said the
  "2/3 cup of water" cup was already filled, so step 1 ("fill the cup")
  became a pour, and the beaker was the only pouring vessel.
  - Fixes: planner autonomy rule "numbered steps keep their order exactly;
    a step the scene shows done gets no sub-tasks"; JELLYFISH notes say both
    clear cups ALREADY hold their water and the beaker holds ONLY the oil,
    poured at step 3 after the baking soda. Not re-run yet.
  - gpt-6-astra not tried: on 2026-10-20 it planned 0 sub-tasks, and
    `tool_completion` is not ported to the Responses API (skill-call agent).
- **Stirrer put-back failed in the jellyfish run (`pipeline_20261021.log`
  l.872-882):** holder at (0.688, −0.193). `vertical_insert` crossed in XY at
  the home height z 0.4865, then "Hover ... unreachable (off by 186mm)" —
  exactly 0.4865 − 0.30, so the arm never came down from the crossing at
  that reach. The pick's 0.30 hover over the same holder worked. Fix (user's
  request): `place` now crosses at the hover height (`safe_height` 0.30)
  by default; `transit_height` overrides. Not yet re-run. Test chain:
  `--skill pick_up stirrer` + `--skill place {"target_location":"stirrer","vertical_insert":true,"release_clearance":0.0,"stop_force_n":1.0}`.
- **Beaker picked with the fingers 10° away from the base (user, 2026-10-21).**
  `pick_up` `tilt_away_deg`: rotates the grasp about world Y so tool Z goes
  from straight down to down-and-+X; the beaker stays upright. pour's
  `_rotation_lean` leans tool Z toward the base (−X) by the pour angle, so
  the beaker ends up pour_angle + 10° over: a wrist that stalls at 80° still
  pours 90°. JELLYFISH `beaker` pick carries `("tilt_away_deg", 10.0)`
  (Magic Beaker unchanged). First sign was wrong (tool Z [−0.174, 0, −0.985]);
  caught by a one-line check, fixed to [+0.174, 0, −0.985].
  - `place` (non-insert): when the pick's tool axis (`pick_tool_z`) differs
    by >2° from the current wrist (pour leaves it straight down at home), it
    goes back to the pick's tilt so the beaker is set down upright. Neither
    is robot-tested.
- **Pipette put-back refused (`agent_run_20261021_184009.json`, step 20):**
  "holding an unidentified object, not the pipette". The orchestrator's
  `_apply_effect` set `held_object` only for `pick_up`, so after
  `pick_pipette` + `dispense` it described the jaws as "something
  unidentified (jaws at 7 mm)". Fixed: `pick_pipette` sets `held_object`
  (also on `--resume`, which replays through `_apply_effect`). Resume that
  run with `--resume latest --from-step 20`.

**Dump: wrist twist after the shake (user, 2026-10-22).** New `dump` params
`twist_deg` 15, `twists` 2, `twist_seconds` 1.0 per cycle: after the tip-back
shake, at the dump angle, the wrist rolls about tool Z by
15°·sin(2π·twists·t/T) — +15, 0, −15, 0, twice in 2 s — as one streamed path
at 50 Hz, the TCP moved so the bowl centre stays fixed over the cup (the bowl
is ~44 mm out along tool X, so a roll about the TCP would swing it ~11 mm).
Falls back to quarter-cycle blocking poses; a failed twist carries on. Skipped
if the shake already sent the arm home. `twist_deg` 0 turns it off. Logged as
`bowl_drift_mm["twisted"]`. Not yet run on the robot (joint 7 may be near
its limit at the dump pose; watch for a refused stream).

**Stir: one continuous stream, faster (user, 2026-10-22).** The smooth path
streamed each revolution as its own min-jerk skill, so the rod eased to a
stop after every circle (2026-10-21 jellyfish: 3 × 4.0 s, 226 setpoints
each). Now `_continuous_circle_path` puts all revolutions in ONE streamed
path: angular speed eases up over the first 0.4 rev (spread over the
revolutions), holds, eases down at the end, and ends exactly at the start
angle. `seconds_per_revolution` 4.0 → **2.5** (steady pace ~2.2 s/rev; 3 rev
= 7.5 s, 376 setpoints at 50 Hz). The hold check now runs once after the
stream, not per revolution. Bench stiffness `[600, 600, 2000, 50, 50, 50]`
unchanged. Not yet run on the robot: at 1.6× the speed watch for the cup
being dragged or z drift (the 2026-10-10 rod-touch issue).

**Scans reused when nothing moved; pour/dump checked afterwards (2026-10-22). Not yet run on the robot.**
- **Label reads run in parallel** (`LabelResolver.read_all`, 8 at once).
  Measured offline on `scene_captures/jellyfish_20261021_173701` with
  gpt-4.1: 25 clear-cup instances took **25.5 s** one after another and
  **3.0 s** in parallel, with the same label on 24 of 25. SAM on 4 cameras
  took 1.4 s, so the label reads were most of a labelled-container scan.
  Every pick, pour, dump, stir and scoop of a labelled container saves about
  22 s.
- **Scan reuse** (`VisionSystem.scans`, `_reuse_scan`, `forget_scans`).
  - A full locate keeps the frames it came from, plus the object's box in
    each camera (its cloud's 2-98% corners projected).
  - The next locate of the same object first takes ONE frame from the
    camera that sees it biggest. If `region_change` in the box is at most
    `reuse_max_change` 0.05, the kept result is returned. Otherwise it
    rescans.
  - `region_change` is the fraction of pixels whose grey level moved by
    more than 25 after each crop's mean is removed.
  - `SkillsExecutor._forget_touched` drops the scans of what a skill
    handled (`_TOUCHES`):
    - pick_up / pick_pipette: the object, and pick_pipette's source.
    - place: the held object and `target_location`.
    - scoop / arc_scoop: the source.
    - stir: the target.
    - dispense: source and target.
  - A failed step, or a skill not in the table, forgets every kept scan.
  - pour and dump touch nothing, but the target looks different afterwards,
    so the frame check rescans it (unless the step check below refreshed
    the reference).
  - Each full scan now prints its own time (`full scan for 'x': N s`) so
    the next run shows where the time goes.
  - Turn it off with `--rescan-always`.
- **Threshold evidence.**
  - Offline, cam 3 of the jellyfish capture: Gaussian noise σ=3 → 0.000;
    the whole frame 20 levels brighter → 0.000; shifted 3 / 5 / 10 px →
    0.098 / 0.146 / 0.241.
  - Real frames minutes apart (`photos_20261019_165442`, s23 vs s26 cam 3):
    the untouched middle cup scored **0.001**, and the cup that was filled
    in between scored 0.864.
  - The 0.05 threshold has not been checked on the bench against a cup
    nudged by 5 mm.
- **Step check after pour and dump** (`orchestrator/step_checks.py`,
  `StepChecker`).
  - The "before" is the kept scan's frame. The "after" is one new frame
    from the same camera.
  - A dump that did not end at home is homed first.
  - Both crops are shown to the strong model, which answers
    `contents_changed` and `container_moved_or_tipped`.
  - Verdicts: `ok`, `suspect` (no change or moved), `unclear`, `skipped`.
  - Each verdict is printed as `[check]` and stored as the step record's
    `check`. The side-by-side photo is written to
    `experiments/checks_<time>/`.
  - Default is report-only (keep the run going). `--strict-checks` makes
    `suspect` fail the step; `--no-step-checks` turns the check off.
  - A cup that did not move gets the after frame as its new reference, so
    the stir that follows a dump can reuse the scan.
  - On the s23/s26 photos: the back cup that turned purple → `ok` (2.2 s);
    the untouched middle cup → `suspect` (1.1 s).
- Offline: `test_agents_offline.py` 133/133. All changed files compile
  under py3.8. A scratchpad test with a fake camera checked the
  reuse / move / touch / failure flow. `test_skills_offline.py` was not run
  (user: go to the robot; it was already crashing on `last_seen` before
  this change).
- **To watch on the first run:**
  - Do `unchanged ... reusing it` lines appear for the second skill on the
    same cup?
  - Does a hand-moved cup print `changed ... full scan`?
  - Does a dump of white baking soda into clear water read as
    `contents_changed: yes` from one camera?

**Four-item chain, first try (2026-10-22 16:3x, `experiments/chain_fast_20261022.log`, `--bench jellyfish`).**
- Bench: larger spoon, baking soda dish, `other clear cup` ("20 ML WATER",
  red cabbage water), stirrer in its holder. Capture
  `scene_captures/four_items_20261022_162033`.
  - `capture_scene.py` (15 warm-up frames per camera, process start
    included) took 29.5 s for four cameras.
- Timing:
  - pick_up spoon: 57.1 s (full scan 11.6 s).
  - scoop baking soda: 68.0 s (full scan 15.3 s). It was 80-86 s in the
    10-21/10-22 agent runs, which also include the skill-call model time.
  - dump: full scan 13.7 s, with 19 clear-cup instances across 4 cameras
    and parallel label reads. 4/4 cameras read the cup as `20 ML WATER`.
- **The dump stopped at step 3.** The cup was at (0.663, −0.009), and
  straight ahead the arm reached only 39°, short of `min_tip_deg` 45. The
  error itself named the fix (`dump_heading_deg` 60).
  - Fix: `dump` now tries `auto_headings_deg` [60, −60, 90, −90, 30, −30]
    when straight ahead cannot reach `dump_angle_deg` and no heading was
    given. It takes the first heading that reaches the full angle, or else
    the best one. Each is printed as `[Dump] sideways +60°: can tip to N°`.
  - `rotation_at` now reads `heading` at call time; its default used to be
    frozen at definition.
  - **A sideways dump has never run on the robot.** The wrist turns ~60° on
    the way over at 0.66 m: watch for a joint-limit fault.
- **`--holding NAME [--held-offset X Y Z] [--picked-at X Y Z]`** on
  `run_experiment.py` seeds the executor (held tool, pick's measured
  offset, pick site) so a stopped `--skill` chain can go on with the tool
  still in the jaws. Restarting from `pick_up` opens the jaws and drops it.
- The `--skill` chain now runs the step check after pour/dump and prints a
  per-step timing table at the end (also when it stops).

**Preflight: check every target's reach before the arm starts, move cups that fail (2026-10-22). Not yet run on the robot.**
- User's decisions:
  - Do NOT change how dump dumps. The auto-sideways heading added earlier
    the same day was reverted; `dump.py`'s motion is as before (the
    wrist-twist change stays).
  - Move a cup that is out of reach, without asking, even with liquid in it.
  - Do not put moved things back.
- `robochem/orchestrator/preflight.py` `Preflight`. It runs before step 1,
  with the gripper empty, in both the `--skill` chain and the agent
  pipeline (`record["preflight"]`). Turn it off with `--no-preflight`.
  1. Homes, then scans every target the steps name. The scans are kept, so
     the skills that use them reuse them.
  2. Checks each target the way its step will use it:
     - dump: `dump.reachable_tip_deg`, a pure copy of the dump's own reach
       check;
     - pour / stir / dispense: distance from the base in `TARGET_REACH`
       0.30-0.60 (a guess);
     - scoop dish: report only. 102 mm is wider than the jaws open.
  3. A failing cup (manifest radius <= 40 mm, "cup" category, gripper empty)
     gets `pick_up`, then `place` at the nearest free spot:
     - 0.35-0.50 m from the base, x >= 0.30;
     - dump reach >= 75° there (for a dump target);
     - >= radius + 60 mm from every occupied 1 cm cell of the last full
       capture's depth (>= 4 points, z 8 mm-0.35 m above the table);
     - >= 2·radius + 60 mm from every clear cup a label vote placed
       (`VisionSystem.seen_containers`; depth misses clear cups);
     - >= 0.18 m from any dish the run scoops from.

     The pick site and the pick grasp TCP are shifted to the new spot
     before `place`, because place puts a slid-in cup down at the held TCP.
- **`reachable_tip_deg` matches the robot.** For the 2026-10-22 cup
  (0.6635, −0.009, rim z 0.0361, offset [0.0406, −0.0037, 0.0294]) it gives
  **39.0°**, which the dump measured.
  - With offset [0.045, 0, 0.0294] it gives: r 0.40/0.45 → 79,
    0.50 → 78, 0.54 → 68, 0.58 → 58, 0.62 → 48.
- **Offline run on the real capture** `scene_captures/four_items_20261022_162033`
  (live intrinsics, real SAM 3 and label reads; homing and the move were
  stubbed):
  - 2 targets, 7.7 s.
  - Baking soda dish at 0.50 m: ok.
  - `other clear cup` at 0.663 m: "a dump can tip only 39°; 0.66 m from the
    base", so it is moved to (0.440, +0.230), 33 cm away.
  - Occupancy: 559 cells. They are the dish, the spoon, the stirrer holder,
    two cage rails at y ≈ ±0.5, and the far bench items. The dish's 18 cm
    ring pushes the spot to its side.
  - The vote also placed a "clear cup" on the dish; it only adds an
    obstacle.
- `SkillsExecutor.skill_instance(name)` was factored out of `execute`.
  `go_home` uses it.
- `test_agents_offline.py` 133/133; py3.8 compile ok.
- **To watch on the robot:**
  - Does `pick_up` of the full 20 ml cup and `place` 33 cm away spill?
  - Does it land where the `[preflight] free spot` line says?
  - Does the dump then print a full tip with no "can only tip" NOTE?

**USB wedge again, at the first preflight run (2026-10-22 ~16:43, pid 4062678).**
- All four cameras: `RuntimeError: Frame didn't arrive within 5000`.
  `dmesg`: `uvcvideo 2-3/2-4/2-5/2-6 ... Non-zero status (-71) in video
  completion handler` and `Failed to query (GET_CUR) UVC control 1 ... -32`.
  `lsusb` still listed all four D415s (bus 2). This is the shared-controller
  wedge from Known issues: the fourth time.
- Earlier the same afternoon the cameras had worked:
  - `capture_scene.py` at 16:20;
  - the first chain run at ~16:3x;
  - a live-intrinsics read (open, one frame, stop each camera) shortly
    before this run.
- The arm only homed (the preflight's go_home). Nothing was moved.
- `Preflight.run` now stops (`blocked`) when a target is not found and the
  capture had no frames at all. Before, it would go on to step 1, which
  could not see either.
- Fix: replug the camera hub, ideally onto the unused ASMedia (`04:00.0`)
  or NVIDIA (`01:00.2`) controller ports.

**Preflight on the robot, second try (2026-10-22 ~16:47, `experiments/chain_preflight_20261022.log`).**
- The cameras were back after the replug.
- Baking soda dish (0.494, −0.012): read 4/4, ok.
- **The 20 ml cup was misidentified.**
  - The `20 ML WATER` paper was no longer under the purple cup. It was on
    the next table, under another cup at about (0.99, −0.37), as seen in
    `scene_captures/label_check_20261022_164817` cam 5.
  - The near cup (0.632, −0.009) was placed by 4/4 cameras with no reads.
  - The vote gave `20 ML WATER` to the far cup, which only cam 5 placed. The
    preflight then judged it out of reach (1.06 m) and tried to fetch it.
  - The hover was clamped to x 0.75 and missed by 102 mm. pick_up gave up,
    and the preflight blocked with the arm left over the bench at about
    (0.67, −0.31, 0.31).
- Fixes:
  - `VisionSystem.on_bench`: x >= 0.15 and r <= `BENCH_MAX_REACH` 0.82.
    `_locate_by_vote` drops containers placed outside it before the vote
    (printed as `ignoring N ... beyond the arm's reach`).
  - Preflight does not fetch beyond `MAX_FETCH_REACH` 0.72 m. It homes
    after a failed move, and blocks before the arm starts when a target is
    not found, naming the label to check.
  - Offline on `label_check_20261022_164817`: 9 bowls and 7 clear cups
    ignored, baking soda ok, `other clear cup` not found, 7.5 s.

**A cup the arm moved is found where the arm put it, not by its label (2026-10-22).**
- The user's point: the preflight moves a cup, and its paper label stays
  where it was. Found by label afterwards, the "cup" would be the old paper.
- `VisionSystem.tracked` (bench name -> xy):
  - `SkillsExecutor._forget_touched` sets it from `place`'s
    `place_position` for the object that was held, and clears it on
    `pick_up` / `pick_pipette`.
  - `_locate_by_vote` first takes the placed container within
    `track_tolerance` 50 mm of that spot, with no label reads.
  - With nothing there (moved by hand?) it prints a warning and reads
    labels as before.
- Offline on `label_check_20261022_164817` (no paper under the cup):
  - Tracked at (0.632, −0.009): found by position, 0 mm off, **1.4 s**
    (a labelled scan was ~13 s).
  - Tracked at an empty spot: fell back to labels, and not found.
- The FIRST time a run looks for a labelled cup, it still needs its paper.
  A "the only cup of its kind within reach" rule was offered to the user,
  not added.
- `test_agents_offline.py` 133/133.

**Preflight moved the cup; the dump then failed on the carry (2026-10-22 ~17:0x, `experiments/chain_preflight_20261022.log`).**
- Before this run: the arm had been parked in franka-interface's virtual
  walls by the earlier failed fetch. The user guided it out by hand.
- Preflight (118.9 s):
  - The cup was at (0.674, −0.011), with a 37° reach.
  - pick_up → place at (0.428, +0.233).
  - Tracking worked: the dump found the cup "where the arm set it down,
    1 mm off; labels not read".
- pick_up spoon 57.3 s; scoop 58.8 s, ok.
- **dump: "Could not bring the bowl over 'other clear cup' (off by 28mm)".**
  The carry ended 25 mm high and 8° off. Error flags were already cleared
  by the reset_joints afterwards. Arm home, joints
  [0, −45, 0, −135, 0, 90, 45]°.
- **Why: the chosen spot was to the side.** Every real dump record
  (`agent_run_202610*.json`):
  - 13 full dumps (76°) stood at x 0.388-0.458, |y| <= 0.138.
  - 2 more at (0.57, 0.02-0.05) were capped at 56-58°.
  - Both side targets failed on the carry: (0.305, −0.207) on 10-07 with
    joint_position_limits_violation, and today's (0.428, +0.233).
  - `reachable_tip_deg` checks wrist reach only, not joint limits.
- Fixes:
  - A moved dump target must stand in `DUMP_SPOT_X` 0.39-0.46,
    `|y| <= DUMP_SPOT_MAX_ABS_Y` 0.14.
  - `DISH_CLEARANCE` 0.18 → 0.17 (sim: hits at 144-161 mm, clear at 172).
  - If no spot is free and the dump would refuse (< 45°), the preflight
    blocks before the arm starts and says to make room by hand.
  - A failed home in the preflight now blocks. It used to let the chain go
    on to a pick_up that was refused the same way.
- Offline on `four_items_20261022_162033` (same layout): blocked, "no free
  spot in the dump box". The box is crowded by the dish in the middle
  (17 cm ring), the spoon at (0.47, +0.10) and the stirrer holder on the
  other side.

**Scoop jammed with the dish pushed out to x 0.642 (2026-10-22 ~17:2x, `chain_preflight_20261022.log`).**
- The user moved the dish out on my advice ("~60 cm"), to make room for
  the dump box. Preflight then moved the cup from (0.614, +0.197) (45°) to
  (0.439, −0.005), inside the dump box: that part worked. 118.6 s.
- scoop: jammed twice at +55°, "TCP 35/32 mm from its waypoint",
  `cartesian_motion_generator_joint_velocity_discontinuity`. The arm was
  left 60 mm above the dish, holding the empty spoon.
- From the logs (`experiments/*.log`, dish centre against the outcome):
  - 12 scoops worked with the dish at x 0.500-0.602 (r up to 0.63).
  - Both jams were at (0.604, 0.183) on 10-21 and (0.642, −0.028) today.
  - The "~60 cm" advice was 4 cm too far.
- Preflight: `SCOOP_MAX_X` 0.60 for a dish being scooped, which blocks
  before the arm starts (a dish cannot be moved by the arm). The old
  check was r within 0.30-0.65, and it passed this dish.
- **Why the zones are tight:**
  - Scoop (60° nose-down) and dump (79°) need the wrist past the target,
    so the tilted reach ends near 0.60 m, against the arm's 0.85.
  - Both skills keep the wrist heading as at home (straight ahead). A
    target well to the side asks a joint for more than it has. Turning the
    heading toward the target would widen this, but that changes the
    dump/scoop motion, which the user does not want.

**Can the cell see the powder surface? Not used today, but the data has it (2026-10-22, offline).**
- **Today's scoop ignores the powder.**
  - `surface_z` is the dish cloud's 97th-percentile z, i.e. the rim (the
    scoop's own comment says so).
  - The default depth mode is `tip_floor_gap`: the tip goes 3 mm over the
    inside floor whatever the bed. A falling surface therefore matters
    less than the fixed stroke line. Every scoop runs the same x −29 →
    +22 mm through the dish centre, so it digs one trench, and powder left
    to the sides is never reached.
- **Measured from the dish's located points** (`powder_surface.py` /
  `powder_map.py` in that session's scratchpad, live intrinsics,
  baking-soda dish, 29 mm tall, nearly full):
  - `four_items_20261022_162033`, before any scoop, cams 2/4/5: median
    height per 5 mm ring from the centre is 21.5-24.5 mm above the table
    at every radius 0-55 mm. The bed is flat to ±1.5 mm. Single points
    spread ±4 mm (p10-p90 ~8 mm).
  - Against `label_check_20261022_164817`, after one scoop (cams 2-5, cam
    3 partly behind the arm), on 1 cm cells:
    - −3 to −6 mm in a patch through the middle (the stroke);
    - +4.7 to +5.5 mm just beyond its far end (powder pushed forward);
    - ±2-3.5 mm elsewhere, which is noise plus a 9 mm difference in the
      centre between the two captures.
  - One scoop shows at about 2× the noise.
- **Not built.** Next would be a dish heightmap at every scan, the stroke
  placed on the highest/fullest strip (lateral offset across the dish),
  and maybe the depth set from the measured surface.

**Scoop picks the fullest line through the dish (`stroke_across`, 2026-10-22). Not yet run on the robot.**
- User asked for the scoop to follow the powder. Agreed scope: move only
  WHICH LINE the stroke runs on. The bite angle, the roll, the 3 mm floor
  gap and the push are unchanged.
- New params:
  - `stroke_across` (default `"auto"`, or metres): moves the stroke across
    its direction. The wall check (`room()`) adds it to every corner's
    across coordinate, so an off-centre line gets the shorter chord. Old
    `lateral_offset` still moves the whole dish model, as before.
  - `stroke_across_step` 6 mm.
  - `stroke_min_travel_frac` 0.75 (a line must keep 75% of the centre
    travel).
  - `stroke_prefer_fuller` 2 mm.
- **How auto chooses** (`ScoopSkill._choose_line`). For each line the bowl
  fits, it takes the median z of this scan's dish points:
  - in a strip the bowl's width + 3 mm each side,
  - inside `reach_limit`,
  - and only along the bite zone (entry − 10 mm to 60% of the travel).

  The whole-line median was tried first and lost. On the real
  before/after pair it read the trench as flat: the stroke shoves powder
  forward and heaps it at the far end, which cancels the dip. The fullest
  line wins only if it is >= 2 mm above the centre; otherwise the
  validated centre line runs.
- After a scoop the executor forgets the dish scan (scoop touches it), so
  the next scoop rescans and sees its own trench.
- Offline, the synthetic dish (`test_scoop_line.py`, scratchpad): 45.5 mm,
  bed 22 mm, ±3 mm noise.
  - Flat bed: centre, lines within ±0.1 mm.
  - 8 mm trench along the centre: line −12 mm, 3.9 mm fuller, travel
    40 mm (centre 52).
  - Forced +6/+12/+18 mm: travel 47/40/29 mm, all 8 mm from the wall.
  - Forced +30 mm: refused (does not fit).
- Offline on real dish clouds (live intrinsics):
  - `four_items` (before the 16:3x scoop; the dish had been scooped in the
    14:52 run): +12 mm is 3.6 mm fuller, so line +12.
  - `label_check` (after one more scoop): all lines within 1.2 mm, so the
    centre.
  - The two scans' dish centres differ by ~9 mm (perception's centre leans
    toward the cameras), which is the size of a one-scoop trench. So this
    pair cannot confirm or refute the choice; several scoops on the robot
    will.
- The log's "comes within N mm of the inside wall" now includes the line
  offset.
- Existing scoop tests: all pass except "scoop refuses a dig depth below
  the workspace floor" (+ its message check). That one fails identically
  with HEAD's scoop.py: the old pair.
- Watch on the robot: the `[Scoop] Powder on each line ...` and `Stroke
  line ...` lines, and whether an off-centre stroke jams (it runs nearer a
  side wall).

**pick_up turns the wrist to a spoon lying at any angle (`yaw`, robot-tested 2026-10-23, `experiments/yaw_test_20261023.log`).**
- Before this, every top grasp used the home rotation
  `diag(1, −1, −1)` (`grasp_analyzer._compute_top_grasp`): the jaws always
  closed along base Y. `forward_offset` −0.010, which moves the jaws off
  the crank, ran along base X, i.e. not along the handle unless the spoon
  lay straight ahead.
- **New `pick_up` param `yaw`.**
  - `None`: as before.
  - `"auto"`: `tool_long_axis_yaw()` takes the principal axis of the
    cloud's XY. The bowl end is the wider of the two outer 30% ends (95−5
    percentile across the axis). It refuses when the cloud is not
    elongated (axes < 2:1) or the ends are within 1.3× in width.
  - A number: that many degrees.
  - The grasp rotation becomes Rz(yaw)·home, so tool X points at the bowl.
    Once homed, the spoon sits in the jaws exactly as a straight-ahead one
    does, and scoop/dump need nothing new.
  - `forward_offset` / `lateral_offset` are rotated with it (along / across
    the tool).
  - Turns past `MAX_GRASP_YAW_DEG` 120 (bowl toward the base) fall back to
    no turn, keeping joint 7 (45° at home, ±166°) clear.
  - Not used for side or slide-in grasps. The result reports `grasp_yaw`.
- **Robot run:**
  - The spoon was at (0.6133, −0.1216). The log read "Turning the wrist
    +81°: 73 mm long, bowl 26 / handle 17 mm wide". The user confirmed the
    direction is right and it picked up.
  - The jaws moved by (−1.7, −9.9) mm, i.e. 10 mm along the handle.
  - Held at 20.5 mm (8.2 mm at the place).
  - Measured tool offset [0.034, 0.007, 0.022]. x is positive (bowl on +tool
    X, like every earlier pick) but shorter than the usual 41-47 mm: the
    jaws closed nearer the bowl this time.
  - place put it back. pick_up 55.6 s, place 26.8 s.
  - Offline first: on `four_items_20261022_162033` the spoon read +1°
    (bowl 27 / handle 19 mm), the arrow on the bowl end.
- **Now the bench default:** both `magic_beaker` and `jellyfish`
  `larger spoon` picks carry `("yaw", "auto")`.
  - `test_executor_uses_bench_names` now expects it. Its two clear-cup
    checks had been failing since the 2026-10-20 `WATER_CUP_FORWARD_OFFSET`
    change; they now expect that too. 25/25 pass.
  - `test_agents_offline.py` 133/133.
- Not yet run: a scoop/dump after a turned pick. The tool frame is the
  same, so none is expected to differ, but the shorter measured x offset
  feeds scoop's depth geometry.

**ROS master would not come up for `start_control_pc.sh` (2026-10-23 20:26).**
- The script launches roscore, sleeps 3 s, then ssh-starts franka-interface
  on the Alienware. That side failed with "unable to contact ROS master at
  http://remote-franka:11311".
- `~/.ros/log` was 1.7 GB in 402 runs. roscore spent ~16 s on "Checking log
  directory for disk usage" (warns over 1 GB) before the master listened.
  The first attempt was then shut down (SIGINT at 20:27:01).
- The second attempt came up: roscore, franka-interface, ros-interface
  and gripper nodes all present.
- `rosclean purge -y` (or deleting old run folders in `~/.ros/log`) should
  stop the stall.
- Note: every roslaunch log there dumps the full environment, including
  the OpenAI API key in plain text.
- The script must be run from `/home/rumi/frankapy` (relative paths): from
  elsewhere it is "No such file".

**Scoop chooses its heading and depth from the powder map; refuses to sweep in air (2026-10-23). Not yet run on the robot.**
- The user's spec: pick_up turns to the spoon (done, see above). scoop
  enters from the angle where the map is highest, digs to a depth set by
  the map, and must be in the powder before it moves. Return home facing
  ahead. dump may have rotation as an option.
- **Heading.** `stroke_heading` (default `"auto"`) searches
  `stroke_headings_deg` [0, ±20, ±40] together with `stroke_across` lines.
  - Each (heading, line) is scored by the median z of the dish points in
    a bowl-wide strip along its bite (entry − 10 mm to 60% of the travel).
  - Fit in the dish is heading-independent: the bowl corners are taken
    along/across the stroke. So each line's fit is computed once.
  - Each heading is checked with dump's `_wrist_reach` at the bite and at
    the end. The best must beat the default (0°, centre) by 2 mm.
  - The chosen turn is applied to `level_rotation`, `forward` and
    `across_axis`, which every planning closure reads at call time.
  - The lift-out now uses the final waypoint's x and y (it had assumed
    forward = +X).
  - After a turned stroke the scoop homes (`reset_joints`), so the wrist
    faces ahead for the next skill. A coordinate `place` does not home by
    itself.
- **Depth.** `depth_from_map` (default True): `surface_z` = the chosen
  bite's median z + `lift` (rim before). The tip goes `map_tip_depth`
  16 mm under it, but never nearer the floor than `tip_floor_gap`.
  - On the bench's usual bed (map: 13-20 mm over the inside floor) the
    floor gap binds. The tip therefore lands where the validated floor rule
    put it.
  - Refused before any motion when the bed under the bite is less than
    `map_min_bed` 6 mm above the floor gap ("Too little powder ...").
  - Off when the call sets `tip_floor_gap` or `powder_surface_z`.
- **In the powder before sweeping** (step 8c). After the descent and its
  sag correction, it computes the tip's actual z (plan + where the TCP
  stopped − where it was sent) against the mapped surface.
  - Under `min_immersion` 4 mm it lowers once more (never below
    floor + clearance).
  - Still short: it lifts out and fails ("did not sweep").
  - A force check was not used: powder resists too little, and stir's 1 N
    guard already false-triggered in free air (2026-10-15).
- Offline, synthetic dish (`test_scoop_heading.py` in that session's
  scratchpad):
  - Flat bed: default, floor-limited 11 mm.
  - Centre trench: −40°, +12 mm, 6.8 mm fuller, homes afterwards.
  - 10 mm mound behind-right: 0°, −12 mm (its bite covers the mound).
  - 1.5 mm bed: refused.
  - `stroke_heading=0, stroke_across=0`: the old floor mode exactly
    (depth 16.8, stroke −29/+22).
  - All strokes stay 8 mm from the wall.
- **The reach model is conservative for scoop.** It flags the default
  stroke at dish (0.60, −0.07), which scooped fine on the robot, as well
  as the jams at (0.642, −0.028) and (0.604, 0.183). So:
  - the default stroke still runs there, with a NOTE;
  - turned strokes are only allowed nearer, e.g. all of ±40° at
    (0.50, 0).
- Real clouds:
  - `four_items` → −40°, +12 mm, 5.5 mm fuller. Map surface 19.5 mm over
    the floor, tip 16.0 mm under it, 3.5 mm over the floor.
  - `label_check` → default. Surface 13.2 mm over the floor, tip
    floor-limited at 10.2 mm.
- Existing scoop tests 41 pass, 2 fail (the old workspace-floor pair, as
  at HEAD).
- dump's rotation option already exists: `dump_heading_deg` (degrees from
  the base→target radial), default None = straight ahead as before; never
  run on the robot.

**Stirrer: the jaws turn to the cube's faces (`yaw` "auto" square fit, 2026-10-23). Not yet run on the robot.**
- `square_top_yaw()`: fits the minimum-area rectangle (`cv2.minAreaRect`)
  to the top 6 mm of the cloud. It must be 20-45 mm and within 1.4:1. The
  turn is taken to the nearest face, in (−45°, 45°].
- `"auto"` tries the long axis first (spoon). An object that is not
  elongated falls through to the square fit.
- **Put-back does not care about the turn.** From the STLs, the rod is round
  (r 5.0 mm, 59 vertex angles) and the holder's bore is round (r 6.0 mm,
  10-65.7 mm up). The r ≈ 15 mm feature is a round chamfer in the top 1 mm,
  not a square pocket.
- Real captures (live intrinsics):
  - `four_items` +0.3° (top 30.9 × 34.8 mm, 2202 points).
  - `label_check` +5.4°.
  - `jellyfish_20261021` +2.7°.
  - The overlay square sits on the cube in cams 3 and 5.
- The `four_items` cube cloud rotated by +10…+44, −15, −35, +60, +80°
  came back exact (+60 → −30, nearest face).
- Both stirrer picks carry `("yaw", "auto")`. The bench test expects it;
  25/25 pass.

**Turned spoon picks closed nearer the bowl; the grasp is now placed from the bowl end (2026-10-23).**
- User: "pick up the spoon sometimes is in the wrong place, too far
  forward". The logs agree:
  - straight picks (10-22 `chain_fast`, `chain_preflight`) measured tool
    offset x 41 / 42 mm;
  - turned picks (10-23 `yaw_test` +81°, `scoop_heading` +110°) measured
    34 / 32 mm.
- Cause: `forward_offset` −0.010 from the centroid was tuned on a spoon
  lying straight ahead.
  - On the same cloud, rotated to any angle, centroid − 10 mm gives a
    geometric offset of 36.5 mm.
  - On the robot, the straight picks landed ~5 mm further back (a world-X
    bias: centroid lean / the TCP drifting while closing). A turned spoon
    does not get that bias along its handle.
- Fix: new `pick_up` param `working_end_distance`. With yaw auto on a long
  tool, the jaws go that far back from the working end (the median of the
  bowl end's far 30%, as `measure_tool_offset` reads it), on the handle's
  centre line. forward/lateral_offset are then not applied; the no-turn
  fallback still uses them.
  - Spoon picks carry `("working_end_distance", 0.042)`.
- Check on the `four_items` spoon cloud rotated 0/81/110/−60°: the
  predicted measured offset is 41.7-41.9 mm at every angle (old method
  36.5).
- On the robot, the close itself still moves the TCP (8.8 mm on 10-23):
  watch `[PickUp] Measured tool offset` after a pick. It should read
  ~0.040-0.044 whichever way the spoon lies.

**First robot run of the turned scoop, and why its choice leaned to the wall (2026-10-23 20:4x-20:5x, `experiments/chain_yaw_20261023.log`).**
- Chain: pick_up spoon → scoop → place → pick_up stirrer → place stirrer.
  - Spoon: grasp 42 mm back from the bowl. Measured tool offset
    **[0.039, 0.005, 0.019]**, against 32-34 for the turned picks before
    the fix; on target.
  - Stirrer: "Square top 32 × 34 mm: faces turned −37°". The wrist turned
    −37° and held at 30.7 mm.
  - Scoop: the dish at (0.5188, 0.0402) used the median-fallback centre.
    It chose **+40°, −12 mm across, "6.4 mm fuller"**. Map surface 17.1 mm
    over the floor, tip planned 14.1 mm under it, "In the powder: tip
    9.8 mm under ... 7.3 mm over the floor" (the descent stopped 4.3 mm
    high). Success, then homed.
- The user: the scoop still does not find where the powder is.
- **Analysis** of a capture right after (`scene_captures/dish_after_scoop_20261023_205347`).
  - Mistake: taken while the chain was still running (the stirrer scan).
    Both captures succeeded, but do not open cameras during a run.
  - The centre was fine: a circle through the rim points lands 2.4 mm from
    the scoop's centre.
  - Of the points inside the 37.5 mm reach circle, **22% were within 4 mm
    of the rim**. The chosen strip had 26% such points (median 23.8 mm over
    the table), the centre strip 1% (18.5 mm).
  - The scored strip was the bowl width + 3 mm a side and started 10 mm
    before the entry, so it reached the wall bank the bowl (8 mm off the
    wall) never digs.
- **Fix:**
  - The map is now 4 mm cells (median per cell, >= 3 points), without
    points within `map_rim_band` 3 mm of the rim.
  - A stroke scores the MEAN of the cells its bowl sweeps (entry to 60% of
    the travel, exactly bowl-wide).
  - Every scoop writes `diag_out/scoop_maps/scoop_map_<time>.png`: the
    cells, the reach circle, every candidate (grey), default (white),
    chosen (orange); top = away from the base, left = the robot's left.
- On the after-scoop cloud it still chooses +40/−12 (+5.1 mm). The new
  map shows:
  - a dark trench under the default line (earlier straight scoops);
  - the high ground on the far side and the robot's right, toward the wall.
- The camera crops cannot settle whether that bank is powder: white
  powder in a white dish shows no relief. Ask the user where the powder
  actually is, against the saved map.
  - `four_items` now: −40/+12 (+2.8).
  - `label_check`: default.
  - Synthetic cases unchanged.
- **User (2026-10-23): the powder is in the right half**, which matches the
  map's high ground (the robot's right, −y).
  - The chosen +40/−12 bite sat half on the centre: lines were capped at
    ±12 mm by `stroke_min_travel_frac` 0.75.
  - On the after-scoop cloud:
    - frac 0.5 → +40°/−18 mm, +6.8 mm, travel 29 mm, the bite wholly in
      the high ground (`maps/frac50` image);
    - headings ±60/±90 at 0.75 → +60/−12, +5.5 mm only.
  - Default is now 0.5. Headings stay ±40 (the bigger turns gained little
    and are nearer joint limits).
  - Untested: whether a 29 mm stroke brings up as much as 40-52 mm.

**`right_spoon`: push a scoop lying on its side onto its back (2026-10-23). Not yet run on the robot.**
- User asked whether the robot can right a spoon that has fallen on its
  side. Agreed: a scripted motion + scan check + retry, not a learned
  policy.
- **Telling the poses apart** (`spoon_pose`, `robochem/skills/right_spoon.py`).
  It compares the cloud's two end thirds along the principal axis: width
  across (95−5 pct) and height (p95 over the table).
  - Flat, mouth up (`four_items_20261022_162033`): widths 19 / 27 mm, the
    wide end (bowl) LOW at 8.9 mm; the handle end high at 23.1 mm (the
    crank lifts it).
  - On its side (`spoon_on_side_20261023_210216`, cams 2+3): widths
    15.2 / 15.2 mm; the bowl standing tall (21.3 mm), the handle low
    (6.7 mm).
  - Upside down is assumed to be the wide end high. Only checked on the
    flat cloud mirrored in z.
  - `tool_long_axis_yaw` on the side cloud already refused ("cannot tell
    the bowl end from the handle (15 vs 15 mm)"), so a turned pick_up of a
    side-lying spoon falls back to the home rotation.
- **Which way the mouth faces.** From above, the side-lying scoop is a "Z":
  each end segment fitted on its own gives two parallel lines 15.1-16.6 mm
  apart (end fractions 0.25-0.35). That is the crank's ~14 mm rise turned
  sideways, so the handle's line lies on the mouth's side of the bowl's.
  - A whole-cloud principal axis hides it (1.9 mm), because the axis runs
    through both ends.
  - On the capture: mouth toward (+0.95, −0.32), away from the base and
    slightly right. **Confirmed by the user at the bench (2026-10-23): the
    mouth faced away from the robot.**
- **Motion:** jaws closed, TCP 18 mm over the table (the flat-spoon grasp
  height, known to clear the table), from 35 mm out on the mouth side,
  pushed under impedance to 30 mm past the bowl centre on the far side.
  The bowl's top edge goes away from the mouth, so the scoop rolls onto its
  back. Then it rescans, up to 2 pushes. Upside down / unknown are
  reported, not handled.
- **Rejected: lift and roll the wrist 90°.** The Franka hand is a ~200 mm
  bar along the closing axis. A 90° roll stands it on end ~100 mm below the
  fingers, so it could only let go from >10 cm up.
- Registered as skill `right_spoon` (touches `object_name`). Run:
  `python -u scripts/run_experiment.py --bench jellyfish --workspace-min 0.25 -0.40 -0.13 --no-preflight --skill right_spoon`
- Watch: the hand's bar sweeps ±100 mm along base Y at ~76 mm over the
  table during the push. Keep the stirrer holder (85 mm + cube) out of
  that band.
- **The first launch failed on import** (2026-10-23): "skill_catalog is out
  of step with robochem.skills: right_spoon: in SKILL_REGISTRY but not in
  the catalogue". Then: "Can't instantiate abstract class RightSpoonSkill
  with abstract methods check_preconditions".
  - Fixed: `right_spoon` has a catalogue entry (`object_name`, `attempts`;
    the planner may use it before a pick_up), and `check_preconditions`
    refuses while the jaws hold something.
  - **Lesson: after registering a skill, run
    `python scripts/run_experiment.py --help` in the robot venv and
    `test_agents_offline.py` (2 s) before handing over the command.**
- **pick_up rights the scoop by itself** (user: it must decide on its own
  whether the spoon is on its side, and leave it alone otherwise).
  - New pick_up param `right_if_on_side` (default False; both benches'
    `larger spoon` picks carry True).
  - After pick_up's own scan it runs `spoon_pose` on those points, so
    there is no extra scan.
    - On its side: `right_spoon`, open the jaws, home, rescan, then the
      normal turned pick.
    - Flat: nothing extra.
    - Upside down: a warning, and it picks anyway.
  - The result reports `righted`.
- Fake-arm checks on the two real clouds:
  - Flat: "lies flat, mouth up already", 0 arm commands.
  - On its side: two pushes (0.587, −0.119) → (0.525, −0.098) at TCP z
    0.0068, a rescan after each, then "still on its side after 2 push(es)"
    (the fake cloud never changes).
  - The flat cloud through pick_up printed "How 'larger spoon' lies: flat"
    and took no righting branch; the test double has no grasp maths, so
    that run stops there.
  - Bench test expectation updated; 34/34 pick checks,
    `test_agents_offline` 133/133, `run_experiment.py --help` imports in
    py3.8.
- **First robot try (2026-10-23, `experiments/right_then_pick_20261023.log`):
  it slid the scoop instead of tipping it.**
  - pick_up's check worked: "How 'larger spoon' lies: on_side (16.3 / 16.2
    mm wide, 21.9 / 9.4 mm high)", and right_spoon was called. Crank step
    18.5 mm, mouth (+0.93, −0.37).
  - Push 1 ran at TCP z 0.0068 (18 mm over the table) to 30 mm past the
    bowl centre. The bowl moved ~40 mm toward the base and turned ~30°
    (mouth then (+0.98, +0.19)), still on its side.
  - User: "pushing too deep; it translates the spoon instead of righting
    it". The run was stopped after push 2.
  - Fix:
    - The fingertips meet the bowl at `push_contact_frac` 0.7 of its
      measured height. `finger_below_tcp` 0.009 is an estimate, not
      measured on this hand. For the 21-22 mm bowl the TCP is ~24 mm over
      the table, 6 mm higher than before.
    - `push_past` 30 → 10 mm.
    - Up to 3 pushes: a push that slid it (> 10 mm, still on its side)
      raises the next by 4 mm; one that did not move it (< 5 mm) lowers
      the next by 3 mm, never below 8 mm.
  - Fake-arm sequences (real clouds):
    - slid 40 mm, then flat → push 2 at fingertips 18.9 mm, "flat after 2
      pushes";
    - never moves → 14.9 → 11.9 → 8.9 mm, then "still on its side after
      3 push(es)".
  - Catalogue `attempts` default synced to 3 (the audit caught the
    mismatch). Imports ok in py3.8; agents 133/133.
- **Pushing does not right it; pressing does (user's hand test, 2026-10-23):**
  "only by pressing the side-lying spoon's bowl down with a very small
  force". right_spoon's default `method` is now `"press"` (the push stays
  as `"push"`).
  - Where: the centroid of the bowl end's top 30% of points
    (`bowl_top_xy`), `press_offset` 4 mm toward the bowl's back (away from
    the mouth), +3 mm more per retry.
  - How: jaws closed, from 15 mm over the bowl's top, 1.5 mm steps. Contact
    is a 0.8 N change from a resting baseline (sign ignored, as in place's
    probe). Then 4 mm further, so it rolls over. Never below 8 mm
    fingertip height. Then rescan, up to 3 tries.
  - Fake arm (force −1.5 N below the 21.3 mm top): start 36 mm, contact
    21.3 mm, lowest fingertips 17.3 mm, flat after 1.
  - **Open question for the user:** where exactly the hand pressed (top
    centre / mouth side / back side) and whether it fell mouth up. The
    back-side offset is reasoned, not observed.
- **Where to press, from the user (2026-10-23):** on the handle's bend
  nearest the bowl, of the Z's two corners, not on the bowl. `press_at`
  `"corner"` is now the default (`"bowl"` = the bowl's top, also the
  fallback).
  - Finding it: walk from the bowl segment's centroid toward the handle in
    3 mm bins. The corner is the first bin whose median across-offset
    leaves the bowl's line by > 25% of the line-to-line step.
  - On `spoon_on_side_20261023_210216` (step 15.7 mm): the across-offset
    stays within 0.13 of the step up to t = +18 mm, then 0.28 at +21 mm.
    Corner (0.5468, −0.1272), 13.6 mm tall (bowl 21 mm, handle ~6 mm).
    Drawn on cams 2 and 3 (`diag_out/powder_heightmap/spoon_corner.png`)
    it sits on the bend by the bowl.
  - Press: from 15 mm over the corner's top, contact at a 0.8 N change,
    then 4 mm further (+2 mm per retry), fingertips never under 8 mm.
  - Fake arm: contact 13.6 mm, lowest 9.6 mm, flat after 1.
  - Imports ok (py3.8), agents 133/133.
- **Robot try of the corner press: wrong spot, and too weak** (user,
  2026-10-23: "the gripper is too big"; "the press is too weak").
  - Closed jaws are ~2 × 2 cm. The bowl, 18-21 mm tall and 1.5-2 cm from
    the 13.6 mm corner, was under them and met first.
  - Fix: press with ONE finger. The jaws open `press_opening` 60 mm; the
    wrist turns (smallest turn, |θ| <= 90°, either finger will do) to put
    the closing axis along the scoop; the TCP goes 30 mm past the corner
    toward the handle. The near finger is then over the corner, and the far
    one 60 mm on, past the handle's end (t ≈ 81 mm vs the handle end
    ≈ 63 mm).
  - Deeper: `press_follow` 4 → 7 mm (+3 per retry). The press is
    position-controlled, so that is real travel. `min_finger_height`
    8 → 5 mm.
  - Fake check: wrist −19°, TCP (0.537, −0.156), the near finger exactly
    on the corner (0.5468, −0.1272), lowest fingertips 6.6 mm.
    `finger_below_tcp` 9 mm is still an estimate.
- **Stopped (user, 2026-10-23): "too hard for now; even by hand it has to
  be picked up and set back down the right way."**
  - Tried on the robot, none of which righted the scoop:
    1. Sideways push at TCP 18 mm, 30 mm past the bowl: it slid ~40 mm
       and turned ~30°.
    2. A light press with the closed jaws on the crank corner by the bowl:
       the ~2 × 2 cm jaws met the taller bowl next to it.
    3. A one-finger press, 7 mm deep: not reported as working before the
       user stopped.
  - The way that does work by hand, pick it up and put it down flat,
    was not tried. It needs a grasp of the side-lying scoop and a wrist
    roll, and the 200 mm hand bar makes a low release impossible (see
    above).
  - State now:
    - `right_spoon` is NOT registered and has no catalogue entry, so the
      planner never sees it. The module stays, with `spoon_pose`.
    - Both benches' `larger spoon` pick carries
      `right_if_on_side "warn"`: "WARNING: ... lies ON ITS SIDE. Set it
      flat, mouth up, by hand", and the pick goes on. No motion is added.
    - `True` would still call right_spoon; do not use it until a working
      motion exists.
  - Checks: imports ok (py3.8), agents 133/133, pick checks 34/34.

**Every skill call carries a speed, 0-100, chosen by the planner (2026-10-27). Not yet run on the robot.**
- User: liquid should move as slowly as now; an empty spoon much faster.
  The current pace is considered slow, and **nothing may run slower than it**.
  A first version had 0 = 2x slower and 25 = today; the user rejected it the
  same day.
- Scale (`robochem/skills/pacing.py`):
  - **0 = today's pace exactly**, and the slowest allowed (`DEFAULT_SPEED`,
    factor 1.0). Any call without a speed, including every manual
    `--params`, is unchanged.
  - `SPEED_ANCHORS` ((0, 1.0), (100, 0.4)) are time multipliers, interpolated
    geometrically, so each step of 10 is about 1.10x faster.
    - 50 → 0.63 of today's time.
    - 100 → 0.4 of today's time, i.e. 2.5x faster.
  - A default 3 s `goto_pose` takes 5.6 s at 0 and 2.25 s at 100. Home takes
    7.5 s at 0 and 3.0 s at 100, a little quicker than frankapy's own
    defaults. 100 was picked as a cautious ceiling, not measured.
  - `MIN_FAST_SECONDS` 1.0: a faster speed never shortens a move below 1 s
    (stir's 1 mm probes and the twist's quarter cycles stay as they are).
- Where it acts:
  - `PacedArm.speed` scales `goto_pose`, `goto_joints` and `reset_joints`,
    explicit `reset_joints` durations included.
  - `stream_pose_path` scales `seconds`. Paths that already have one setpoint
    per tick (stir's circles, dump's twist) are time-compressed as they are
    (`resample_path_uniform`) and are not eased again.
  - Holds and waits are not scaled.
  - **Stir's circles speed up with the rest.** The 2026-10-22 note already
    flagged cup drag at the current 2.5 s/rev. The guide puts stir at 30
    (1.32x).
- How a call gets it:
  - `SkillsExecutor.execute` pops `params["speed"]` for any skill. It sets the
    speed on the arm for that call and puts it back to 0 afterwards, even if
    the skill raises. A non-number runs at 0 with a printed note.
  - Manual use: `--params '{"object_name":"larger spoon","speed":60}'`.
- Planner (`skill_planner.py`):
  - The schema now requires `speed_reason` and then `speed`, so the model
    writes its reason before picking the number.
  - `skill_catalog.SPEED_GUIDE` / `speed_prompt_block()` render the scale
    every 10 marks, with the seconds computed from the code:
    - liquid 0;
    - little liquid and the pipette 10;
    - a loaded scoop 20;
    - stir and a just-poured container 30;
    - an empty container and a tool into its holder 40;
    - an empty tool 50-70;
    - nothing held 80+.
  - Only `moves_arm` skills get `speed` in params. The orchestrator prints
    `(speed N: reason)` after the rationale. It can be edited at the step
    prompt like any other parameter.
- **What the planner picks, replayed (gpt-4.1, 2026-10-27).**
  - Method: each recorded sub-task went through `plan_skill_call` with the
    record's scene, the held object and the history. Scratch script
    `replay_speeds.py`, not kept.
  - The skill chosen matched the record on every step of both runs.
  - Jellyfish (`agent_run_20261022_145232`, 20 steps):
    - spoon pick 60, scoop 20, dump 20;
    - spoon and stirrer back to their holders 40, stirrer pick 60;
    - stir 30;
    - beaker of oil pick and pour 0, beaker put back 30;
    - pick_pipette and dispense 10, pipette back 40.
  - Magic Beaker (`agent_run_20261020_142834`, 32 steps):
    - the same pattern for the three scoop/dump/stir rounds, except stirrer
      picks at 50 twice and 60 once;
    - beaker and cups A/B picks and pours 0;
    - put-backs after a pour of 90 at 30.
    - Step 32 (place cup B) got 0: the record ends before 31 ran, so the
      model saw no pour of cup B in the history and believed it still held
      10 ml. In a live run it would follow a 90 pour.
- Checks:
  - `test_agents_offline.py` 137/137: 3 new speed checks; 2 expectations now
    include `"speed"`.
  - `test_robot_and_sim_share_one_pace` passes unchanged.
  - Fake-arm check (first scale, same code path): the default speed
    reproduced 5.625 / 0.9375 / home 7.5. At speed 100 the same moves were
    2.25 / 0.9375 (floor) / home 3.0.
- First things to watch on the robot:
  - The empty-spoon `pick_up` at 60, for velocity or acceleration
    discontinuity faults (`last_motion_errors`).
  - The stir at 30 for cup drag.
  - The dump twist at 20.
- **First robot run, speed 0 (2026-10-27 15:4x, `experiments/speed_test_20261027.log`).**
  - Spoon pick_up took 67.2 s and place took 25.1 s. This is the baseline
    for the speed-60 comparison.
  - A second run, also at speed 0, failed on the lift ("Object lost during
    the lift: gripper went from 19.4mm to 1.7mm", pick_up 61.6 s). The spoon
    was still on its side; yaw again refused (26 vs 20 mm). No speed-60 run
    yet.
  - The first attempt died on `No module named 'numpy'` because
    `scripts/env.sh` had not been sourced.
  - **The wrist did not turn.** It is not the speed change: `yaw auto`
    refused.
    - The spoon lay on its side ("lies ON ITS SIDE" warning). Its two ends
      read 27 vs 24 mm wide, inside the 1.3x the bowl test needs, and the
      square-top fallback is for the stirrer cube.
    - It grasped at home rotation and still picked and placed.
    - User report: "it never turns, always straight ahead".
  - **The spoon was NOT on its side (user). It was a perception error, now fixed.
    Not yet re-run on the robot.**
    - What was checked:
      - `check_object.py --object "white plastic tool"` at 15:49/15:5x,
        spoon at (0.557, -0.196), saved `diag_out/spoon_points.npy`.
      - The 2D masks are clean on all four cameras. The scoop on the bench
        is now red, and `"white plastic tool"` still finds it; `"red
        plastic tool"`, `"red plastic spoon"` and `"orange plastic scoop"`
        return no masks at all, so keep the prompt.
      - cam 5 sees only part of it: 262 px, conf 0.54, 25 mm off the other
        cameras.
    - Cause: a top view of the fused cloud shows the spoon itself is right,
      about 70 mm long, bowl ~27 mm, handle ~10 mm. Next to the handle lie
      arcs of sparse stray points (depth edge pixels). They put the
      handle-end width (5-95 percentile) at 22 mm. Bowl and handle then
      failed the 1.3x width test, so `spoon_pose` said `on_side` and `yaw
      auto` refused.
    - Fix: `pick_up.drop_stray_points`.
      - It keeps points with at least 8 points (itself included) within 3 mm,
        using scipy cKDTree. When it would remove more than half the cloud,
        the cloud is kept whole.
      - It is used only for `spoon_pose` and `tool_long_axis_yaw`; the grasp
        cloud is unchanged.
      - On the saved cloud it kept 2805 of 3052 points. Result: `flat`,
        handle 11 / bowl 21 mm, yaw +9.6°.
      - Stricter settings (k12 at 3 mm, k15 at 4 mm) give the same.
    - Not checked: a truly side-lying scoop after the filter. The 2026-10-23
      capture has images only, no cloud.
    - The spoon lies nearly straight ahead (+9.6°), so even a working turn
      is small. Angle it on the bench to see a clear one.
    - `test_cups_are_taken_from_behind` fails offline (`FakePose` into
      `to_rigid_transform` from `place.py:151`). It fails the same with the
      filter off; not from this change.

**Stirrer put-back: one move down and open, no force probe (user, 2026-10-27). Not yet run on the robot.**
- Robot run `experiments/stirrer_speed_20261027.log` (pick 60, place 40):
  - pick_up 42.5 s ok; place 40.0 s ok.
  - The place went: home, cross at 0.30, down to 15 mm above the release,
    then six 2 mm force-probe steps. Contact was at z 0.1088 (−1.12 N),
    5 mm ABOVE the pick's grasp z 0.1038.
- User: too slow, "just open the gripper".
- `tool_geometry` stirrer `put_back` is now
  `{"vertical_insert": True, "release_clearance": 0.015}`. `stop_force_n` is
  gone, so place uses its plain single descent.
- Why 15 mm:
  - The seat has been felt 13 mm (2026-10-05) and 5 mm (today) above the
    grasp z; the rod slips in the jaws.
  - Releasing at the grasp z would drive it 5-13 mm into the seat
    (−96 N on 2026-10-05).
  - At 15 mm the rod drops at most 15 mm down the 66 mm bore.
  - If it ever lands crooked, lower it toward 0.010.
- Still in place: the home first and the 0.30 crossing.
- Checks:
  - `test_executor_uses_bench_names` 25/25.
  - `test_place_releases_on_a_negative_press` dies on the same `FakePose` /
    `place.py:151` error as `test_cups_are_taken_from_behind`. That error is
    pre-existing.

**New objects by name alone: what SAM 3 needs (2026-10-27).**
- `"pen"` (a red marker standing upright) was found by SAM 3 on all four
  cameras with no manifest entry: about 13 cm tall, 13-19 mm wide.
  - Cam 4 put it 350 mm off: the gripper stood in front of it, so the mask's
    depth was the gripper.
  - Put back it needs `vertical_insert` + `release_clearance`. The plain
    put-back releases at centroid z + 20 mm, and a top-held upright object
    would be driven into the table.
- `"blue papercup"` failed (`speed_test_20261027.log` l.544-597). Two causes:
  - Any name with "cup" goes first through the labelled-cup path: it looked
    for a paper label reading "blue papercup", which cost 20+ s.
  - The object is a blue drink CARTON, not a cup. `"blue papercup"`,
    `"blue paper cup"` and `"blue cup"` all gave no masks.
- Prompts that find it on all four cameras (same frames):
  - `"blue carton"` (conf 0.55-0.83);
  - `"juice box"` (0.62-0.94);
  - `"drink carton"` (0.40-0.71).
  - `"blue box"` found it on cam 5 only.
- Its geometry (`check_object`): 139 mm tall, cloud 55-65 mm wide, top 46-51
  mm. Cameras up to 38 mm apart, so the width is probably inflated.
- Lesson: name what the object is, not what it is used as, and keep "cup"
  out of a name that is not a labelled cup.
