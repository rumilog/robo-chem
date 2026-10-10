# Progress — robo-chem (Franka + RealSense)

Last updated: 2026-10-08

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
    before it is put down; B's pour, the beaker's last, stays 90.
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
