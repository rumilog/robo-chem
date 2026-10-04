# Progress — robo-chem (Franka + RealSense)

Last updated: 2026-10-03

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
