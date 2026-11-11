# robo-chem

Autonomous robotic chemist stack for a **Franka Panda** arm in a **RealSense** camera cage. Skills use multi-view SAM 3 segmentation, fused 3D localization, and frankapy motion. A VLM orchestrator can plan tasks; individual skills can also be run directly from the CLI.

Barati Farimani Lab, CMU.

---

Working in this repo: read [`progress.md`](progress.md) before you start and
update it when you finish — see [`CLAUDE.md`](CLAUDE.md).

---

## Running the experiments from another account on this machine

A second Linux account on the robot workstation can run every experiment with
its own clone, its own OpenAI key and its **own processes**: the grounding
service and the robot control stack. It also works while rumi is logged out.

The installed stack under `/home/rumi` is used **read-only**. That covers the
ROS workspace `frankapy/`, the robot venv `franka/`, `robomail` with the
camera calibration, librealsense, and the SAM 3 env and weights. These are
world-readable files on disk, so they work whether or not rumi is logged in.
Nothing is reinstalled. `<user>` below is the new account's name.

### 0. Once per account, by an admin (sudo)

```bash
sudo usermod -aG video <user>      # then <user> logs out and back in
```

The cameras are `/dev/video*` (`root:video 0660`). Without the group, a
camera opens only for whoever is logged in at the desktop.

### 1. Once per account, as `<user>`

```bash
git clone git@github.com:rumilog/robo-chem.git ~/robo-chem   # needs read access on GitHub
cd ~/robo-chem

# Borrow the two gitignored pieces from rumi's checkout (read-only)
ln -s /home/rumi/Desktop/robo-chem/perception_env perception_env    # SAM 3 service env
mkdir -p weights && ln -s /home/rumi/Desktop/robo-chem/weights/sam3.pt weights/sam3.pt
echo perception_env >> .git/info/exclude     # .gitignore does not match a symlink

# Your own key, and your own port for the grounding service
( umask 077; printf 'OPENAI_API_KEY=sk-...\nGROUNDING_URL=http://127.0.0.1:%s\n' \
    "$((5100 + $(id -u) % 800))" > .env )

# Passwordless login to the robot's control PC (asks for the franka password once)
ssh-copy-id franka@franka-Alienware-Area-51-R5
```

- Do **not** run `git submodule update --init`. Nothing here needs a
  submodule, and the `robomail_Aliyah` entry has no URL, so the command errors.
- Do **not** run `calibrate_cameras.py`. It installs into the shared
  `robomail` (`/home/rumi/Desktop/camera examples/robomail`), which every
  account reads its calibration from.

### 2. Every session

1. **Robot.** Unlock the arm in Franka Desk and activate FCI. The lights must
   be **blue**: white means the e-stop is down, and a stack started under
   white lights fails with `FrankaInterface status not ready`. Then start the
   control stack from a desktop terminal, because the script opens
   gnome-terminals. It has to run from `/home/rumi/frankapy`, because the
   script uses relative paths:

   ```bash
   cd /home/rumi/frankapy && bash bash_scripts/start_control_pc.sh -u franka -i franka-Alienware-Area-51-R5
   ```

   Only one stack can drive the robot. If an earlier session's stack is still
   up, whoever started it closes its terminals first.

2. **Env + perception service** (same terminal as the runs):

   ```bash
   cd ~/robo-chem
   source scripts/env.sh
   scripts/grounding.sh start      # idempotent; status | log | restart | stop
   ```

   Your service runs on your own port (`GROUNDING_URL` in `.env`), so you
   never use, or try to stop, another account's service.

3. **Bench, before each trial** (see [`PARAMETERS.md`](PARAMETERS.md)):
   - each powder dish holds a levelled bed of **at least 15 mm**
   - the spoon and stirrer holders sit at **x ≤ 0.74 m**: preflight does not
     move tools, and past the 0.75 m hand clamp the grasp lands short
   - scoop dishes are within about **0.62 m** of the base: scoops jammed at
     0.631-0.640 m
   - optional camera-only layout check (the arm does not move):
     `python scripts/check_bench_layout.py --bench magic_beaker` (or `--bench jellyfish`)

### 3. The experiments

All commands run from `~/robo-chem` after `source scripts/env.sh`. Each run
writes everything (record, `console.log`, photos, checks, scoop maps) to
`experiments/run_<time>/` in **your** clone. That folder is gitignored, so it
is not shared through git.

**Magic Beaker** (`instructions/magic_beaker.png`, 31 sub-tasks, ~30 min):

```bash
python -u scripts/run_experiment.py --bench magic_beaker --workspace-min 0.25 -0.40 -0.13 \
  --instruction-image instructions/magic_beaker.png --follow-steps --no-verify --reset
```

**Jellyfish** (`instructions/jellyfish.jpg`, 20 sub-tasks, ~20 min):

```bash
python -u scripts/run_experiment.py --bench jellyfish --workspace-min 0.25 -0.40 -0.13 \
  --instruction-image instructions/jellyfish.jpg --follow-steps --no-verify --reset
```

- `--reset` homes the arm and **opens the gripper** before step 1. Use it
  only when starting from scratch.
- **Resume after a stop:** run the same command **without `--reset`** and
  add `--resume latest --from-step N`. Start from a step that does not need
  something already in the gripper, e.g. the pick before a scoop.
- To see the plan without moving the arm, replace `--reset` with
  `--dry-run`. The cameras and LLM calls still run.
- Write the report afterwards with
  `python scripts/write_report.py --bench magic_beaker` (or `--bench jellyfish`).

**Label experiments** (cameras only, the arm is never imported; bench =
`LABEL_TEST` in `progress.md`):

```bash
# Label reading benchmark: printed vs handwritten label sets
python scripts/label_reading_benchmark.py scan --set <set-name>   # takes ground truth first for a new set
python scripts/label_reading_benchmark.py report

# Label distance test: objects "baking soda cup" | "10 ml water cup" | "beaker",
# phases labelled | unlabelled
python scripts/label_distance_test.py run --object "beaker" --phase labelled   # ground-truth + trials
python scripts/label_distance_test.py report
```

If the arm gets stuck, stop the run (Ctrl-C). Before you resume, home it, and
open the gripper only after taking out anything it holds:

```bash
python -c "from frankapy import FrankaArm; fa=FrankaArm(); fa.reset_joints(); print(fa.get_pose().translation)"
python -c "from frankapy import FrankaArm; fa=FrankaArm(); fa.open_gripper(); print(fa.get_gripper_width())"
```

### 4. End of every session

```bash
scripts/grounding.sh stop
```

Then close the control-stack terminals. Both keep running after you log out.
After its first query the SAM 3 service holds ~3 GB of RAM and ~5 GB of the
GPU's 11 GB, so a forgotten one can leave no room for the next person's.
Only its owner or an admin can stop it (`sudo pkill -u <user> -f grounding_service.py`).

---

## Requirements

| Piece | Notes |
| --- | --- |
| Franka + `franka-interface` / ROS Noetic | Control PC stack under `/home/rumi/frankapy` |
| Python 3.8 venv with frankapy | `/home/rumi/franka` (activated by `scripts/env.sh`) |
| Perception env (Python 3.10) | `perception_env/` — runs the grounding service |
| SAM 3 weights | `weights/sam3.pt` (see `scripts/fetch_sam3.py`) |
| OpenAI API key (optional) | `.env` / `OPENAI_API_KEY` for VLM planning and labelled-cup matching |

Do **not** stream all four RealSense cameras at once — capture is sequential (open → frame → close) to avoid USB bus freezes.

---

## Quick start

### 1. Robot / ROS env

```bash
cd /home/rumi/Desktop/robo-chem
source scripts/env.sh
```

### 2. Grounding service (separate terminal)

```bash
cd /home/rumi/Desktop/robo-chem
perception_env/bin/python perception_service/grounding_service.py \
  --backend sam3 --model weights/sam3.pt --no-exemplars
```

Service listens on `http://127.0.0.1:5005` by default. The printed scoop is
the text prompt `"white plastic tool"`. Exemplar matching (DINOv2 plus a
second SAM, for the stirrer and the holder) stays unloaded unless you pass
`--exemplars exemplars`. See `exemplars/README.md`.

### 3. Run a skill

```bash
source scripts/env.sh
python scripts/run_experiment.py --skill pick_up \
  --params '{"object_name":"plastic beaker","z_offset":0.02,"squeeze":0.013}'
```

---

## Validated skill commands

### Pick beaker (upright top-down)

```bash
python scripts/run_experiment.py --skill pick_up \
  --params '{"object_name":"plastic beaker","z_offset":0.02,"squeeze":0.013}'
```

Opens the gripper, **`reset_joints`** (joint-space home) before scanning, then top-down grasp. Closing uses measured width − squeeze.

### Pick spoon (lower workspace floor)

```bash
python scripts/run_experiment.py --skill pick_up \
  --workspace-min 0.25 -0.40 -0.13 \
  --params '{"object_name":"larger spoon","z_offset":0.0,"grasp_force":1.0}'
```

### Pour into white paper cup

```bash
python scripts/run_experiment.py --skill pour \
  --params '{"target_container":"white paper cup","pour_angle":90,"hold_duration":2,"forward_offset":-0.08}'
```

- `reset_joints` before scanning and again after the pour hold
- Tips toward the robot base (−X) to 90°
- Each 10° tip step advances the EE **+X by 0.25 cm** (`tip_advance_m`) so the mouth stays over the cup
- Tune `forward_offset` / `tip_advance_m` if the stream misses the cup

### Scoop from a labelled cup

Cups sit on handwritten labels (`CITRIC ACID`, `BAKING SODA`, …). Locate matches label → cup via GPT-4o + multi-instance SAM:

```bash
python scripts/run_experiment.py --skill scoop \
  --workspace-min 0.25 -0.40 -0.13 \
  --params '{"powder_source":"citric acid","tool_length":0.08}'
```

Offline label check (no robot):

```bash
python scripts/check_labeled_cups.py \
  --images-dir "scene_captures/spoon graspped_20260909_155259" \
  --label "citric acid" \
  --out-dir diag_out/labeled_cups
```

---

## Repository layout

```
robo-chem/
├── robochem/                 # Python package
│   ├── skills/               # pick_up, pour, scoop, place, …
│   ├── agents/               # LLM planner: scene → plan → skill call
│   ├── vision/               # multi-cam localizer, grasp analyzer, grounding client
│   ├── orchestrator/         # the closed loop over the agents and the skills
│   └── verification/         # chemistry outcome checks
├── perception_service/       # Flask SAM 3 / GroundingDINO / exemplar service
├── scripts/
│   ├── env.sh                # ROS + frankapy + PYTHONPATH
│   ├── run_experiment.py     # main skill / task entry
│   ├── calibrate_cameras.py  # board-free cube calibration
│   ├── capture_scene.py      # dump color/depth stills
│   ├── register_object.py    # teach an object SAM cannot be told about
│   └── smoke_test_*.py
├── weights/                  # sam3.pt, etc.
├── exemplars/                # objects registered by appearance, not by name
├── calibration_out/          # extrinsics, taught poses
├── diag_out/                 # debug overlays / fused clouds
├── scene_captures/           # saved multi-cam scenes
├── progress.md               # detailed status & known-good params
└── IMPLEMENTATION_PLAN.md    # longer design notes
```

---

## Skills

| Skill | Role |
| --- | --- |
| `pick_up` | Segment → fuse → top (or side) grasp → lift |
| `pour` | Locate target → tip toward base → hold → `reset_joints` |
| `scoop` | Dig from a powder source (optionally by reagent label) |
| `place`, `dispense`, `stir`, `move_to`, `tilt` | Other manipulation |
| `open_gripper` / `close_gripper` | Gripper |
| `analyze_scene`, `locate_object`, `check_container` | Perception |

Many skills besides `pick_up` / `pour` are less hardened (workspace clamps, reach retries, pre-scan home).

Run any skill:

```bash
python scripts/run_experiment.py --skill <name> --params '<json>'
```

---

## Natural-language tasks (the LLM agents)

`--task` runs the agent pipeline in [`robochem/agents/`](robochem/agents/): it
looks at the bench through the cage, plans sub-tasks, chooses a skill and its
parameters for each one, executes them, and replans when one fails. It is
robomail_Aliyah's multi-agent planner — structured output, a robot capability
profile, closed-loop correction — with its fake executor replaced by the real
`SkillsExecutor`.

```bash
source scripts/env.sh
python scripts/run_experiment.py --task "Scoop citric acid into the white paper cup"
```

**See the plan without moving the arm.** `--dry-run` runs every LLM stage and
prints the skill call it would make for each sub-task, then stops. This is the
cheap way to check a task before committing the bench to it:

```bash
perception_env/bin/python scripts/run_experiment.py --sim --no-viewer --dry-run \
    --workspace-min 0.25 -0.40 -0.13 \
    --task "Scoop citric acid into the white paper cup"
```

```
[scene] 4 objects: citric acid cup, larger spoon, plastic beaker, white paper cup
[plan] 4 sub-tasks
   1. Pick up the larger spoon
   2. Scoop citric acid from the citric acid cup with the held larger spoon
   3. Dump the loaded larger spoon into the white paper cup
   4. Place the larger spoon beside the citric acid cup
```

Drop `--dry-run` and the same run executes. Step 2 then comes out as:

```
[step 2] Scoop citric acid from the citric acid cup with the held larger spoon
   -> --skill scoop --params '{"powder_source": "citric acid cup",
                               "tool_offset": [-0.0306, -0.0015, 0.0039]}'
```

That `tool_offset` is not a number the model invented, and it is not in the dry
run either: `pick_up` measured it off the point cloud at the moment it grasped
the spoon, and the orchestrator put it in the prompt for the next step. A dry run
grasps nothing, so there is nothing to measure and the parameter is absent.

| Flag | Effect |
| --- | --- |
| `--dry-run` | Plan in full, execute nothing |
| `--max-retries N` | Corrective replanning attempts (default 3) |
| `--no-verify` / `--verify` | Chemistry check at the end. On by default on hardware, off in `--sim`, which has no chemistry to check |
| `--legacy-planner` | The older single-prompt `VLMOrchestrator` |

Every run writes a JSON record to `--log-dir` (default `experiments/`) holding the
scene, every plan and correction, every skill call with its parameters, and what
each one returned.

### How the pieces fit

| Stage | Module | Does |
| --- | --- | --- |
| Scene | `agents/scene.py` | Names what is on the bench, using names perception can actually resolve |
| Plan | `agents/planner.py` | Goal → ordered sub-tasks; also the corrective replan |
| Skill call | `agents/skill_planner.py` | One sub-task → one `SkillsExecutor` call, with parameters |
| Vocabulary | `agents/skill_catalog.py` | The closed skill list, generated from `SKILL_REGISTRY` |
| Loop | `orchestrator/agent_orchestrator.py` | Executes, tracks the gripper, replans on failure |

Three things are deliberate and worth knowing:

- **The model does not author trajectories.** It picks the skill and its
  arguments; the motion inside each skill is hand-written and grounded by SAM 3.
  This is a weaker autonomy claim than robomail_Aliyah's "all motion comes from
  the LLM Step Planner", and every trial record says so under `provenance`.
- **Object names are grounded, not captions.** A name goes straight to
  `VisionSystem.locate`. Where the cell can enumerate its bench — the simulator
  can — the scene agent must name objects from that inventory, so it cannot plan
  against a "measuring scoop" the bench calls a "larger spoon".
- **Tool offsets come from perception.** `pick_up` measures where a grasped
  tool's working end sits and reports it; the orchestrator hands that
  measurement to the agent planning the next step rather than letting the model
  guess a number it cannot see.

Instead of a typed task, `--instruction-image` reads the goal off a photographed
protocol page.

**Status.** Exercised end to end in `--sim`: plan, execute, an autonomous replan
after a failed `place`, success. With granules on, 2 of 90 citric acid grains
reached the paper cup — the scoop's yield is limited by `measure_tool_offset`
reading the bowl 24 mm shallower than CAD, which is a perception limitation that
predates this pipeline (see `progress.md`). **Not yet run on the arm.** The hardware path
differs in two ways worth watching the first time: object names are open
vocabulary rather than a bench inventory, so a name the scene agent invents can
fail to ground; and verification is on by default.

---

## Calibration & perception

1. Prefer **live RealSense factory intrinsics** (old `.intr` files were corrupt).
2. Extrinsics: green cube in gripper + Kabsch (`scripts/calibrate_cameras.py`).
3. Validate with `scripts/validate_calibration.py` / `scripts/check_object.py`.
4. Prefer prompt **`plastic beaker`** for the red translucent beaker; use **`--backend sam3`**.
5. The printed scoop is the prompt **`white plastic tool`**. Appearance
   matching for the stirrer and holder is opt-in (`--exemplars exemplars`);
   it is off by default because DINOv2 plus a second SAM exhausts RAM
   (`exemplars/README.md`).

---

## Simulation (no robot, no cage)

`robochem/sim/` runs the **same skills** against a MuJoCo Franka Panda so motions
can be watched before they touch hardware. The arm is mujoco_menagerie's vendor
Panda model with the Franka Hand; the four cage cameras are placed at the
extrinsics in `calibration_out/` and given the cage's frame geometry (848x480 at
a 42.5° vertical field, so fx = fy ≈ 617 as the D435 reports), rendered for
depth and segmentation, and back-projected through the project's own
`ObjectLocalizer` / `VisionSystem` fusion — so perception really reconstructs
the props from four views, with real self-occlusion, and rejects them when the
views disagree.

`SimFrankaArm` implements the frankapy surface the skills use and `SimVision`
the `VisionSystem` surface, so `SkillsExecutor` is unmodified. The simulator has
**no frankapy and no ROS**, so it runs out of `perception_env` (Python 3.10),
*not* the robot venv — do **not** `source scripts/env.sh` first.

### Setup (once)

```bash
bash scripts/setup_sim.sh
```

Reuses `perception_env/` if present, otherwise builds `sim_env/`; installs the
dependencies, pre-fetches the Panda MJCF into `~/.cache/robot_descriptions/`,
and runs the smoke test. Full setup notes, including a fresh machine after a
`git pull`, are in [`robochem/sim/README.md`](robochem/sim/README.md).

### Run a skill in the viewer

```bash
perception_env/bin/python scripts/run_experiment.py --sim \
  --skill pick_up --params '{"object_name":"plastic beaker","z_offset":0.02,"squeeze":0.013}'

perception_env/bin/python scripts/run_experiment.py --sim --sim-granules \
  --skill pour --params '{"target_container":"white paper cup","pour_angle":90,"hold_duration":2,"forward_offset":-0.08}'

perception_env/bin/python scripts/run_experiment.py --sim \
  --workspace-min 0.25 -0.40 -0.13 \
  --skill scoop --params '{"powder_source":"citric acid","tool_length":0.08}'
```

| Flag | Effect |
| --- | --- |
| `--sim-speed N` | Genuinely moves the arm N× faster (compresses commanded durations). Fling-and-spill at high N is real, not an artifact — use `1.0` when granule behaviour matters. |
| `--sim-fast` | Keeps commanded durations, just skips the wall-clock wait. This is the flag for "same motion, less waiting". |
| `--no-viewer` | Headless (CI, remote shells) |
| `--sim-granules` | Loose particles in the beaker and reagent cups, so pours and scoops move material |
| `--sim-tool-length M` | Bolt a fixed tool of length M onto the hand and move `franka_tool` out to its tip |
| `--sim-grasp-mode` | `magnet` (kinematic attach on close, default) or `physics` (friction contacts) |
| `--sim-hold S` | Seconds to keep the window open afterwards (default: until you close it) |

### From Python

```python
from robochem.sim import build_cell

cell = build_cell(viewer=True, granules=True)
cell.skills.execute("pick_up", {"object_name": "plastic beaker", "z_offset": 0.02})
cell.arm.hold()      # keep watching
cell.close()
```

Edit the bench (what is on the table, and where) in `robochem/sim/bench.py`.

### What it is and is not

- Cartesian targets are solved by damped least-squares IK over the seven arm
  joints and tracked to **under a millimetre**, with gravity compensated as the
  real controller does. An unreachable target leaves the arm short, so the
  skills' own arrival checks fire rather than being papered over.
- Reconstructed centroids land within a few millimetres of truth for the cups;
  the spoon reads ~25 mm off its body origin because its point cloud includes
  the bowl.
- It does **not** model fluid, powder rheology, RealSense noise characteristics,
  frankapy impedance, or force thresholds. Granules are bouncy spheres — they
  show *where a stream goes*, not how a powder behaves.

---

## Smoke tests

```bash
# Gripper / small moves only (no vision)
python scripts/smoke_test_motion.py
python scripts/smoke_test_motion.py --move

# Pick (+ optional pour) with pauses
python scripts/smoke_test_skills.py
python scripts/smoke_test_skills.py --pour
```

No hardware needed (run these from `perception_env`, without `scripts/env.sh`):

```bash
# Skill geometry and failure logic against a fake arm
perception_env/bin/python scripts/test_skills_offline.py

# The agent loop against a fake model: vocabulary, parameter checking,
# gripper bookkeeping, replanning. Needs no API key.
perception_env/bin/python scripts/test_agents_offline.py

# Perception + motion + pick/pour/scoop against the simulated cell
perception_env/bin/python scripts/smoke_test_sim.py
perception_env/bin/python scripts/smoke_test_sim.py --viewer --speed 2
```

---

## Related docs

| Doc | Contents |
| --- | --- |
| [`progress.md`](progress.md) | Current validated commands, design choices, known issues |
| [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) | Broader architecture plan |
| [`SKILLS_ROADMAP.md`](SKILLS_ROADMAP.md) | Skill hardening roadmap |
| [`robochem/sim/README.md`](robochem/sim/README.md) | MuJoCo preview of the cell: setup, usage, troubleshooting |
| [`robomail/docs/progress.md`](robomail/docs/progress.md) | Older PLATO / agent-pipeline notes |

---

## License / attribution

Research software for the Autonomous Robotic Chemist project (Barati Farimani Lab). Robot control depends on frankapy / Franka; perception on Meta SAM 3 and RealSense; simulation on MuJoCo and the Franka Panda model from DeepMind's mujoco_menagerie (Apache-2.0).
