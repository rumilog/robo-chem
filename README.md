# robo-chem

Autonomous robotic chemist stack for a **Franka Panda** arm in a **RealSense** camera cage. Skills use multi-view SAM 3 segmentation, fused 3D localization, and frankapy motion. A VLM orchestrator can plan tasks; individual skills can also be run directly from the CLI.

Barati Farimani Lab, CMU.

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
  --backend sam3 --model weights/sam3.pt
```

Service listens on `http://127.0.0.1:5005` by default.

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
│   ├── vision/               # multi-cam localizer, grasp analyzer, grounding client
│   ├── orchestrator/         # VLM task planning
│   └── verification/         # chemistry outcome checks
├── perception_service/       # Flask SAM 3 / GroundingDINO service
├── scripts/
│   ├── env.sh                # ROS + frankapy + PYTHONPATH
│   ├── run_experiment.py     # main skill / task entry
│   ├── calibrate_cameras.py  # board-free cube calibration
│   ├── capture_scene.py      # dump color/depth stills
│   └── smoke_test_*.py
├── weights/                  # sam3.pt, etc.
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

Natural-language tasks (needs API key):

```bash
python scripts/run_experiment.py --task "Pour the water into the beaker"
```

---

## Calibration & perception

1. Prefer **live RealSense factory intrinsics** (old `.intr` files were corrupt).
2. Extrinsics: green cube in gripper + Kabsch (`scripts/calibrate_cameras.py`).
3. Validate with `scripts/validate_calibration.py` / `scripts/check_object.py`.
4. Prefer prompt **`plastic beaker`** for the red translucent beaker; use **`--backend sam3`**.

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

---

## Related docs

| Doc | Contents |
| --- | --- |
| [`progress.md`](progress.md) | Current validated commands, design choices, known issues |
| [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) | Broader architecture plan |
| [`SKILLS_ROADMAP.md`](SKILLS_ROADMAP.md) | Skill hardening roadmap |
| [`robomail/docs/progress.md`](robomail/docs/progress.md) | Older PLATO / agent-pipeline notes |

---

## License / attribution

Research software for the Autonomous Robotic Chemist project (Barati Farimani Lab). Robot control depends on frankapy / Franka; perception on Meta SAM 3 and RealSense.
