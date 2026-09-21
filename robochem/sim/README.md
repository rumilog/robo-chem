# Simulated Franka cell (`robochem.sim`)

A MuJoCo model of this lab's cell — Franka Emika Panda, Franka Hand, four-camera
RealSense cage — that the **real skills run against unmodified**, so arm motions
can be watched and debugged with no hardware present.

> **Setting this up on a new machine?** Run `bash scripts/setup_sim.sh` from the
> repo root and skip to [Verify](#3-verify). The rest of this file explains what
> that does and how to use the result.

---

## What it actually simulates

`SkillsExecutor` talks to the arm through thirteen methods and to perception
through the `VisionSystem` surface. `SimFrankaArm` and `SimVision` implement
those two surfaces, so **no skill code is aware of the simulator**:

- **Arm** — mujoco_menagerie's vendor Panda MJCF (real link geometry, joint
  limits, Franka Hand). Cartesian targets are solved by damped least-squares IK
  over the seven arm joints and tracked to under a millimetre, with gravity
  compensated the way the real controller does. An unreachable target leaves the
  arm short, so the skills' own arrival checks fire instead of being papered over.
- **Perception** — each of the four cage cameras is placed at the measured
  extrinsics from `calibration_out/` and rendered for depth and segmentation in
  the real frame geometry: 848x480 at a 42.5° vertical field, which is the D435
  colour stream the cage runs and the intrinsics its depth is deprojected with
  (fx = fy ≈ 617, cx = 424, cy = 240). A prop at the edge of a real view is at
  the edge of the simulated one.
  The segmentation supplies the mask SAM 3 would have produced; everything after
  that is this project's own code — `ObjectLocalizer._depth_to_points`,
  `_transform_points`, `VisionSystem._filter_by_consensus`, `_check_plausible`.
  So a skill sees a genuine multi-view point cloud with real self-occlusion, and
  perception fails the same way when a container is hidden from too many cameras.
- **Grasping** — closing the jaws on a prop attaches it kinematically and reports
  the prop's width back through `get_gripper_width()`, which is what a real grasp
  stalling on an object looks like. `grasp_mode="physics"` uses friction instead.

### What it does not simulate

Fluids, powder rheology, RealSense noise characteristics, frankapy impedance
behaviour, or force thresholds. Granules are bouncy spheres: they show *where a
stream goes*, not how a powder behaves. Depth is also complete where the real
sensor drops out — the simulated beaker returns depth through its transparent
wall, so a passing sim run does not prove the real cage can see a clear
container.

---

## Setup

### 0. Prerequisites

| Need | Detail |
| --- | --- |
| Python | 3.9+. Verified on **3.10** and **3.13** |
| Disk | ~500 MB — mujoco_menagerie is cloned to `~/.cache/robot_descriptions/` on first use |
| Graphics | Any OpenGL. A display for the viewer window; `MUJOCO_GL=egl` for headless |
| **Not** needed | frankapy, ROS, RealSense, SAM weights, `OPENAI_API_KEY` |

The camera extrinsics (`calibration_out/realsense_camera{2,3,4,5}w.npy`) **are
committed**, so a fresh clone gets the real cage geometry. If they are ever
missing the scene falls back to a synthetic camera ring and says so:

```
[sim] No calibration in calibration_out; using a synthetic camera ring
```

`calibration_out/` holds the **measured** extrinsics from the last
`scripts/calibrate_cameras.py` sweep. The robot reads a *different* copy —
`robomail/robomail/vision/calib/`, which `calibrate_cameras.py --apply` installs
into. Until that install runs the two disagree (in this checkout by about 4 cm;
see `shift_from_current_mm` in `calibration_out/report.json`), and a simulated
camera is not where the robot believes its camera is. To preview against what
the robot is actually running, point the sim at the same files:

```bash
sim_env/bin/python scripts/run_experiment.py --sim \
  --sim-calib-dir robomail/robomail/vision/calib ...
```

### 1. Environment

Python environments are **not** in git (`perception_env/`, `sim_env/` are
ignored), so a fresh clone has none. Build one:

```bash
cd /path/to/robo-chem
bash scripts/setup_sim.sh
```

That reuses `perception_env/` if it exists, otherwise creates `sim_env/`, then
installs, pre-fetches the model and runs the smoke test. Pass a path to target a
specific env (`bash scripts/setup_sim.sh path/to/venv`), or set `SKIP_TEST=1` to
install without verifying.

> **Do not `source scripts/env.sh` first.** That activates the Python 3.8
> frankapy venv and sources ROS. The simulator needs neither and will not run
> there.

#### On Windows

`setup_sim.sh` assumes a POSIX venv layout (`bin/python`), so do it directly —
the interpreter lives in `Scripts\` and the rest is identical:

```powershell
winget install --id Python.Python.3.12 -e --scope user
& "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe" -m venv sim_env
sim_env\Scripts\python.exe -m pip install numpy mujoco robot_descriptions opencv-python scikit-learn openai Pillow
sim_env\Scripts\python.exe scripts\smoke_test_sim.py
```

The viewer and offscreen rendering both work through WGL; there is no `MUJOCO_GL`
to set, and `egl` is not available. If `import mujoco` dies with `WinError 1114`,
see Troubleshooting — the Visual C++ runtime needs updating.

<details>
<summary>Manual equivalent, if you would rather not use the script</summary>

```bash
python3 -m venv sim_env
sim_env/bin/pip install numpy mujoco robot_descriptions opencv-python scikit-learn openai Pillow
```

`numpy`, `mujoco` and `robot_descriptions` drive the simulation. `opencv-python`,
`scikit-learn` and `openai` are imported on the way into `robochem.vision`, which
`SimVision` subclasses so reconstruction runs through the real fusion code.
`Pillow` comes in through `ExperimentLogger`, which `run_experiment.py` imports
at module level whether or not the run is simulated. `open3d` is **not**
required — nothing on the simulated path uses it.

Check it with `sim_env/bin/python scripts/run_experiment.py --help`, which
exercises every module-level import and should exit 0.
</details>

### 2. First run downloads the model

The first call clones mujoco_menagerie (a few hundred MB) with a progress bar.
`setup_sim.sh` does this up front so a later run does not appear to hang.

### 3. Verify

```bash
sim_env/bin/python scripts/smoke_test_sim.py
```

Expected tail — **all five groups must pass**:

```
====================================================
  PASS  perception
  PASS  motion
  PASS  pick + pour
  PASS  pick + scoop
  PASS  failure handling
====================================================
all groups passed
```

Add `--viewer` to watch it (`--viewer --speed 2`). Takes a few minutes headless.

`--only` runs just the groups whose name contains what you pass, which is how you
watch one sequence end to end — the skills run against **one** cell, so the spoon
is still in the jaws when `scoop` starts:

```bash
sim_env/bin/python scripts/smoke_test_sim.py --viewer --granules \
  --only scoop --speed 2 --hold 0
```

`--hold 0` leaves the window open until you close it instead of the default five
seconds. Running `pick_up` and then `scoop` as two `run_experiment.py --sim`
commands does **not** work: each process builds its own cell, so the second one
starts with an empty gripper and `scoop` refuses.

---

## Running skills

Same entry point and the same `--params` as the hardware route, plus `--sim`:

```bash
sim_env/bin/python scripts/run_experiment.py --sim \
  --skill pick_up --params '{"object_name":"plastic beaker","z_offset":0.02,"squeeze":0.013}'

sim_env/bin/python scripts/run_experiment.py --sim --sim-granules \
  --skill pour --params '{"target_container":"white paper cup","pour_angle":90,"hold_duration":2,"forward_offset":-0.08}'

sim_env/bin/python scripts/run_experiment.py --sim \
  --workspace-min 0.25 -0.40 -0.13 \
  --skill scoop --params '{"powder_source":"citric acid","tool_length":0.08}'
```

| Flag | Effect |
| --- | --- |
| `--sim-speed N` | Genuinely moves the arm N× faster by compressing commanded durations. Spilling at high N is real, not an artifact — use `1.0` when granule behaviour matters |
| `--sim-fast` | Keeps commanded durations, only skips the wall-clock wait. **This** is the flag for "same motion, less waiting" |
| `--no-viewer` | Headless, for CI and remote shells |
| `--sim-granules` | Loose particles in the beaker and reagent cups so pours and scoops move material |
| `--sim-tool-length M` | Bolt a fixed tool of length M (metres) to the hand and move `franka_tool` out to its tip |
| `--sim-grasp-mode` | `magnet` (kinematic attach, default) or `physics` (friction contacts) |
| `--sim-calib-dir DIR` | Where to read `realsense_cameraNw.npy` from. Defaults to `calibration_out/`; point it at `robomail/robomail/vision/calib` to place the cage exactly where the robot currently believes it is |
| `--sim-hold S` | Seconds to keep the window open afterwards (default: until you close it) |

### From Python

```python
from robochem.sim import build_cell

cell = build_cell(viewer=True, granules=True)
cell.skills.execute("pick_up", {"object_name": "plastic beaker", "z_offset": 0.02})
cell.arm.hold()     # keep watching until the window is closed
cell.close()        # always: see the segfault note in Troubleshooting
```

`cell.arm` is the frankapy stand-in, `cell.vision` the `VisionSystem` stand-in,
`cell.skills` an ordinary `SkillsExecutor`. `cell.vision.ground_truth(name)`
gives a prop's true position, for checking what perception did.

### Headless (no display)

```bash
MUJOCO_GL=egl sim_env/bin/python scripts/smoke_test_sim.py
```

`egl` is verified working with `DISPLAY` unset. Offscreen rendering for the
cameras works regardless; only the interactive window needs a display, and it
degrades gracefully:

```
[sim] Viewer unavailable (...); running headless
```

---

## Layout

| File | Role |
| --- | --- |
| [`scene.py`](scene.py) | Compiles the MJCF: Panda + Hand, the `franka_tool` TCP site, the four cage cameras, the table, the props. Also builds the arm-only model used for IK |
| [`sim_arm.py`](sim_arm.py) | `SimFrankaArm` — the frankapy surface, DLS IK, Cartesian interpolation, kinematic grasping |
| [`sim_vision.py`](sim_vision.py) | `SimVision` — renders depth + segmentation, deprojects through the project's real fusion path |
| [`bench.py`](bench.py) | **What is on the table, and where.** Edit here to change the scene |
| [`rigid_transform.py`](rigid_transform.py) | `RigidTransform` shim, so the skills build poses without autolab_core (which exists only in the robot venv) |
| [`../../scripts/smoke_test_sim.py`](../../scripts/smoke_test_sim.py) | End-to-end check: perception, motion, pick+pour, pick+scoop, failure handling |
| [`../../scripts/capture_sim_scene.py`](../../scripts/capture_sim_scene.py) | Colour + depth stills from the simulated cage, in the same filenames `capture_scene.py` writes on hardware |
| [`../../scripts/setup_sim.sh`](../../scripts/setup_sim.sh) | One-command setup on a new machine |

### Changing the bench

Props are described semantically in `bench.py` and compiled to MJCF. Their names
are the queries the skills already use (`"plastic beaker"`, `"white paper cup"`,
`"larger spoon"`), and a `label` makes a cup resolve from a reagent name the way
the real labelled-cup path does:

```python
Prop(name="plastic beaker", pos=(0.46, 0.14), radius=0.035, height=0.095,
     rgba=(0.75, 0.85, 0.95, 0.55), fill=30)          # fill = granules inside
```

The table top is `z = 0`, which is also the Panda's base plane, matching the real
cell where object centroids land just above zero.

#### Props that carry their CAD

Where the real shape matters to perception, a prop names an STL instead of being
approximated. The spoon is the lab's printed scoop, straight off `spoon.stl`:

```python
Prop(name="larger spoon", kind="rod", pos=(0.58, -0.02),
     mesh="spoon.stl", mesh_scale=0.001,          # the CAD is in mm
     mesh_pos=(-0.015, 0.0, 0.0035),              # origin -> middle of the handle
     height=0.030, wall=0.004,                    # handle: 30mm long, 8mm jaws
     bowl_size=(0.01375, 0.01025, 0.00575),       # open box, half-extents
     bowl_offset=(0.0408, 0.0, -0.0223))
```

The mesh is what the cameras see and what perception reconstructs. It carries no
contacts: MuJoCo collides a mesh by its **convex hull**, and the hull of an open
scoop is a solid block that granules could never enter. Contact instead comes
from the primitives `bowl_size` and `bowl_offset` describe — a floor, four walls
and a handle box — which sit in group 3, so the renderer does not draw them and
the segmentation buffer stays mesh-only.

Put the body origin where the jaws close (`mesh_pos` shifts the mesh, not the
body): `sim_arm._try_grasp` tests that point against the jaw box, and
`Prop.grasp_width` is `2 * wall`.

---

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| `ModuleNotFoundError: No module named 'cv2'` (or `sklearn`, `openai`) | Incomplete install — `robochem.vision` needs them on import. Re-run `scripts/setup_sim.sh` |
| `ModuleNotFoundError: No module named 'mujoco'` | Running the wrong interpreter. The sim is not in the frankapy venv; use the env `setup_sim.sh` built |
| `OSError: [WinError 1114] A dynamic link library (DLL) initialization routine failed` on `import mujoco` (Windows) | The machine's Visual C++ 2015-2022 runtime predates the mujoco wheel, and every mujoco version fails the same way. Confirm it in Event Viewer — Application: a `python.exe` crash naming `MSVCP140.dll` with `0xc0000005`. Fix with `winget upgrade --id Microsoft.VCRedist.2015+.x64` (needs admin; 14.44 or newer works). A venv-local copy of the runtime does **not** help: System32 wins once the DLL is loaded |
| `ImportError: ... frankapy` | You sourced `scripts/env.sh`, or are running the hardware path. `--sim` imports frankapy lazily and never on the simulated route |
| `openai.OpenAIError: Missing credentials` | Only from constructing `VisionSystem` directly. `SimVision` handles this; no key is needed for simulation |
| Process exits `139` / **segmentation fault after the run finishes** | The MuJoCo viewer handle must be garbage-collected before interpreter shutdown. `cell.close()` (and `--sim`) already does this — call it, and do not hold your own reference to `cell.arm._viewer` |
| First run seems to hang | It is cloning mujoco_menagerie. Let it finish, or pre-fetch with `setup_sim.sh` |
| `[sim] jaws closed on nothing` | The gripper closed with no prop between the jaws. Usually a grasp pose that missed, which is the simulator reporting a real failure |
| `[SimVision] Cameras could not agree on 'x'` | Views disagree beyond `consensus_tolerance` — normally the arm occluding the prop. Same rejection the real stack makes |
| Granules fling everywhere | `--sim-speed` above ~2 genuinely accelerates the motion. Use `--sim-fast` instead when you want speed without changed dynamics |

---

## Known accuracy

Measured by `scripts/smoke_test_sim.py` on the default bench:

- Cartesian targets reached to **< 0.5 mm** (hardware `reach_tol` is 35 mm).
- Cup centroids reconstruct within **~1–13 mm** of truth from four cameras.
- The spoon reads **~28 mm** off its body origin — its point cloud includes the
  bowl while the origin is mid-handle. Expected, not an error. It reconstructs to
  **78 x 29 x 26 mm** against the STL's 69.5 x 20.5 x 31.5 mm, the spread being
  depth noise plus PCA axes that do not line up with the CAD axes.
- Pouring at `--sim-speed 1` with `--sim-granules` lands 27/30 particles in the
  target cup, with the median **~3 cm past cup centre in +X** — the `tip_advance_m`
  of 0.25 cm per 10° accumulating. Worth checking against hardware.
