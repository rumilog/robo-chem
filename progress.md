# Progress — robo-chem (Franka + RealSense)

Last updated: 2026-09-09

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
| `robochem/integration/plato_bridge.py` | Runs `robomail_Aliyah` plans on the real cell |
| `scripts/test_skills_offline.py` | Offline skill + bridge checks |
| `SKILLS_ROADMAP.md` | Skill gap, hardening playbook, integration plan |
| `robochem/vision/` | Localizer, SAM client, grasp analyzer |
| `perception_service/grounding_service.py` | SAM 3 / GDINO HTTP service |
| `scripts/env.sh` | ROS + frankapy + PYTHONPATH |
| `scripts/run_experiment.py` | Skill / task entry |
| `calibration_out/` | Extrinsics, taught poses, reports |
| `diag_out/` | Debug overlays / fused clouds |
