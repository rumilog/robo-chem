# Skills roadmap — what the agent stack needs, and how to build it

Last updated: 2026-09-16

Two halves of this project are converging:

- **`robochem/`** (this repo's root) — the real Franka + RealSense manipulation
  stack. `pick_up` and `pour` are bench-validated; see [`progress.md`](progress.md).
- **[`robomail_Aliyah/`](robomail_Aliyah/)** — the PLATO-derived agent pipeline
  (goal extraction → planning → affordance → step planning → verification →
  logging). It runs end to end against live models, but **its skill executor is
  a stub**: `robomail_Aliyah/skills/executor.py` fakes every success from
  `FAKE_SUCCESS_RATE` and the arm never moves.

This document answers: *which skills does the agent stack expect, how do we
build them to the standard `pick_up`/`pour` set, and how do the two halves
eventually join up.*

---

## 1. The contract: six actions, closed and exhaustive

`robomail_Aliyah/agents/action_vocabulary.py` defines a **closed** enum. The
Step Planner is constrained to it by JSON schema and *cannot invent an action*;
anything it cannot express raises `InfeasibleStep`, which is logged and fed back
into replanning. So this list is the complete set of skills the agent side will
ever ask for.

| Action | Semantics (verbatim from their prompt block) | robochem skill | Status |
| --- | --- | --- | --- |
| `PICKUP` | Grasp an object (tool or container) and lift it | `pick_up` | **Bench-validated** |
| `POUR` | Tip a held container so its contents flow into a target | `pour` | **Bench-validated** |
| `PLACE` | Set a held object down and release, freeing the gripper | `place` | **Bench-validated 2026-09-11** |
| `SCOOP` | Use a held scoop to transfer a measured quantity of powder | `scoop` | **Bench-tuned 2026-09-16** (4 runs — see §3) |
| `STIR` | Agitate a container's contents with a held implement | `stir` | **Rewritten, needs bench time** |
| `PIPETTE_DISPENSE` | Use a held pipette to **draw and release** a small volume | `dispense` | **Rewritten, needs bench time** |

**So: four skills were missing in practice.** All four existed as files, but
none of them had been through the hardening that `pick_up` and `pour` went
through — they used fire-and-forget motion, misread object height, and returned
`success=True` regardless of what the arm did. All four are now rewritten
(§3–§4) and pass 79 offline checks. `place` and `scoop` have since been
bench-validated; `stir` and `dispense` have not yet run on the robot.

`PLACE` deserves emphasis. The robot profile
(`robomail_Aliyah/config/robot_profile.py`) states *"Only ONE object may be held
at any moment. To use a second object, the first must be PLACEd down first"*,
and the planner is directed to *"insert an explicit PLACE before any step that
needs a different tool."* Every multi-reagent plan is therefore
PICKUP → use → PLACE → PICKUP → use → PLACE. **Without a reliable `place`, no
plan longer than one tool can run**, regardless of how good pick and pour are.

### Not in the vocabulary

Our registry also has `move_to`, `tilt`, `wait`, `open_gripper`/`close_gripper`,
`analyze_scene`, `locate_object`, `check_container`. The agent stack will never
call these — they are useful for our own VLM orchestrator and for bench
debugging, and they do not need the same hardening.

---

## 2. The hardening playbook

Distilled from what actually made `pick_up` and `pour` work. Each pattern below
is there because its absence caused a specific failure. Apply all of them to any
new skill.

### P1. `reset_joints` before every scan

Not a hardcoded XYZ park. From an arbitrary pose a cartesian retract is often
unreachable, and scanning with the arm over the table occludes the object and
pollutes the fused cloud. Joint-space home is always reachable.

Safe while holding something — `pour` does exactly this with the beaker in the
gripper. Helper: `BaseSkill.clear_cameras(params, tag=...)`, honouring
`reset_before_scan` (default `True`) and `retract_xyz` as the escape hatch.

### P2. Measure the object in the world frame, never from `dimensions[2]`

`vision.get_object_dimensions()` returns **PCA extents sorted descending**, so
`dimensions[2]` is the object's *smallest* extent. For an upright beaker
(120 mm tall, 40 mm across) that is the **diameter**. Every skill that did
`target_pos[2] + dims[2]` to find the rim was aiming ~80 mm low, straight into
the cup.

Use `BaseSkill.locate_container(name)`, which returns from the raw cloud:

| Key | Meaning |
| --- | --- |
| `top_z` | 97th-percentile Z — the rim. Percentile, not `max`, so a few depth outliers do not set the height |
| `base_z`, `height` | 3rd-percentile Z, and the difference |
| `rim_center` | **Median XY of the top 15 mm band.** Cage coverage is one-sided, so the fused cloud leans toward whichever cameras saw the object and the full-cloud centroid sits off the opening |
| `rim_radius` | 90th-percentile radial distance within that band |

### P3. Every motion is verified, or it did not happen

`move_to_position` was fire-and-forget: it issued `goto_pose` and returned
`True` unconditionally. That is the single largest source of false successes in
the old skills. It now takes `reach_tol` and re-reads the pose; `goto_pose_rigid`
+ `reached()` is the position-and-orientation version.

Tolerances are not uniform, and the reason matters:

| Phase | Tolerance | Why |
| --- | --- | --- |
| Free-space hover | 50 mm | Coarse; only gross unreachability matters |
| Descent into/above a container | 25–30 mm | Tight — this is where being high means dropping or missing |
| Contact motion (digging powder) | 50 mm | Loose — the medium resists, so the arm legitimately stops short |
| Circle waypoints while stirring | 50 mm | Loose — the tool is in fluid; only a jam matters |

### P4. Fail loudly rather than succeeding quietly

`pour`'s hardest-won lesson: it used to report `success=True` at a measured 39°
while having commanded 90°. Now the tip is measured after every step and a
shortfall fails the skill.

Generalise: **measure the thing the skill exists to achieve.** `scoop`'s bite
tilt is measured before it commits to the powder — pushing level collects
nothing, so an unverified tilt is not a scoop worth attempting.

**But know what your measurement cannot see.** `tool_tip_deg()` is an `arccos`
magnitude: it reports *how far* from vertical, never *which way*. A tilt in
exactly the wrong direction passes it cleanly. `scoop` shipped that bug to the
bench twice. Where direction matters, check a signed quantity — `pour` checks
`lean·dir` against its intended lean vector, which is the pattern to copy.

### P5. Retry a lagging step before giving up

frankapy often needs a second, longer `goto_pose` before the wrist tracks a
commanded orientation. `pour` retries each tip step up to `step_retries` times
with a lengthening duration; `scoop` does the same for its bite tilt.

Retry the **remaining** delta, not the original command —
`rotate_about_tool_axis` is relative, so re-commanding the full angle stacks
rotations and overshoots.

Better still, where a *sequence* of orientations is needed, compute them
analytically against a fixed reference and command absolute targets — see
`scoop.tool_y_delta` and its use across the push. Absolute targets are
idempotent under retry; relative nudges are not.

### P6. Tool-frame rotation for anything gripper-relative

`rotate_wrist` left-multiplies (base frame); `rotate_about_tool_axis`
right-multiplies (tool frame). For any tip that must keep the jaws
table-parallel — pouring, lifting a scoop bowl — the tool axis is the correct
one. The base-frame version rolls the bowl over.

### P7. `use_impedance=False` for orientation, `True` for contact

`pour` only started tracking commanded orientation once it stopped using the
impedance controller, which quietly settles tens of degrees short.
`goto_pose_rigid` defaults to `use_impedance=False` for that reason. Invert it
for contact-rich motion: `scoop`'s dig and drag pass `use_impedance=True`,
because yielding to the powder bed is the desired behaviour.

### P8. Gripper width is the only grasp sensor — use it constantly

Soft paper cups give no useful contact force, so `grasp=True` force-limited
closing just keeps squeezing and crushes them. `pick_up` closes to
*measured diameter − squeeze* with `grasp=False` (a position command) instead.

Every skill after `pick_up` runs with something already held, and the commonest
silent failure is that it was dropped three motions ago while the skill happily
reported success. Record the width at skill start and re-check it with
`check_still_holding()` **before, during and after** the motion — `stir` checks
once per revolution.

Never call `open_gripper()` to relax a squeeze: it opens to 80 mm and drops the
tool. Command the held width back instead.

### P9. Clamp geometry to what was measured

A 15 mm stir circle in a 12 mm-radius cup scrapes the wall and tips it over; a
50 mm scoop drag in a 35 mm tub rams the far wall and stalls the arm. Clamp
against `rim_radius` and refuse when there is genuinely no room, rather than
attempting it.

### P10. `clear_cache()` after the scene changes

The localizer only reads from its cache. After any pick, place, pour or scoop
the cached segmentation is stale and the next skill will act on where the object
*used to be*.

### P11. Blocking waypoints, not a command stream

`stir` previously issued non-blocking `goto_pose` calls in a 50 Hz loop.
frankapy treats each `goto_pose` as a new *skill*, so that pattern starts and
cancels hundreds of skills a second: the arm judders, barely tracks the circle,
and any reach check is meaningless. Walk a modest number of blocking waypoints
instead — slower, but it is motion that actually happens and can be verified.

### P12. Say what you did not measure

`scoop` returns `quantity_measured: False`; `dispense` returns
`volume_measured: False`. Nothing weighs the powder or meters the drops — the
quantity is nominal, set by dig depth and drag length. The robot profile already
concedes this (*"quantities are metered by scoop or pipette"*), and the trial
log should not imply a precision that does not exist.

---

## 3. What each new skill does

### `place` — [`robochem/skills/place.py`](robochem/skills/place.py)

Record held width → clear cameras → scan → measure support surface → hover →
descend (verified) → open → **confirm release** → retract → clear cache.

- **The gripper is never opened from a pose that was not reached.** If the
  descent stops short the skill goes back up, keeps hold, and returns
  `still_holding: True` so the caller knows the object is still in the gripper.
- Defaults to placing *beside* a located target, not on top of it — placing onto
  a container is almost never what a chemistry step wants. `on_top: true` stacks.
- Accepts explicit `[x, y]` or `[x, y, z]` coordinates and then skips the scan
  entirely. This is the path the agent bridge uses for "put the scoop back".

### `stir` — [`robochem/skills/stir.py`](robochem/skills/stir.py)

Clear cameras → scan → clamp radius to the opening → flatten wrist → hover →
descend to immersion depth → walk N revolutions of discrete waypoints → lift.

- Radius clamped to `rim_radius − wall_clearance`; refuses below 3 mm of usable
  radius rather than scraping.
- Grasp re-checked once per revolution; a mid-stir drop aborts instead of
  stirring with an empty gripper for the remaining revolutions.
- Reports `revolutions_completed` so a partial stir is visible in the log.

### `scoop` — [`robochem/skills/scoop.py`](robochem/skills/scoop.py)

Clear cameras → scan → clamp stroke to the opening → hover level (stiff) →
**tilt forward and verify** → plunge in while advancing (tilted, compliant) →
**push away from the base, rolling back to level over the last stretch**
(compliant waypoints) → lift level (stiff).

This one took four hardware runs to get right, and every correction is worth
recording because none of them were visible offline.

**Run 1 — the tilt was in the wrong place.** The original design dragged level
and only tilted *afterwards*, to retain the powder for the lift. That plows the
powder flat instead of collecting it. A scoop has to bite in at an angle before
it moves through the medium, the way pour tips toward its site before liquid
can flow. Moved the tilt ahead of the stroke.

**Run 2 — the tilt direction was backwards, and nothing could have caught it.**
`tool_tip_deg()` measures tilt *magnitude* from vertical via `arccos(-R[2,2])`,
which is always ≥ 0. It cannot distinguish "tilted forward" from "tilted the
wrong way" — only how far from vertical. So the verification reported "achieved
30° of 30° commanded" as a clean success while the wrist leaned the opposite
way. Fixed by negating the angle into `rotate_about_tool_axis`. **This blind
spot applies to any skill that verifies a tool-axis rotation by magnitude** —
pour avoids it by construction, because it tips toward a computed lean vector
and checks `lean·dir`, not a bare angle.

**Run 2 also — the plunge had no forward component.** It descended straight
down and only then started moving. Real digging is already moving into the
medium as it enters. Added `dig_advance`: the entry stroke carries travel.

**Run 3 — the push disappeared.** `dig_advance` was subtracted out of a total
`scoop_distance`, so passing `scoop_distance == dig_advance` silently produced
a zero-length push. Replaced the derived arithmetic with two independent
distances, `dig_advance` and `drag_distance`, because a parameter that can
silently zero out another one is a bad parameter. `scoop_distance` is now
rejected with a message naming its replacements rather than being ignored.

**Run 4 — direction, and the un-tilt belongs inside the push.** Two changes:

- The stroke now pushes **away from the robot base (+X)**, not toward it. (The
  earlier note here claiming toward-base was needed for reach was wrong for
  this motion — that constraint came from `pour`, where the payload is a full
  container at arm's length, not a scoop inside a cup.)
- The roll back to level happens **during** the push, blended across the last
  `untilt_over` metres, rather than as a separate step after the motion stops.
  Closing the bowl over the powder while it is still moving is what keeps the
  load on; stopping and then flicking upright drops it back in the cup.

**Run 5 — the push is continuous, not subdivided.** The first version of the
blend walked eight discrete waypoints and visibly stepped, because each one
decelerated to a stop. frankapy's `goto_pose` is *already* a min-jerk
interpolation of translation **and** orientation, so subdivision buys nothing
here. The push is now two `goto_pose` calls:

| Segment | Motion | Why two, not one |
| --- | --- | --- |
| A | translate `drag_distance − untilt_over` at the full bite angle | a single min-jerk segment interpolates orientation monotonically end-to-end, so it cannot hold an angle and *then* roll off |
| B | translate the last `untilt_over` while rotating to level | |

Setting `untilt_over == drag_distance` collapses this to one unbroken motion
that rolls level across the whole push. Segment durations split proportionally
to distance out of `push_seconds`.

Note this does **not** contradict P11 (stir's blocking waypoints). That lesson
is about a 50 Hz *non-blocking* stream, where frankapy starts and cancels
hundreds of skills a second. A handful of blocking min-jerk segments is the
opposite situation — and where the motion should read as continuous, fewer is
better.

The target orientations are computed analytically — `level_rotation @
tool_y_delta(-angle)` — rather than by issuing relative nudges and hoping they
compose. The final target is exactly `level_rotation`, so the stroke ends level
by construction rather than by a convergence loop.

| Parameter | Default | Role |
| --- | --- | --- |
| `dig_advance` | 20 mm | Forward travel during the tilted plunge |
| `drag_distance` | 40 mm | The push, away from the base |
| `untilt_over` | 20 mm | Tail of the push across which tilt → level |
| `dig_tilt_deg` | 30° | Bite angle (direction corrected run 2) |
| `push_seconds` | 3.0 s | Duration of the push, split across its segments by distance |
| `wall_clearance` | 10 mm | Lowered from 15 mm — it was clamping strokes that fit |

The whole stroke (`dig_advance + drag_distance`) is clamped to the measured
opening, scaling both parts together so their ratio survives. Dig depth is
measured from the top of the cloud *inside* the container, which for a
part-full tub is the powder surface, not the rim.

**Still not confirmed**: whether 30° is the right bite angle, and whether the
quantity collected is repeatable. Nothing weighs the scoop — `quantity_measured`
is `False` for a reason.

### `dispense` — [`robochem/skills/dispense.py`](robochem/skills/dispense.py)

Three modes, because `PIPETTE_DISPENSE`'s stated semantics are "draw **and**
release":

| Mode | Behaviour |
| --- | --- |
| `dispense` (default) | Squeeze/release pulses with the tip above the target rim |
| `aspirate` | Squeeze **in air**, lower the tip into the source, release to draw in |
| `transfer` | `aspirate` from `source_container`, then `dispense` into `target_container` |

- Squeezing before entering the liquid is the whole point of the aspirate
  ordering — squeeze while submerged and the air is blown into the source.
- Squeeze depth is floored at `min_width_fraction` (0.55) of the held width, so
  no squeeze can crush the barrel or pop the pipette out of the jaws.
- The tip stays *above* the target rim while dispensing; dipping it in would
  contaminate the pipette and wick liquid back out.

---

## 4. Bench validation — the order to do it in

None of the four has touched hardware. Offline first, then the robot.

```bash
# No robot, no cameras. 57 checks on geometry, failure paths and the bridge.
perception_env/bin/python scripts/test_skills_offline.py
```

Then, with the arm and grounding service up (see [`progress.md`](progress.md)):

```bash
source scripts/env.sh
perception_env/bin/python perception_service/grounding_service.py \
  --backend sam3 --model weights/sam3.pt   # separate terminal
```

**Step 1 — `place`, because everything else depends on it.** Nothing is held
yet, so pick first:

```bash
python scripts/run_experiment.py --skill pick_up \
  --params '{"object_name":"plastic beaker","z_offset":0.02,"squeeze":0.013}'
python scripts/run_experiment.py --skill place \
  --params '{"target_location":[0.45,-0.15,0.02]}'
```

Start with explicit coordinates (no scan in the loop), then try
`{"target_location":"white paper cup"}` to exercise the beside-a-target path.

**Step 2 — `stir`.** Needs a stirrer the gripper can hold. Start shallow and
slow in a wide container:

```bash
python scripts/run_experiment.py --skill pick_up --params '{"object_name":"stirring rod"}'
python scripts/run_experiment.py --skill stir \
  --params '{"target_container":"white paper cup","revolutions":1,"stir_depth":0.02,"seconds_per_waypoint":1.0}'
```

**Step 3 — `dispense`.** Dry run first (`num_drops: 1`, empty pipette) to see
whether the squeeze travel actually deforms the bulb without losing the grasp.
Tune `squeeze_amount` upward from 3 mm only as far as the grasp survives.

**Step 4 — `scoop`.** Last, because it is the most contact-heavy. Use a shallow
open tub, not a deep one.

### The parameter you must measure first: `tool_length`

`stir`, `scoop` and `dispense` all default `tool_length` to **0.0**, which means
the arm commands the *gripper*, not the tool tip, to the target depth. That is
correct only for a zero-length tool.

**Measure the distance from the gripper TCP to the working end of each tool and
pass it.** Left at 0, a 90 mm stirrer will be driven 90 mm too deep — through
the bottom of the cup.

### Other values that need bench numbers

| Parameter | Skill | Default | Note |
| --- | --- | --- | --- |
| `wall_clearance` | `stir` | 12 mm | Conservative. A 13 mm-radius cup leaves only 1 mm of usable radius and the skill refuses — lower it once the stirrer's real width is known |
| `squeeze_amount` | `dispense` | 3 mm | Enough to deform a bulb? Unknown until tried |
| `scoop_depth` | `scoop` | 15 mm | Nominal; with `dig_advance`/`drag_distance` it sets the quantity, which nothing measures |
| `release_clearance` | `place` | 20 mm | Soft cups may need more; a rigid beaker less |

---

## 5. Integrating with `robomail_Aliyah`

### What is already built

[`robochem/integration/plato_bridge.py`](robochem/integration/plato_bridge.py)
— `RoboChemExecutor` is a drop-in replacement for their stub executor:

```python
from robochem.integration.plato_bridge import RoboChemExecutor

orchestrator = Orchestrator(cell=cell)
orchestrator.executor = RoboChemExecutor(
    skills_executor,                                    # robochem SkillsExecutor
    param_overrides={"pour": {"forward_offset": -0.08}},  # bench-tuned values
    arm=cell.arm,                                       # for gripper bookkeeping
)
```

It is field-compatible with their `ExecutionResult` (`action`, `success`,
`reason`, `primitives_issued`, `simulated_seconds`, `is_fake`, `detail`,
`as_dict()`), so the whole logging path works unchanged — with `is_fake=False`,
which is the entire point. It provides:

- **Name translation.** The planner speaks the robot profile's *location*
  vocabulary (`"Original Position of clear cup 1"`); SAM 3 wants an object
  phrase (`"clear cup 1"`). `normalize_object_name()` strips the prefix.
- **Per-action parameter mapping**, covering all six actions. An unmapped or
  under-specified action fails explicitly rather than no-opping.
- **Real gripper-state guards.** Their `validate_sequence()` checks the plan;
  the bridge checks the *robot*. A plan that passes their checker can still
  arrive with an empty gripper because an earlier pick actually failed, and a
  `POUR` is then refused before the arm moves.
- **Pick-site memory.** `"PLACE at Original Position of beaker"` while holding
  the beaker cannot be segmented — the beaker is in the gripper, not there. The
  bridge records where each object was picked from and hands `place` those
  coordinates.
- **`build_position_lookup()`** — grounds the profile's workspace positions into
  base-frame coordinates. `hardware/real.py::RealFrankaArm.goto_delta` raises
  without one, noting *"the position lookup is populated by the perception stack
  at run start"*; this is that function, for anyone who does want the
  primitive-replay path instead.

The bridge imports nothing from `robomail_Aliyah` at load time — steps are
consumed structurally — so it stays importable and testable off the robot PC.
`scripts/test_skills_offline.py` cross-checks the mapping against their real
`action_vocabulary.py` when the package is present.

### The architectural decision this forces

Their definition of done says: *"No hand-scripted/fixed-trajectory motion code
anywhere. All motion comes from the LLM Step Planner."*

**That does not survive this integration, and it should not be inherited
silently.** Their step planner emits centimetre-level GOTO/GRASP/TILT
primitives. A robochem skill re-grounds its target with SAM 3, computes its own
grasp or rim geometry, and verifies arrival — feeding it "move 4 cm in −X" on
top of that would fight it. So the bridge **logs the primitives but does not
replay them**, and the division of labour becomes:

- the LLM chooses **which skill** and **with what arguments**;
- the trajectory inside the skill is hand-written and perception-grounded.

That is real manipulation in exchange for a weaker autonomy claim.
`executor_provenance()` returns the exact wording to write into the trial log so
the paper's Experimental Setup section matches the code that ran.

The alternative — LLM-authored trajectories against a `position_lookup`, using
their `RealFrankaArm` — is still available and `build_position_lookup()` exists
for it. It preserves the stronger claim, and gives up SAM-grounded grasp
selection, reach verification and every failure check in §2. **This is a genuine
decision to make deliberately, not a detail.**

### Still open before an end-to-end run

1. **Their fourth known bug bites harder with real skills.** Their own notes:
   the scene is grounded once per trial and never re-grounded, so *"pipette the
   sodium bicarbonate solution from the clear cup"* is judged against the
   original scene where that cup was empty. On a real robot the *positions* also
   go stale, not just the contents. Our skills re-scan per call (P1, P10), which
   partly covers it — but the **planner's** world model still needs either
   per-sub-task re-grounding or a symbolic world state. Their doc calls this
   "the most substantive open question in the architecture"; it is unchanged.
2. **Object vocabulary alignment.** Their kit names
   (`config/robot_profile.py::workspace_positions`) must actually ground on our
   bench. `progress.md` already records that `"plastic beaker"` detects far more
   reliably than bare `"beaker"` — the profile's names may need the same
   treatment, or a synonym table in the bridge.
3. **Verification camera.** Their `VerificationAgent` assumes one fixed view
   with a fixed cup ROI (`hardware/mock.py::CUP_ROI`). Our cage is four
   sequentially-captured RealSenses. Something has to map one to the other, and
   their open question 3 (camera setup, research plan §6) is the place it gets
   decided.
4. **Real failures will change the replanning loop's character.** Their cap of 3
   replans was tuned against a stub with `FAKE_SUCCESS_RATE=1.0`. Real skills
   fail for reasons a replan cannot fix — an unreachable pose, a bloated SAM
   mask. Their `failure_kind` split (`planning` / `execution` / `verification`)
   already exists to keep those honest; the bridge populates it correctly, but
   nobody has looked at the resulting distribution yet.

---

## 6. Summary of what changed in this pass

| File | Change |
| --- | --- |
| [`robochem/skills/base_skill.py`](robochem/skills/base_skill.py) | `locate_container`, `clear_cameras`, `check_still_holding`, `tool_down_rotation`, `tool_tip_deg`, `goto_pose_rigid`, `reached`; `move_to_position` gained optional reach verification |
| [`robochem/skills/place.py`](robochem/skills/place.py) | Rewritten — never releases from an unreached pose |
| [`robochem/skills/stir.py`](robochem/skills/stir.py) | Rewritten — blocking waypoints, clamped radius, mid-stir grasp checks |
| [`robochem/skills/scoop.py`](robochem/skills/scoop.py) | Rewritten — compliant dig/drag, verified retaining tilt, clamped drag |
| [`robochem/skills/dispense.py`](robochem/skills/dispense.py) | Rewritten — aspirate/dispense/transfer, floored squeeze |
| [`robochem/integration/plato_bridge.py`](robochem/integration/plato_bridge.py) | New — `RoboChemExecutor` replacing `robomail_Aliyah`'s stub |
| [`scripts/test_skills_offline.py`](scripts/test_skills_offline.py) | New — 57 offline checks, no robot required |

**All 57 offline checks pass. None of the four skills has run on the robot yet.**
