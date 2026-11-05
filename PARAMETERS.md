# Decision parameters — Magic Beaker bench

Frozen 2026-11-04. Status is either **FROZEN** (do not change between reported
trials) or **PENDING**, which needs a code change or a bench test first. Every
trial reported in the paper must run with the FROZEN values and with the PENDING
rows resolved. Record the git commit the trials ran on.

Source: *model* = computed from the arm model or geometry; *bench N* = N real
runs in the logs; *sim* = simulation only; *limit* = a hardware or software
limit; *estimate* = engineering judgement, no data.

## Preflight (`robochem/orchestrator/preflight.py`)

| Parameter | Value | Source | Status |
|---|---|---|---|
| `DUMP_TIP_TOO_LOW`: move a dump target that tips less than this | 60° | estimate (a bowl at 45-60° empties poorly) | FROZEN |
| `DUMP_TIP_WANTED`: tip a moved dump target must allow | 75° | model (`reachable_tip_deg`; 79° is the dump angle) | FROZEN |
| `TARGET_REACH`: pour / stir target distance from base | 0.30-0.65 m | bench 23 pours; furthest successful 0.62 m | FROZEN |
| `SCOOP_MAX_X`: farthest dish x (warning only) | 0.60 m | bench 27 scoops ok, jams at 0.604 / 0.642 | FROZEN |
| `MAX_FETCH_X`: farthest container fetched | 0.745 m | limit (hand clamp x 0.75), pick at 0.747 ok | FROZEN |
| `MAX_FETCH_REACH`: sanity radius | 0.78 m | limit | FROZEN |
| `MAX_MOVABLE_RADIUS`: widest container moved | 0.040 m | limit (jaws open 80 mm) | FROZEN |
| `SPOT_REACH`: moved container distance from base | 0.35-0.50 m | sim layout range, outer edge from model | FROZEN |
| `DUMP_SPOT_X`, `DUMP_SPOT_MAX_ABS_Y`: region for moved dump targets | x 0.39-0.46, \|y\| ≤ 0.14 | bench: 22 successful dumps at x 0.388-0.594, \|y\| ≤ 0.145 | PENDING: grid test |
| `DUMP_WIDE_X`, `DUMP_WIDE_MAX_ABS_Y`: fallback region | x 0.39-0.50, \|y\| ≤ 0.21 | estimate; 1 bench try at (0.42, +0.15), carry 22 mm off | PENDING: grid test, then merge with the row above |
| `MOVE_MAX_ABS_Y`: furthest sideways for any move | 0.21 m | estimate (a move to y −0.32 was refused) | PENDING: grid test |
| `DISH_CLEARANCE`: moved container to a scooped dish | 0.17 m | sim (hit at 144-161 mm, clear at 172) | PENDING: bench test |
| `FINGER_ROOM`: free space around a moved container | 0.06 m | estimate, no record | PENDING: bench test (with the row above) |

## Dump (`bench_manifest.py` skill_defaults → `robochem/skills/dump.py`)

| Parameter | Value | Source | Status |
|---|---|---|---|
| `min_tip_deg`: least tip that counts as emptied | 40° | bench: near-misses 59.9/60, 54.6/55, 44.6/45 | FROZEN (optional: residual-powder vs tip test) |
| `hover_tol_xy`: carry may end this far sideways from over the cup | 20 mm | model: cup r 36 − bowl half-width 10 − margin 6 | FROZEN |
| `hover_tol_below` / `hover_tol_above`: carry height vs plan | −10 / +40 mm | model: 10 mm rim clearance below; above only drops the powder further (both bench misses stopped 19-25 mm high) | FROZEN |
| `forward_offset` | 0.0 | bench | FROZEN |
| `max_bowl_drift` (warning only) | 0.015 m | bench | FROZEN |

## Scoop (`robochem/skills/scoop.py`, bench dish values in `bench_manifest.py`)

| Parameter | Value | Source | Status |
|---|---|---|---|
| Stroke | fixed: heading 0°, centre line | the user's choice; the map-chosen stroke behaved differently run to run | FROZEN |
| `depth_from_map` | on: surface = scan level over the first `entry_patch` 12 mm of the fixed stroke (where the bowl enters). The bowl descends until the tip is on that surface, then digs in until the tip is `map_tip_depth` **10 mm** under it (was 16: the floor stopped every scoop of 2026-11-05), never nearer than 3 mm to the floor, so the surface decides on any bed over 13 mm. The level is read even where the wrist-reach model calls the stroke out of reach, and without the rim band (a dish heaped over its rim lost 98% of its points to it). Floor rule when no level can be read | the user's choice; 10 mm = the 10.2-10.6 mm under the surface that the 8 scoops of 2026-11-05 dug and that worked | FROZEN (not yet on the robot) |
| `tip_floor_gap`: head's lowest point over the inside floor | 3 mm | bench | FROZEN |
| `floor_thickness` | 8 mm | measured dish | FROZEN |
| `DISH_SCOOP_Z_OFFSET` | −5 mm | bench (−11 mm pressed 6 mm into the floor, 2 of 5 jammed) | FROZEN |
| `dig_tilt_deg`, sweep 60° → 0° | 60°, 5.0 s | bench | FROZEN |
| `map_min_bed`, `min_immersion` | 3 mm, 1 mm | warnings only: a thin bed digs to the floor, and a shallow tip still sweeps | FROZEN |
| **Protocol**: powder bed in every dish | ≥ 15 mm, levelled | bench: worked on 13-21 mm beds; ~8 mm brought up nothing | FROZEN (set by hand before each trial) |

## Pick up (`bench_manifest.py`)

| Parameter | Value | Source | Status |
|---|---|---|---|
| Clear cups: `slide_in`, `grip_width`, `z_offset` | 0.06 m, 0.058 m, 0 | bench | FROZEN |
| Water cups A/B `forward_offset` | 0.010 m | bench 7 picks | FROZEN |
| Beaker `z_offset` | 0.0 | bench (17:00 run ok; +0.02 closed above the rim) | FROZEN |
| Planner may override these per call | no: on planner runs a bench value replaces the call's (`SkillsExecutor.bench_wins`) | the beaker failure on 2026-11-04 | FROZEN |

## Pour (`robochem/orchestrator/agent_orchestrator.py`, `robochem/skills/pour.py`)

| Parameter | Value | Source | Status |
|---|---|---|---|
| `POUR_KEEP_SOME_DEG`: a container that pours again (beaker into A) | 60° | the user (2026-10-08); bench: held at 49.9° on 2026-11-05 | FROZEN |
| `POUR_FIRST_INTO_SHARED_DEG`: first pour into a cup poured into again (A into C) | 90° | bench: 65° poured only a little into C (2026-11-05) | FROZEN, not yet rerun |
| `POUR_LAST_DEG`: last pour of a container | 90° | the user (2026-10-15) | FROZEN |
| A tip that falls short | warning; pours at the furthest tip reached and goes on | the user (2026-11-05) | FROZEN |

## Stir

Every `STIR_*` value in `bench_manifest.py`: bench, **FROZEN**.
