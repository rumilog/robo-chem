"""
Before the arm starts: can every step reach its target? Move what it cannot.

A target that is out of reach used to be found out halfway through the run:
on 2026-10-22 the dump into the 20 ml cup, 0.66 m from the base, stopped at
step 3 with the scoop loaded, because straight ahead the arm can tip the bowl
only 39 deg there. This runs first, with the gripper still empty:

1. Every target the steps name is scanned from home. The scans are kept
   (``VisionSystem.scans``), so the skills that use them later reuse them.
2. Each is checked the way its step will use it:
   - dump: ``reachable_tip_deg``, the dump skill's own reach check without
     the robot. It gives 39.0 deg for the 2026-10-22 cup, which is what the
     dump measured;
   - pour / stir / dispense: distance from the base, ``TARGET_REACH``;
   - scoop: the dish cannot be moved (wider than the jaws open), so a far
     one is only reported.
3. A container that fails is moved, if the jaws can take it: ``pick_up``,
   then ``place`` at the nearest free spot that passes the same check, kept
   clear of everything the cameras see and of the dishes being scooped. It is
   not put back afterwards (the user's choice, 2026-10-22).

Dumps themselves are not changed: they still dump straight ahead, as they
always have. Moving the cup is what fixes the reach.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from robochem.skills.dump import reachable_tip_deg

#: A dump that can tip less than this is fixed by moving its target. The dump
#: skill itself stops below its min_tip_deg (45 on the bench); a bowl tipped
#: only 45-60 deg empties poorly.
DUMP_TIP_TOO_LOW = 60.0
#: Below this the dump skill itself refuses (the bench's min_tip_deg).
DUMP_MIN_TIP = 45.0
#: What a moved dump target must allow at its new spot (79 is the default
#: dump angle; 0.50 m from the base gives 78).
DUMP_TIP_WANTED = 75.0
#: Distance from the base (m) outside which a pour, stir or dispense target is
#: moved. A guess, not measured: stirs and pours have worked out to ~0.6 m.
TARGET_REACH = (0.30, 0.60)
#: Where a moved container may stand: the sim's LAYOUT_REACH (0.35-0.56),
#: its outer edge pulled in so a dump there reaches ~78 deg.
SPOT_REACH = (0.35, 0.50)
SPOT_MIN_X = 0.30
#: Free space around a moved container beyond its own radius, for the jaws.
FINGER_ROOM = 0.06
#: Farthest out (x, m) a dish can be scooped from. From the logs: 12 scoops
#: worked with the dish at x 0.500-0.602; both jams were further out,
#: (0.604, 0.183) on 2026-10-21 and (0.642, -0.028) on 2026-10-22, each
#: at +55 deg with joint_velocity_discontinuity.
SCOOP_MAX_X = 0.60
#: Where a moved DUMP target may stand: the box every real dump that tipped
#: fully stood in. 13 dumps there reached 76 deg (2026-10-15..22, x 0.388-
#: 0.458, |y| <= 0.138). Both side targets failed on the carry: (0.305,
#: -0.207) on 2026-10-07 with joint_position_limits_violation, and (0.428,
#: +0.233) on 2026-10-22, the spot this preflight chose before the box,
#: 28 mm short and 25 mm high. The wrist-reach model does not see joint
#: limits, so it alone is not enough.
DUMP_SPOT_X = (0.39, 0.46)
DUMP_SPOT_MAX_ABS_Y = 0.14
#: Tall containers keep this far from a dish being scooped: the scooping hand
#: hit them at 144-161 mm in sim and cleared 172 mm (LAYOUT_SCOOP_CLEARANCE
#: rounds that to 0.18).
DISH_CLEARANCE = 0.17
#: A container farther than this is not fetched: the pick hovers 60 mm
#: behind it and the safety clamp stops the hand at x 0.75.
MAX_FETCH_REACH = 0.72
#: Containers wider than this (radius, m) are not picked up to be moved.
MAX_MOVABLE_RADIUS = 0.040
#: The scoop's tool offset for planning: measured at the pick 40.6-47.3 mm
#: out on the bench (2026-10-21/22); z from CAD.
PLAN_TOOL_OFFSET = (0.045, 0.0, 0.0294)
GRID = 0.01

#: Words in a sub-task's description, and the step kind they mean.
_KINDS = (("dump", "dump"), ("tip the", "dump"), ("pour", "pour"),
          ("stir", "stir"), ("dispense", "dispense"), ("drop", "dispense"),
          ("scoop", "scoop"))
_TARGET_PARAM = {"dump": "target_container", "pour": "target_container",
                 "stir": "target_container", "dispense": "target_container",
                 "scoop": "powder_source"}


class Preflight:
    """Checks a run's targets before anything moves, and moves the ones out of reach."""

    def __init__(self, vision, skills, say=print):
        self.vision = vision
        self.skills = skills
        self.say = say

    # -------------------------------------------------------------- inputs
    @staticmethod
    def items_from_calls(calls) -> List[Tuple[str, str]]:
        """(kind, target) for each ``(skill, params)`` of a --skill chain."""
        out = []
        for skill, params in calls:
            key = _TARGET_PARAM.get(skill)
            value = (params or {}).get(key) if key else None
            if isinstance(value, str) and value.strip():
                out.append((skill, value))
        return out

    @staticmethod
    def items_from_plan(plan) -> List[Tuple[str, str]]:
        """(kind, target) for each sub-task of a planner's plan."""
        out = []
        for sub in plan.subtasks:
            target = str(getattr(sub, "target_container", "") or "").strip()
            if not target or target.lower() == "none":
                continue
            text = sub.description.lower()
            kind = next((k for word, k in _KINDS if word in text), None)
            if kind is not None:
                out.append((kind, target))
        return out

    # ----------------------------------------------------------------- run
    def run(self, items) -> dict:
        """Check and fix. ``blocked`` is set only when a move itself failed."""
        started = time.time()
        report = {"checked": [], "moved": [], "warnings": [], "blocked": None}
        targets: Dict[str, set] = {}
        for kind, name in items:
            name = self.vision.canonical_name(name)
            targets.setdefault(name, set()).add(kind)
        if not targets:
            return report
        self.say(f"\n[preflight] checking {len(targets)} target(s) before the arm "
                 f"starts: {', '.join(sorted(targets))}")
        if not self.skills.go_home():
            # 2026-10-22: refused (virtual walls), and the chain went on to a
            # pick_up that was refused the same way.
            report["blocked"] = "the arm could not go home: check franka-interface"
            self.say(f"[preflight] {report['blocked']}")
            return report

        found = {}
        for name in targets:
            found[name] = self._locate(name)
            frames = (getattr(self.vision.object_localizer, "last_capture", None)
                      or {}).get("images")
            if found[name] is None and not frames:
                # 2026-10-22 16:4x: all four cameras "Frame didn't arrive
                # within 5000", dmesg uvcvideo -71 -- the shared USB
                # controller had wedged. Nothing after this can see either.
                report["blocked"] = ("no camera delivered a frame (USB wedged? dmesg "
                                     "shows uvcvideo -71): replug the camera hub")
                self.say(f"[preflight] {report['blocked']}")
                return report
        dishes = [found[n] for n, kinds in targets.items()
                  if "scoop" in kinds and found.get(n)]

        for name, kinds in targets.items():
            loc = found.get(name)
            if loc is None:
                # Its step would fail the same way, after the arm had already
                # scooped or picked things up. Stop while nothing has moved.
                entry = self.vision.bench.find(name) if getattr(self.vision, "bench", None) else None
                hint = (f" -- is the {entry.label.upper()!r} label under it and in view?"
                        if entry is not None and entry.label else "")
                report["blocked"] = f"{name} not found on the bench{hint}"
                self.say(f"[preflight] {report['blocked']}")
                break
            xy = np.asarray(loc["rim_center"], float)[:2]
            r = float(np.linalg.norm(xy))
            problems, row = self._problems(name, kinds, loc, xy, r)
            report["checked"].append(row)
            if not problems:
                self.say(f"[preflight] {name} at ({xy[0]:.3f}, {xy[1]:+.3f}), "
                         f"{r:.2f} m: ok" + (f", dump {row['dump_tip']:.0f}°"
                                             if "dump_tip" in row else ""))
                continue
            why = "; ".join(problems)
            self.say(f"[preflight] {name} at ({xy[0]:.3f}, {xy[1]:+.3f}): {why}")
            movable, reason = self._movable(name)
            if "scoop" in kinds and not movable:
                # The scoop would jam after the pick_up; stop while nothing has moved.
                report["blocked"] = f"{name}: {why}; move it closer to the robot by hand"
                self.say(f"[preflight] {report['blocked']}")
                break
            if movable and r > MAX_FETCH_REACH:
                movable, reason = False, (f"{r:.2f} m out, beyond where the arm can pick "
                                          f"it up; move it closer by hand")
            if not movable:
                report["warnings"].append(f"{name}: {why}; not moved: {reason}")
                self.say(f"[preflight]   not moving it: {reason}")
                continue
            spot = self._find_spot(name, loc, need_dump="dump" in kinds, dishes=dishes,
                                   others=[found[n] for n in targets if n != name and found[n]])
            if spot is None:
                if "dump" in kinds and row.get("dump_tip", 90.0) < DUMP_MIN_TIP:
                    # The dump would refuse it after the scoop; say so now.
                    report["blocked"] = (
                        f"{name} is out of reach for the dump ({why}) and no free spot "
                        f"in the dump box x {DUMP_SPOT_X[0]:.2f}-{DUMP_SPOT_X[1]:.2f}, "
                        f"|y| <= {DUMP_SPOT_MAX_ABS_Y:.2f} is clear of everything else: "
                        f"make room there by hand")
                    self.say(f"[preflight] {report['blocked']}")
                    break
                report["warnings"].append(f"{name}: {why}; no free spot in reach")
                self.say(f"[preflight]   no free spot in reach found; leaving it")
                continue
            ok, detail = self._move(name, spot)
            report["moved"].append({"object": name, "from": [round(float(v), 4) for v in xy],
                                    "to": [round(float(v), 4) for v in spot],
                                    "why": why, "ok": ok, "detail": detail})
            if not ok:
                report["blocked"] = f"moving {name} out of the way failed: {detail}"
                # A pick that gave up leaves the hand wherever it stopped.
                self.skills.go_home()
                break
        report["seconds"] = round(time.time() - started, 1)
        self.say(f"[preflight] done in {report['seconds']}s: "
                 f"{len(report['moved'])} moved, {len(report['warnings'])} warning(s)")
        return report

    # ------------------------------------------------------------- checks
    def _locate(self, name):
        try:
            return self.skills.skill_instance("dump").locate_container(name, force_refresh=True)
        except Exception as exc:
            self.say(f"[preflight] {name}: scan failed ({type(exc).__name__}: {exc})")
            return None

    def _problems(self, name, kinds, loc, xy, r):
        problems = []
        row = {"object": name, "kinds": sorted(kinds),
               "xy": [round(float(v), 4) for v in xy], "r": round(r, 3)}
        if "dump" in kinds:
            tip = reachable_tip_deg(xy, float(loc["top_z"]), PLAN_TOOL_OFFSET)
            row["dump_tip"] = tip
            if tip < DUMP_TIP_TOO_LOW:
                problems.append(f"a dump can tip only {tip:.0f}° there "
                                f"(wants >= {DUMP_TIP_TOO_LOW:.0f})")
        if kinds & {"pour", "stir", "dispense"} and not (TARGET_REACH[0] <= r <= TARGET_REACH[1]):
            problems.append(f"{r:.2f} m from the base, outside "
                            f"{TARGET_REACH[0]:.2f}-{TARGET_REACH[1]:.2f}")
        if "scoop" in kinds and xy[0] > SCOOP_MAX_X:
            problems.append(f"x {xy[0]:.3f} m: past {SCOOP_MAX_X:.2f}, where the scoop jams")
        return problems, row

    def _movable(self, name) -> Tuple[bool, str]:
        if getattr(self.skills, "held", None):
            return False, f"the gripper already holds {self.skills.held!r}"
        entry = self.vision.bench.find(name) if getattr(self.vision, "bench", None) else None
        if entry is None:
            return False, "not on the bench manifest, so its size is unknown"
        category = (entry.category or "").lower()
        if "cup" not in category or entry.radius is None:
            return False, f"a {entry.category or 'fixed object'}, not a cup the jaws can take"
        if entry.radius > MAX_MOVABLE_RADIUS:
            return False, f"{entry.radius * 2000:.0f} mm across, too wide for the jaws"
        return True, ""

    # --------------------------------------------------------- free spot
    def _obstacles(self, here, radius):
        """World xy of occupied 1 cm cells, from the last full capture's depth."""
        loc = self.vision.object_localizer
        data = getattr(loc, "last_capture", None) or {}
        table = float(getattr(self.vision, "table_z", 0.0))
        cells = {}
        for cam, depth in (data.get("depth_images") or {}).items():
            mask = np.zeros(depth.shape[:2], dtype=bool)
            mask[::3, ::3] = True
            try:
                pts = loc.get_object_points_by_camera(
                    {cam: mask}, depth_images={cam: depth},
                    intrinsics=data.get("intrinsics")).get(cam)
            except Exception:
                continue
            if pts is None or len(pts) == 0:
                continue
            pts = np.asarray(pts, float)
            keep = ((pts[:, 2] > table + 0.008) & (pts[:, 2] < 0.35)
                    & (pts[:, 0] > 0.15) & (pts[:, 0] < 0.95) & (np.abs(pts[:, 1]) < 0.55)
                    & (np.linalg.norm(pts[:, :2], axis=1) > 0.25)
                    & (np.linalg.norm(pts[:, :2] - here, axis=1) > radius + 0.02))
            for c in map(tuple, np.floor(pts[keep, :2] / GRID).astype(int)):
                cells[c] = cells.get(c, 0) + 1
        # A few stray points are depth noise; a real object leaves many.
        return np.array([[(i + 0.5) * GRID, (j + 0.5) * GRID]
                         for (i, j), n in cells.items() if n >= 4]).reshape(-1, 2)

    def _find_spot(self, name, loc, need_dump, dishes, others):
        here = np.asarray(loc["rim_center"], float)[:2]
        entry = self.vision.bench.find(name)
        radius = float(entry.radius)
        occupied = self._obstacles(here, radius)
        containers = [np.asarray(p, float) for ps in
                      (getattr(self.vision, "seen_containers", {}) or {}).values() for p in ps]
        containers += [np.asarray(o["rim_center"], float)[:2] for o in others]
        containers = [c for c in containers if np.linalg.norm(c - here) > 0.03]
        dish_xy = [np.asarray(d["rim_center"], float)[:2] for d in dishes]
        top_z = float(loc["top_z"])

        best, best_d = None, None
        for x in np.arange(SPOT_MIN_X, SPOT_REACH[1] + 1e-9, GRID):
            for y in np.arange(-SPOT_REACH[1], SPOT_REACH[1] + 1e-9, GRID):
                p = np.array([x, y])
                r = float(np.linalg.norm(p))
                if not (SPOT_REACH[0] <= r <= SPOT_REACH[1]):
                    continue
                d = float(np.linalg.norm(p - here))
                if best_d is not None and d >= best_d:
                    continue
                if len(occupied) and np.min(np.linalg.norm(occupied - p, axis=1)) < radius + FINGER_ROOM:
                    continue
                if any(np.linalg.norm(c - p) < 2 * radius + FINGER_ROOM for c in containers):
                    continue
                if any(np.linalg.norm(c - p) < DISH_CLEARANCE for c in dish_xy):
                    continue
                if need_dump and not (DUMP_SPOT_X[0] <= x <= DUMP_SPOT_X[1]
                                      and abs(y) <= DUMP_SPOT_MAX_ABS_Y):
                    continue
                if need_dump and reachable_tip_deg(p, top_z, PLAN_TOOL_OFFSET) < DUMP_TIP_WANTED:
                    continue
                best, best_d = p, d
        if best is not None:
            self.say(f"[preflight]   free spot ({best[0]:.3f}, {best[1]:+.3f}), "
                     f"{np.linalg.norm(best):.2f} m from the base, {best_d * 100:.0f} cm "
                     f"from where it stands ({len(occupied)} occupied cells, "
                     f"{len(containers)} other containers avoided)")
        return best

    # --------------------------------------------------------------- move
    def _move(self, name, spot) -> Tuple[bool, str]:
        self.say(f"[preflight]   moving {name!r} there: pick_up, then place")
        ok, result = self.skills.execute("pick_up", {"object_name": name})
        if not ok:
            return False, f"pick_up: {(result or {}).get('error', 'failed')}"
        key = str(name).strip().lower()
        site = self.skills.pick_sites.get(key)
        if site is None:
            return False, "pick_up did not record where it took the cup from"
        shift = np.array([spot[0] - site[0], spot[1] - site[1]])
        # place puts a picked object back at its pick site (and a cup that was
        # slid in, at the TCP where it was held). Move both to the new spot.
        self.skills.pick_sites[key] = [float(spot[0]), float(spot[1]), float(site[2])]
        grasp = self.skills.pick_grasps.get(key)
        if grasp is not None:
            self.skills.pick_grasps[key] = [float(grasp[0] + shift[0]),
                                            float(grasp[1] + shift[1]), float(grasp[2])]
        ok, result = self.skills.execute("place", {"target_location": name})
        if not ok:
            return False, f"place: {(result or {}).get('error', 'failed')}"
        remember = getattr(self.vision, "remember_position", None)
        if callable(remember):
            remember(name, spot)
        return True, f"moved {np.linalg.norm(shift) * 100:.0f} cm"
