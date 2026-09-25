"""
Dump Skill

Empties a loaded scoop into a target container, pouring from JUST above the
rim with the bowl held over the opening the whole time:

1. Go home (still holding the loaded scoop) and scan the target from there,
   with the cameras clear.
2. Carry the bowl straight to just above the target's rim, cupped the way
   scoop left it, pointing straight ahead as it did at home -- no turn.
3. Tip nose-down, straight ahead, in ONE continuous motion. The bowl's centre
   stays fixed over the opening; only its height changes, so that the spoon's
   lowest point stays a set clearance above the rim at every angle.
4. Hold, shake loose what clings, tip back level, lift straight up.

How far it tips is limited by REACH, not by the wrist. Tipping a forward-
pointing bowl nose-down turns the gripper back toward the robot, so the wrist
has to sit beyond the cup and fold back over it. Over the paper cup, 0.54 m
from the base, the Panda runs out of arm at about 71 degrees: 300 random IK
seeds found no way at all to hold 80 or 90 there (2026-09-24). A cup at 0.50 m
allows ~86 degrees, at 0.45 m past 110. So the tip stops where the arm can
still hold the bowl in place (see ``_wrist_reach``), and says so, rather than
reaching for ``dump_angle_deg`` and dragging the bowl -- and the powder --
back toward the base, which is what the first version of this skill did: 80 mm
of drift and 11% of the powder outside the cup.

Turning the bowl sideways (``dump_heading_deg``) buys the full 90 degrees
anywhere on the bench, at the cost of a wrist turn on the way over; +60 was the
heading that also kept the forearm clear of the neighbouring beaker.

The bowl hangs on a cranked handle 40 mm from the TCP, so every pose here is
solved for the BOWL and the TCP put wherever that requires -- see
``BaseSkill.tcp_for_tip``. Tipping the wrist in place would swing the bowl out
on that radius and throw the powder wide.
"""

from typing import Any, Dict, Tuple

import numpy as np

from .base_skill import BaseSkill, _orthonormalize, tool_y_delta


def _level(heading: float) -> np.ndarray:
    """Tool pointing straight down with its long axis (tool X) along ``heading``."""
    x = np.array([np.cos(heading), np.sin(heading), 0.0])
    z = np.array([0.0, 0.0, -1.0])
    return np.column_stack([x, np.cross(z, x), z])


def _tilt_about(level: np.ndarray, rotation: np.ndarray) -> float:
    """Signed nose-down tilt of ``rotation`` from ``level``, about tool Y, degrees."""
    rel = level.T @ np.asarray(rotation, dtype=float)
    return float(np.degrees(np.arctan2(rel[2, 0], rel[0, 0])))


# Panda geometry for the reach check, base frame. Joint 2 (the shoulder) is
# 0.333 m up. Joint 7's axis runs through the TCP and 0.2104 m behind it (the
# 0.107 m flange plus the Franka hand's 0.1034 m), and joint 6's axis sits
# 0.088 m off joint 7's, in whichever direction joint 7 turns it.
PANDA_SHOULDER = np.array([0.0, 0.0, 0.333])
PANDA_J7_BEHIND_TCP = 0.2104
PANDA_J6_OFFSET = 0.088
#: Furthest joint 6 can get from the shoulder with the elbow as straight as
#: joint 4 allows. Every nose-down limit the sim's IK found sat right on it:
#: 71 deg at 0.54 m, 86 deg at 0.50 m, 114 deg at 0.45 m all needed
#: 0.717-0.719 m.
PANDA_REACH = 0.718


def _wrist_reach(tcp: np.ndarray, rotation: np.ndarray) -> float:
    """
    How far from the shoulder joint 6 has to be for this TCP pose, metres,
    with joint 7 turned to bring it as close as possible.
    """
    z = np.asarray(rotation, dtype=float)[:, 2]
    v = np.asarray(tcp, dtype=float) - PANDA_J7_BEHIND_TCP * z - PANDA_SHOULDER
    along = float(v @ z)
    across = float(np.linalg.norm(v - along * z))
    return float(np.hypot(along, across - PANDA_J6_OFFSET))


class DumpSkill(BaseSkill):
    """
    Tip a loaded scoop out into a target container.

    Pipeline:
    1. Record held width; the scoop must survive the whole motion
    2. Home, and locate the target from there
    3. Carry the bowl to just above the target rim, cupped, pointing ahead
    4. Tip straight ahead in one motion, bowl centre fixed over the opening,
       to dump_angle_deg or as far as the arm can reach, whichever is less
    5. Hold, shake, tip back level, lift
    """

    name = "dump"
    required_params = ["target_container"]

    #: ``target_container`` may be a name to segment, or explicit [x, y, z_rim]
    #: coordinates. Coordinates exist because identical unlabelled containers
    #: cannot be told apart by description: with four clear cups on the bench,
    #: each camera picks its own "best" one, they disagree in 3D, and consensus
    #: rejects the lot.

    @property
    def optional_params(self) -> Dict[str, Any]:
        return {
            # How far past level to tip, nose-down. Vertical is enough to pour.
            # Straight ahead, the arm's reach caps it first unless the target
            # is close to the base (~71 deg over a cup 0.54 m out); the tip
            # then stops at the cap -- see the module docstring.
            "dump_angle_deg": 90.0,
            # Below this the bowl has not tipped far enough to count as dumped.
            "min_tip_deg": 60.0,
            # Keep joint 6 this much short of PANDA_REACH when capping the tip.
            "reach_margin": 0.006,
            # Tilt the bowl is carried at, nose-UP: cupping the load, the way
            # scoop's exit leaves it (its cup_tilt_deg).
            "carry_tilt_deg": -15.0,
            # Gap between the spoon's LOWEST point and the rim while pouring,
            # at every tilt. Small on purpose: powder poured from high
            # scatters, and the rim estimate is good to a few millimetres.
            "rim_clearance": 0.010,
            # Which way the bowl points while it tips. None: straight ahead, as
            # the wrist points at home -- no turn on the way over. A number
            # turns it that many degrees counterclockwise (seen from above) of
            # the direction from the robot base to the target: 60 reaches the
            # full 90 deg anywhere on the bench and keeps the forearm clear of
            # the neighbouring beaker; at 90 the forearm hits it past ~95 deg.
            "dump_heading_deg": None,
            # --- timing (each divided by the arm's speed) -----------------
            "transfer_seconds": 5.0,
            "tip_seconds": 4.0,
            "hold_duration": 1.0,
            # Static and damp powder cling; a couple of small swings back from
            # the dump angle shift far more than a longer hold does.
            "shakes": 2,
            "shake_deg": 12.0,
            "shake_seconds": 0.6,       # per half swing
            "return_seconds": 3.0,
            # Final lift: the spoon's lowest point this far above the rim.
            "lift_height": 0.10,
            "lift_seconds": 2.5,
            "transfer_waypoints": 20,
            "tip_waypoints": 24,
            # --- finding the target ---------------------------------------
            # Go home before dumping (and scan the target from there, cameras
            # clear). False: start from wherever scoop left the arm, scanning
            # from there and homing only if the target cannot be seen.
            "home_first": True,
            "home_if_unseen": True,
            "reset_before_scan": True,   # how clear_cameras gets out of the way
            # SAM category to search when the target is identified by a
            # written label. The label path only considers instances of this
            # category, so a labelled CLEAR cup is invisible under the default
            # "white paper cup" and the query quietly finds nothing.
            "container_category": None,
            # --- the tool -------------------------------------------------
            # TCP -> bottom of the bowl, in the TOOL frame: the same value
            # scoop used. bowl_length / bowl_width / bowl_depth are the bowl's
            # outside dimensions, from that point up to the mouth; they are
            # what the clearance above the rim is measured on. Zero treats the
            # bowl as a point, which is only safe with a generous clearance.
            "tool_length": 0.0,
            "tool_offset": None,
            "bowl_length": 0.0,
            "bowl_width": 0.0,
            "bowl_depth": 0.0,
            # Site trim in the base frame, same convention as pour and scoop:
            # +X is forward from the base, so negative pulls it back.
            "forward_offset": 0.0,
            "lateral_offset": 0.0,
            "hover_tol": 0.02,
            "tip_tol_deg": 10.0,
            # Warn when the bowl ends up further than this from over the
            # target: the powder is then landing somewhere else.
            "max_bowl_drift": 0.015,
        }

    def check_preconditions(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        valid, msg = self.validate_params(params)
        if not valid:
            return False, msg

        if not self.is_gripper_holding():
            return False, ("Gripper is not holding anything. Pick up a scoop "
                           "and scoop something first.")

        angle = float(params.get("dump_angle_deg", 90.0))
        if angle < 60.0:
            print(f"[Dump] WARNING: dump_angle_deg={angle:.0f} barely tips the "
                  f"bowl; most of the powder will stay in it.")
        return True, "Preconditions met"

    # ---------------------------------------------------------------- target
    def _find_target(self, target, params, homed: bool) -> Tuple[Any, str]:
        """(rim_center, rim_z, rim_radius) and how it was found, or (None, why)."""
        if isinstance(target, (list, tuple, np.ndarray)):
            coords = np.asarray(target, dtype=float)
            if coords.shape[0] != 3:
                return None, (f"target_container coordinates need [x, y, z], "
                              f"got {coords.shape[0]} values")
            print(f"[Dump] Target given as coordinates {np.round(coords, 4)} "
                  f"-- no scan. Rim assumed at z={coords[2]:.4f}.")
            return (coords[:2], float(coords[2]), None), "coordinates"

        category = params.get("container_category")
        if homed:
            located = self.locate_container(target, force_refresh=True,
                                            category=category)
            how = "scan from home"
        else:
            # Scan from right here. The arm is over the source, not the target,
            # and the fused cloud only needs the views that agree -- so a camera
            # the arm hides is dropped by consensus rather than fatal.
            print(f"[Dump] Locating '{target}' from where the arm is...")
            located = self.locate_container(target, force_refresh=True,
                                            category=category)
            how = "scan in place"
            if located is None and bool(params["home_if_unseen"]):
                print(f"[Dump] '{target}' not found from here -- homing to scan "
                      f"with the cameras clear")
                if not self.clear_cameras(params, tag="Dump"):
                    return None, "Failed to clear the cameras before scanning"
                located = self.locate_container(target, force_refresh=True,
                                                category=category)
                how = "scan from home (not visible in place)"
        if located is None:
            return None, f"Cannot locate '{target}'"
        return (located["rim_center"], float(located["top_z"]),
                float(located["rim_radius"])), how

    # --------------------------------------------------------------- execute
    def execute(self, params: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        params = self.get_params_with_defaults(params)
        target = params["target_container"]
        dump_angle = float(params["dump_angle_deg"])
        carry = float(params["carry_tilt_deg"])
        clearance = float(params["rim_clearance"])
        tip_tol = float(params["tip_tol_deg"])
        min_tip = float(params["min_tip_deg"])
        reach_limit = PANDA_REACH - float(params["reach_margin"])
        hover_tol = float(params["hover_tol"])
        tool_offset = self.resolve_tool_offset(params)

        print(f"[Dump] Emptying the scoop into '{target}'")
        start_width = self.held_width()
        print(f"[Dump] Holding scoop at {start_width * 1000:.1f}mm")

        # 1. Home first, still holding the loaded scoop, so every dump starts
        # from the same pose -- and the scan has the cameras clear.
        homed = False
        if bool(params["home_first"]):
            print("[Dump] Back to home first, still holding the scoop...")
            if not self.go_home():
                return False, {"error": "Failed to go home before dumping"}
            self.wait(0.4)
            homed = True
            ok, msg = self.check_still_holding(start_width, tag="Dump")
            if not ok:
                return False, {"error": f"Lost the scoop while homing: {msg}"}
        found, how = self._find_target(target, params, homed)
        if found is None:
            return False, {"error": how}
        rim_center, rim_z, rim_radius = found
        ok, msg = self.check_still_holding(start_width, tag="Dump")
        if not ok:
            return False, {"error": f"Lost the scoop before dumping: {msg}"}
        site = np.asarray(rim_center, dtype=float) + np.array([
            float(params["forward_offset"]), float(params["lateral_offset"])])
        print(f"[Dump] '{target}' rim centre {np.round(rim_center, 4)}, "
              f"z={rim_z:.4f}"
              + (f", radius {rim_radius * 1000:.0f}mm" if rim_radius else "")
              + f" ({how})")
        if not np.allclose(site, rim_center):
            print(f"[Dump] Dump site shifted to {np.round(site, 4)}")

        # 2. The bowl, in the tool frame: its bottom is the tool offset, its
        # mouth faces tool -Z, its long axis is tool X.
        length = max(0.0, float(params["bowl_length"]))
        width = max(0.0, float(params["bowl_width"]))
        depth = max(0.0, float(params["bowl_depth"]))
        bottom = np.asarray(tool_offset, dtype=float)
        centre_tool = bottom - np.array([0.0, 0.0, depth / 2])
        corners = np.array([
            [bottom[0] + sx * length / 2, bottom[1] + sy * width / 2, bottom[2] - dz]
            for sx in (-1, 1) for sy in (-1, 1) for dz in (0.0, depth)])

        # The heading to tip at: straight ahead as the wrist points now, or a
        # set turn from the radial, reached the shorter way round.
        now = self.get_current_pose()
        tcp_now = np.asarray(now.translation, dtype=float)
        R_now = _orthonormalize(np.asarray(now.rotation, dtype=float))
        heading_now = float(np.arctan2(R_now[1, 0], R_now[0, 0]))
        tilt_now = _tilt_about(_level(heading_now), R_now)
        heading_rel = params.get("dump_heading_deg")
        if heading_rel is None:
            turn = 0.0
        else:
            radial = float(np.arctan2(site[1], site[0]))
            want = radial + np.radians(float(heading_rel))
            turn = float((want - heading_now + np.pi) % (2 * np.pi) - np.pi)
        heading = heading_now + turn
        level = _level(heading)

        def rotation_at(tilt: float, psi: float = heading) -> np.ndarray:
            return _orthonormalize(_level(psi) @ tool_y_delta(-float(tilt)))

        def pose_at(tilt: float, above_rim: float = clearance):
            """TCP and rotation putting the bowl centre over the site, its
            lowest point ``above_rim`` over the rim."""
            R = rotation_at(tilt)
            below = float(min((R @ c)[2] for c in corners) - (R @ centre_tool)[2])
            centre = np.array([site[0], site[1], rim_z + above_rim - below])
            return centre - R @ centre_tool, R

        def bowl_now():
            pose = self.get_current_pose()
            R = np.asarray(pose.rotation, dtype=float)
            return np.asarray(pose.translation, dtype=float) + R @ centre_tool, R

        drift = {}

        def record(stage: str) -> float:
            centre, _ = bowl_now()
            d = float(np.linalg.norm(centre[:2] - site))
            drift[stage] = d
            return d

        def run_path(tilts, psis, centres, seconds, stage):
            """One continuous motion through bowl poses; blocking steps if the
            arm cannot stream."""
            rots = [rotation_at(t, p) for t, p in zip(tilts, psis)]
            tcps = [c - R @ centre_tool for c, R in zip(centres, rots)]
            streamed, why = self.stream_pose_path(tcps, rots, seconds=seconds,
                                                  tag="Dump")
            if streamed:
                return True
            print(f"[Dump]   continuous {stage} unavailable ({why}); "
                  f"stepping through it instead")
            stride = max(1, len(tcps) // 6)
            picks = list(range(stride - 1, len(tcps), stride))
            if picks[-1] != len(tcps) - 1:
                picks.append(len(tcps) - 1)
            for i in picks:
                if not self.goto_pose_rigid(tcps[i], rots[i],
                                            duration=seconds / len(picks)):
                    return False
            return True

        def centres_for(tilts):
            return [pose_at(t)[0] + rotation_at(t) @ centre_tool for t in tilts]

        # How far the arm can tip here while still holding the bowl in place.
        # Past this the wrist cannot stay behind the bowl, and the bowl -- and
        # the powder -- gets dragged back toward the base instead.
        tip_to = carry
        for a in np.arange(carry, dump_angle + 1e-6, 1.0):
            if _wrist_reach(*pose_at(float(a))) > reach_limit:
                break
            tip_to = float(a)
        if tip_to < dump_angle - 0.5:
            reach_r = float(np.linalg.norm(site))
            print(f"[Dump] NOTE: pointing this way the arm can only tip to "
                  f"{tip_to:.0f}° with the bowl held over '{target}' "
                  f"({reach_r:.2f} m from the base); stopping there instead of "
                  f"{dump_angle:.0f}°. Closer to the base, or a sideways "
                  f"dump_heading_deg, reaches further.")
        if tip_to < min_tip:
            return False, {
                "error": (f"'{target}' is out of reach for a dump: the arm can "
                          f"tip only {tip_to:.0f}° over it pointing this way, "
                          f"short of min_tip_deg={min_tip:.0f}°. Move it closer "
                          f"to the base, or pass dump_heading_deg (60 reaches "
                          f"90° across the bench)."),
                "tip_reachable": tip_to,
            }

        # 3. Carry the bowl straight to just above the rim, cupped, still
        # pointing ahead (or turning to dump_heading_deg on the way).
        start_tcp, start_R = pose_at(carry)
        start_centre = start_tcp + start_R @ centre_tool
        centre_now = tcp_now + R_now @ centre_tool
        n = max(2, int(params["transfer_waypoints"]))
        fracs = np.arange(1, n + 1) / n
        rise = start_centre[2] - centre_now[2]
        centres = []
        for f in fracs:
            c = centre_now + (start_centre - centre_now) * f
            if rise > 0:
                # Coming UP to this rim: climb in the first half, so the bowl
                # is already above it before it gets there.
                c[2] = centre_now[2] + rise * min(1.0, 2.0 * f)
            centres.append(c)
        print(f"[Dump] Carrying the bowl {np.linalg.norm(start_centre[:2] - centre_now[:2]) * 1000:.0f}mm "
              f"to over the rim, "
              + ("pointing straight ahead" if heading_rel is None else
                 f"turning the wrist {np.degrees(turn):+.0f}° to a "
                 f"{float(heading_rel):+.0f}° heading")
              + f", lowest point {clearance * 1000:.0f}mm above the rim...")
        if not run_path(tilt_now + (carry - tilt_now) * fracs,
                        heading_now + turn * fracs, centres,
                        float(params["transfer_seconds"]), "carry"):
            return False, {"error": "Failed to carry the scoop to the target"}
        arrived, err = self.reached(start_tcp, hover_tol)
        if not arrived:
            return False, {
                "error": (f"Could not bring the bowl over '{target}' "
                          f"(off by {err * 1000:.0f}mm)"),
            }
        record("arrived")

        # 4. Tip, in one motion, the bowl centre fixed over the opening.
        tilts = np.linspace(carry, tip_to, max(2, int(params["tip_waypoints"])) + 1)[1:]
        print(f"[Dump] Tipping {carry:+.0f}° -> {tip_to:+.0f}° over "
              f"{float(params['tip_seconds']):.1f}s, bowl held over the rim...")
        if not run_path(tilts, [heading] * len(tilts), centres_for(tilts),
                        float(params["tip_seconds"]), "tip"):
            return False, {"error": "Failed to command the tip"}
        achieved = _tilt_about(level, bowl_now()[1])
        if achieved < tip_to - tip_tol:
            print(f"[Dump]   tipped {achieved:.1f}° of {tip_to:.0f}°; "
                  f"commanding the last pose again")
            final_tcp, final_R = pose_at(tip_to)
            self.goto_pose_rigid(final_tcp, final_R, duration=1.5)
            achieved = _tilt_about(level, bowl_now()[1])
        tipped_drift = record("tipped")
        print(f"[Dump]   measured tip {achieved:.1f}°, bowl "
              f"{tipped_drift * 1000:.1f}mm from over the target")
        if achieved < min_tip:
            print(f"[Dump] Tip stalled at {achieved:.1f}° -- returning to carry")
            back_tcp, back_R = pose_at(carry)
            self.goto_pose_rigid(back_tcp, back_R, duration=3.0)
            return False, {
                "error": (f"Tip stalled at {achieved:.1f}° of {tip_to:.0f}°, "
                          f"short of min_tip_deg={min_tip:.0f}° -- the bowl never "
                          f"got far enough over for the powder to come out"),
                "tip_achieved": achieved,
                "tip_commanded": tip_to,
            }

        # 5. Hold, then shake -- swinging back from the dump angle only, so the
        # wrist never goes further than it just proved it can.
        hold = float(params["hold_duration"])
        print(f"[Dump] Holding {hold:.1f}s at {achieved:.1f}°...")
        self.wait(hold)
        top = min(tip_to, achieved)
        shakes = max(0, int(params["shakes"]))
        shake_deg = abs(float(params["shake_deg"]))
        if shakes and shake_deg > 0.5:
            print(f"[Dump] Shaking {shakes}x {top - shake_deg:.0f}°<->{top:.0f}° "
                  f"to dislodge...")
            # Each half swing is its own motion, starting and ending at rest.
            # Strung into one path, the swings reverse at speed and the servos
            # overshoot: the bowl lurched 9 mm off target and to within 3.5 mm
            # of the rim, with the arm near full reach (sim, 2026-09-24).
            swing_s = float(params["shake_seconds"])
            for _ in range(shakes):
                for a, b in ((top, top - shake_deg), (top - shake_deg, top)):
                    seg = list(np.linspace(a, b, 5)[1:])
                    if not run_path(seg, [heading] * len(seg), centres_for(seg),
                                    swing_s, "shake"):
                        print("[Dump]   shake failed; carrying on")
            record("shaken")

        # 6. Back to level over the rim, and CHECK: an unverified return once
        # left the scoop hanging inverted through the lift (residual 111 deg on
        # the 2026-09-21 bench run), handing the next skill a tool pointing the
        # wrong way.
        print("[Dump] Returning level...")
        back = np.linspace(top, 0.0, 13)[1:]
        run_path(back, [heading] * len(back), centres_for(back),
                 float(params["return_seconds"]), "return")
        residual = _tilt_about(level, bowl_now()[1])
        if abs(residual) > tip_tol:
            level_tcp, level_R = pose_at(0.0)
            self.goto_pose_rigid(level_tcp, level_R, duration=2.0)
            residual = _tilt_about(level, bowl_now()[1])
        print(f"[Dump]   residual tilt {residual:.1f}°")
        if abs(residual) > tip_tol:
            print(f"[Dump] WARNING: still {residual:.1f}° from level. The scoop "
                  f"is lifting away tipped -- re-home before the next skill.")
        record("level")

        # 7. Straight up, clear of the rim.
        lift_tcp, lift_R = pose_at(0.0, above_rim=float(params["lift_height"]))
        print("[Dump] Lifting clear...")
        if not self.goto_pose_rigid(lift_tcp, lift_R,
                                    duration=float(params["lift_seconds"])):
            return False, {"error": "Dumped, but failed to lift clear",
                           "tip_achieved": achieved}

        ok, msg = self.check_still_holding(start_width, tag="Dump")
        if not ok:
            return False, {"error": f"Scoop lost during the dump: {msg}"}

        worst = max(drift.values()) if drift else 0.0
        if worst > float(params["max_bowl_drift"]):
            print(f"[Dump] WARNING: the bowl was {worst * 1000:.0f}mm from over "
                  f"the target at one point ({max(drift, key=drift.get)}); "
                  f"powder may have landed outside it.")

        # The target's contents changed, so the cached cloud is stale.
        self.vision.clear_cache()

        print(f"[Dump] Emptied the scoop into '{target}' (tipped {achieved:.1f}°)")
        return True, {
            "dumped_into": (target if isinstance(target, str) else "coordinates"),
            "target_found_by": how,
            "rim_center": [float(v) for v in rim_center],
            "rim_z": rim_z,
            "rim_radius": rim_radius,
            "heading_deg": float(np.degrees(heading)),
            "dump_heading_deg": heading_rel,
            "wrist_turn_deg": float(np.degrees(turn)),
            "tip_requested": dump_angle,
            "tip_commanded": tip_to,
            "tip_capped_by_reach": tip_to < dump_angle - 0.5,
            "tip_achieved": achieved,
            "rim_clearance": clearance,
            "bowl_drift_mm": {k: round(v * 1000, 1) for k, v in drift.items()},
            "residual_tilt": residual,
            "shakes": shakes,
            "tool_offset": [float(v) for v in tool_offset],
            "bowl_size": [length, width, depth],
            "forward_offset": float(params["forward_offset"]),
            "lateral_offset": float(params["lateral_offset"]),
            # Nothing weighs what left the bowl, or what stayed stuck in it.
            "quantity_measured": False,
        }
