"""
Pour Skill

After an upright top-down grasp: tip the beaker *toward the robot base*
(−X) until ~90° from vertical. Each tip step also nudges the EE forward
(+X) so the mouth stays over the cup. After the hold, ``reset_joints``.
"""

from typing import Dict, Any, Tuple
import numpy as np

from .base_skill import (
    BaseSkill,
    _orthonormalize,
    to_rigid_transform,
)

#: The Franka hand: flange to TCP along the tool's z axis, metres.
FLANGE_BEHIND_TCP = 0.1034


class PourSkill(BaseSkill):
    name = "pour"
    required_params = ["target_container"]

    @property
    def optional_params(self) -> Dict[str, Any]:
        return {
            "pour_angle": 90.0,
            "hold_duration": 2.0,
            "approach_height": 0.15,
            "horizontal_offset": 0.05,
            # +X = forward from base toward workspace. Negative = closer to base.
            # Validated working value: -0.08.
            "forward_offset": -0.08,
            "lateral_offset": 0.0,
            "reset_before_scan": True,
            "step_deg": 10.0,
            "duration_per_step": 3.0,
            # As the beaker tips toward the base, the mouth drifts toward the
            # robot. Push the EE forward (+X) by this much each tip step so
            # the stream stays over the cup (0.0025 m = 0.25 cm).
            "tip_advance_m": 0.0025,
            "tip_only": False,
            # Always tip toward robot base (−X).
            "tip_dir": "toward_base",
            # How far measured tip may lag a commanded step before we retry /
            # eventually fail. Kept loose so a ~0.1° miss does not abort.
            "tip_tol_deg": 25.0,
            # Retries per tip step when the wrist lags (frankapy often needs
            # a second longer goto_pose to track orientation).
            "step_retries": 3,
            # After hold: joint-space home (same as pre-scan).
            "reset_after_pour": True,
            # --- keeping the held cup's lip just over the target -----------
            # The held cup's shape relative to the jaws: its rim's height above
            # the TCP, its height and radius. SkillsExecutor fills these from
            # the pick that took it; with them pour plans every tip step so
            # the lip -- the rim's lowest point, where the liquid leaves --
            # stays lip_clearance over the target's rim and lip_inset of the
            # target's radius toward the base from its centre, and no part of
            # the cup comes within body_clearance of the target's rim height.
            # Without them it uses the fixed site above (forward_offset,
            # approach_height), which put the lip 155-185 mm over the target
            # (sim, 2026-10-03).
            "source_top_above_tcp": None,
            "source_height": None,
            "source_radius": None,
            "lip_clearance": 0.015,
            "body_clearance": 0.010,
            "lip_inset": 0.4,
            # Lowest the flange may go, base-frame z. franka-interface has a
            # virtual floor of its own (a plane at z = -0.015 that the flange,
            # its "frame 7", must stay ~0.1 m from) and aborts any motion that
            # crosses it -- then refuses every motion from there. On 2026-10-08
            # the 90 deg tip over a 47 mm cup put the flange 82.5 mm up: aborted,
            # and the arm had to be guided out by hand. The 80 deg pour (flange
            # 105 mm) ran. Late tip steps are raised to keep the flange here.
            "flange_min_z": 0.11,
        }

    def check_preconditions(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        valid, msg = self.validate_params(params)
        if not valid:
            return False, msg
        if not self.is_gripper_holding():
            return False, "Gripper is not holding anything. Pick up a container first."
        return True, "Preconditions met"

    def execute(self, params: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        params = self.get_params_with_defaults(params)
        target = params["target_container"]
        pour_tip = abs(float(params["pour_angle"]))
        hold_duration = float(params["hold_duration"])
        step_deg = max(5.0, abs(float(params["step_deg"])))
        step_dur = float(params["duration_per_step"])
        tip_only = bool(params["tip_only"])
        tip_tol = float(params["tip_tol_deg"])
        step_retries = max(1, int(params["step_retries"]))
        tip_advance = float(params["tip_advance_m"])
        reset_after = bool(params["reset_after_pour"])

        lean_xy = np.array([-1.0, 0.0])  # toward base
        if not tip_only and all(params.get(k) is not None for k in
                                ("source_top_above_tcp", "source_height", "source_radius")):
            return self._pour_over_lip(params, target, pour_tip, hold_duration,
                                       step_deg, step_dur, tip_tol, step_retries,
                                       reset_after, lean_xy)
        print(f"[Pour] Tip toward base (−X) to {pour_tip:.0f}° from vertical")
        print(f"[Pour] Advance +X by {tip_advance * 100:.2f} cm each "
              f"{step_deg:.0f}° tip step")

        R0 = _orthonormalize(np.asarray(self.get_current_pose().rotation, dtype=float).copy())
        fwd = float(params["forward_offset"])
        lat = float(params["lateral_offset"])
        h_offset = float(params["horizontal_offset"])

        if not tip_only:
            if params.get("reset_before_scan", True):
                print("[Pour] reset_joints (home, joint-space) clear of cameras "
                      "before scan...")
                if not self.go_home():
                    return False, {"error": "Failed to reset_joints before scanning"}
            self.wait(0.3)

            self.vision.clear_cache()
            target_pos = self.get_object_centroid(target)
            if target_pos is None:
                return False, {"error": f"Cannot locate '{target}'"}

            target_dims = self.get_object_dimensions(target)
            target_height = float(target_dims[2]) if target_dims is not None else 0.05
            approach_height = float(params["approach_height"])

            # lean −X → subtract lean*h means +X shift from lean term; then
            # forward_offset (usually negative) pulls back toward base.
            pour_xyz = target_pos.copy()
            pour_xyz[0] -= lean_xy[0] * h_offset
            pour_xyz[1] -= lean_xy[1] * h_offset
            pour_xyz[0] += fwd
            pour_xyz[1] += lat
            pour_xyz[2] = target_pos[2] + target_height + approach_height
            pour_xyz = self._clamp_position(pour_xyz)
            print(f"[Pour] Site offsets: forward(X)={fwd:+.3f}m  "
                  f"lateral(Y)={lat:+.3f}m  lean_shift={h_offset:.3f}m")
            print(f"[Pour] Moving above target: {np.round(pour_xyz, 4)}...")
            above = self.get_current_pose()
            above.translation = pour_xyz
            above.rotation = R0
            if not self.move_to_pose(above, speed="normal", reach_tol=0.05):
                return False, {"error": "Failed to move above target"}
            xyz = pour_xyz.copy()
        else:
            xyz = np.asarray(self.get_current_pose().translation, dtype=float).copy()

        tip0 = self._tip_deg(np.asarray(self.get_current_pose().rotation))
        print(f"[Pour] Starting tip {tip0:.1f}° → {pour_tip:.0f}°")

        ok, tip_now, xyz = self._tip_sequence(
            xyz=xyz,
            lean_xy=lean_xy,
            pour_tip=pour_tip,
            step_deg=step_deg,
            step_dur=step_dur,
            tip_tol=tip_tol,
            step_retries=step_retries,
            tip_advance_m=tip_advance,
        )
        if not ok:
            print("[Pour] Incomplete tip — reset_joints to clear...")
            self.go_home()
            return False, {
                "error": (
                    f"Pour tip incomplete: commanded {pour_tip:.0f}° toward base, "
                    f"measured {tip_now:.1f}°"
                ),
                "tip_before": tip0,
                "tip_after": tip_now,
            }

        print(f"[Pour] Holding {hold_duration}s at tip={tip_now:.1f}° "
              f"xyz={np.round(xyz, 4)}...")
        self.wait(hold_duration)

        if reset_after:
            print("[Pour] reset_joints (home, joint-space) after pour...")
            if not self.go_home():
                return False, {
                    "error": ("Poured, but the arm did not come back home; the robot "
                              "refused the motion (see [Safety] above)"),
                    "tip_after": tip_now,
                    "still_holding": True,
                }
        else:
            print("[Pour] Returning upright in place...")
            self._goto_orientation(xyz, R0, duration=float(max(step_dur, 4.0)))

        print(f"[Pour] Done pouring into '{target}'")
        return True, {
            "poured_into": target,
            "pour_angle": pour_tip,
            "tip_dir": "toward_base",
            "tip_before": tip0,
            "tip_after": tip_now,
            "forward_offset": fwd,
            "lateral_offset": lat,
            "tip_advance_m": tip_advance,
        }

    # ------------------------------------------------- lip over the target
    @staticmethod
    def _cup_points(top: float, height: float, radius: float, rings: int = 6,
                    per_ring: int = 32) -> np.ndarray:
        """The held cup's outline in the TOOL frame (z down): rings rim to bottom."""
        phi = np.linspace(0.0, 2 * np.pi, per_ring, endpoint=False)
        out = []
        for k in range(rings):
            z = -top + height * k / (rings - 1)          # rim first, then down
            out.append(np.c_[radius * np.cos(phi), radius * np.sin(phi),
                             np.full(per_ring, z)])
        return np.vstack(out)

    def _lip_pose(self, R: np.ndarray, cup: np.ndarray, per_ring: int,
                  lip_xy: np.ndarray, rim_z: float, lip_clear: float,
                  body_clear: float, lean_xy=(-1.0, 0.0),
                  flange_min_z: float = None) -> Tuple[np.ndarray, float]:
        """
        TCP position putting the lip at ``lip_xy``, ``lip_clear`` over the
        target's rim, raised as far as needed for every point of the cup to
        stay ``body_clear`` over it, and the flange ``flange_min_z`` or more
        above the base. Returns (xyz, lip height over the rim).
        """
        world = cup @ R.T                                  # relative to the TCP
        rim = world[:per_ring]
        # Upright, every rim point is lowest; take the one on the side the cup
        # will tip toward, so the start already sits where the pour goes on.
        lean = np.asarray(lean_xy, dtype=float)
        lip = rim[int(np.argmin(rim[:, 2] - 1e-4 * (rim[:, :2] @ lean)))]
        xy = np.asarray(lip_xy, dtype=float) - lip[:2]
        z = max(rim_z + lip_clear - lip[2],
                rim_z + body_clear - float(world[:, 2].min()))
        if flange_min_z is not None:
            # The flange is FLANGE_BEHIND_TCP back along the tool's z axis.
            z = max(z, float(flange_min_z) + FLANGE_BEHIND_TCP * float(R[2, 2]))
        return np.array([xy[0], xy[1], z]), float(z + lip[2] - rim_z)

    def _pour_over_lip(self, params, target, pour_tip, hold_duration, step_deg,
                       step_dur, tip_tol, step_retries, reset_after, lean_xy):
        top = float(params["source_top_above_tcp"])
        height = float(params["source_height"])
        radius = float(params["source_radius"])
        lip_clear = float(params["lip_clearance"])
        body_clear = float(params["body_clearance"])
        print(f"[Pour] Held cup from the pick: rim {top * 1000:+.0f}mm over the TCP, "
              f"{height * 1000:.0f}mm tall, radius {radius * 1000:.0f}mm")

        if params.get("reset_before_scan", True):
            print("[Pour] reset_joints (home, joint-space) clear of cameras before scan...")
            if not self.go_home():
                return False, {"error": "Failed to reset_joints before scanning"}
        self.wait(0.3)
        self.vision.clear_cache()
        located = self.locate_container(target, force_refresh=True)
        if located is None:
            return False, {"error": f"Cannot locate '{target}'"}
        centre = np.asarray(located["rim_center"], dtype=float)[:2]
        rim_z = float(located["top_z"])
        inside = float(located["rim_radius"])
        lip_xy = centre + lean_xy * float(params["lip_inset"]) * inside
        print(f"[Pour] '{target}' rim centre {np.round(centre, 4)}, z={rim_z:.4f}, "
              f"radius {inside * 1000:.0f}mm; the lip goes to {np.round(lip_xy, 4)}, "
              f"{lip_clear * 1000:.0f}mm over the rim")

        per_ring = 32
        cup = self._cup_points(top, height, radius, per_ring=per_ring)
        closing = np.array([-lean_xy[1], lean_xy[0]])
        angles = list(np.arange(0.0, pour_tip + 1e-6, step_deg))
        if abs(angles[-1] - pour_tip) > 0.5:
            angles.append(pour_tip)
        flange_min = params.get("flange_min_z")
        flange_min = None if flange_min is None else float(flange_min)
        plan = []
        for deg in angles:
            R = self._rotation_lean(closing, float(deg), lean_xy)
            xyz, lip_h = self._lip_pose(R, cup, per_ring, lip_xy, rim_z, lip_clear,
                                        body_clear, lean_xy, flange_min_z=flange_min)
            plan.append((float(deg), R, self._clamp_position(xyz), lip_h))
        if flange_min is not None:
            raised = [(deg, lip_h) for deg, R, xyz, lip_h in plan
                      if abs(xyz[2] - FLANGE_BEHIND_TCP * R[2, 2] - flange_min) < 1e-6]
            if raised:
                print(f"[Pour] From {raised[0][0]:.0f} deg the cup rides higher to keep the "
                      f"flange {flange_min * 1000:.0f}mm up (franka-interface's virtual "
                      f"floor): lip up to {max(h for _d, h in raised) * 1000:.0f}mm "
                      f"over the rim")

        # Over the target, higher by a margin, upright; then down onto the plan.
        start = plan[0][2].copy()
        above = start.copy()
        above[2] += 0.06
        print(f"[Pour] Over the target at {np.round(above, 4)}, then down to "
              f"{np.round(start, 4)}...")
        for xyz, R in ((above, plan[0][1]), (start, plan[0][1])):
            try:
                self._goto_orientation(xyz, R, duration=float(max(step_dur, 3.0)))
            except Exception as e:
                return False, {"error": f"Failed to reach the pour start: {e}"}

        tip_now = self._tip_deg(np.asarray(self.get_current_pose().rotation))
        lips = []
        for deg, R, xyz, lip_h in plan[1:]:
            reached = False
            for attempt in range(1, step_retries + 1):
                duration = float(step_dur * (1.0 + 0.4 * (attempt - 1)))
                print(f"[Pour] → tip {deg:.0f}°, TCP {np.round(xyz, 4)}, lip "
                      f"{lip_h * 1000:.0f}mm over the rim (attempt {attempt}/{step_retries})")
                try:
                    self._goto_orientation(xyz, R, duration=duration)
                except Exception as e:
                    print(f"[Pour]   goto_pose error: {e}")
                    continue
                tip_now = self._tip_deg(np.asarray(self.get_current_pose().rotation))
                if tip_now >= deg - tip_tol:
                    reached = True
                    break
                print(f"[Pour]   short of {deg:.0f}° ({tip_now:.1f}°) — retrying")
            if not reached:
                print("[Pour] Incomplete tip — reset_joints to clear...")
                self.go_home()
                return False, {"error": (f"Pour tip incomplete: commanded {deg:.0f}°, "
                                         f"measured {tip_now:.1f}°"),
                               "tip_after": tip_now}
            lips.append(round(lip_h * 1000, 1))

        print(f"[Pour] Holding {hold_duration}s at tip={tip_now:.1f}°...")
        self.wait(hold_duration)

        # Back upright where it is, lifted clear, before going anywhere: a
        # cup that still holds something keeps it once it is level.
        last = plan[-1][2].copy()
        last[2] += 0.04
        try:
            self._goto_orientation(last, plan[0][1], duration=float(max(step_dur, 3.0)))
        except Exception as e:
            print(f"[Pour] Warning: could not right the cup in place: {e}")
        if reset_after:
            print("[Pour] reset_joints (home, joint-space) after pour...")
            if not self.go_home():
                # 2026-10-08: the robot faulted at the end of the 90 deg step
                # and refused the righting and this; pour said "success" and
                # the next step failed 271 mm away instead.
                tilt = self._tip_deg(np.asarray(self.get_current_pose().rotation))
                return False, {"error": (f"Poured, but the arm did not come back: it is "
                                         f"still tipped {tilt:.0f} deg over '{target}', "
                                         f"holding the cup. The robot refused the motion; "
                                         f"see [Safety] above"),
                               "tip_after": tip_now, "still_holding": True}

        print(f"[Pour] Done pouring into '{target}'")
        return True, {
            "poured_into": target,
            "pour_angle": pour_tip,
            "tip_dir": "toward_base",
            "tip_after": tip_now,
            "mode": "lip_over_target",
            "lip_over_rim_mm": lips,
            "lip_xy": [float(v) for v in lip_xy],
            "target_rim_z": rim_z,
        }

    def _tip_sequence(
        self,
        xyz: np.ndarray,
        lean_xy: np.ndarray,
        pour_tip: float,
        step_deg: float,
        step_dur: float,
        tip_tol: float,
        step_retries: int,
        tip_advance_m: float,
    ) -> Tuple[bool, float, np.ndarray]:
        """
        Step tip toward base. Each successful tip step also nudges EE
        forward (+X) by ``tip_advance_m`` so the mouth stays over the cup.

        Returns (ok, final_tip_deg, final_xyz).
        """
        closing = np.array([-lean_xy[1], lean_xy[0]])
        waypoints = list(np.arange(step_deg, pour_tip + 1e-6, step_deg))
        if not waypoints or abs(waypoints[-1] - pour_tip) > 0.5:
            waypoints.append(pour_tip)

        xyz = np.asarray(xyz, dtype=float).copy()
        tip_now = self._tip_deg(np.asarray(self.get_current_pose().rotation))
        for goal_tip in waypoints:
            goal_tip = float(goal_tip)
            # Advance forward with tip so the mouth does not drift toward base.
            xyz = xyz.copy()
            xyz[0] += tip_advance_m
            xyz = self._clamp_position(xyz)
            R_cmd = self._rotation_lean(closing, goal_tip, lean_xy)
            reached = False
            for attempt in range(1, step_retries + 1):
                duration = float(step_dur * (1.0 + 0.4 * (attempt - 1)))
                print(f"[Pour] → tip {goal_tip:.0f}°, xyz={np.round(xyz, 4)} "
                      f"(+{tip_advance_m * 100:.2f} cm X this step; "
                      f"attempt {attempt}/{step_retries}, {duration:.1f}s)...")
                try:
                    self._goto_orientation(xyz, R_cmd, duration=duration)
                except Exception as e:
                    print(f"[Pour]   goto_pose error: {e}")
                    continue

                R_act = np.asarray(self.get_current_pose().rotation, dtype=float)
                tip_now = self._tip_deg(R_act)
                lean_now = float(R_act[:2, 2] @ lean_xy)
                print(f"[Pour]   measured tip={tip_now:.1f}°, lean·dir={lean_now:.3f}")

                if lean_now < -0.05:
                    print("[Pour]   leaning the wrong way — abort")
                    return False, tip_now, xyz

                if tip_now >= goal_tip - tip_tol:
                    reached = True
                    break
                print(f"[Pour]   short of {goal_tip:.0f}° "
                      f"(need ≥ {goal_tip - tip_tol:.0f}°) — retrying")

            if not reached:
                print(f"[Pour]   giving up on {goal_tip:.0f}° step "
                      f"(last tip {tip_now:.1f}°)")
                return False, tip_now, xyz

        if tip_now < pour_tip - tip_tol:
            print(f"[Pour] final tip {tip_now:.1f}° short of "
                  f"{pour_tip - tip_tol:.0f}° required")
            return False, tip_now, xyz

        print(f"[Pour] OK: tip {tip_now:.1f}° (goal {pour_tip:.0f}°), "
              f"final xyz={np.round(xyz, 4)}")
        return True, tip_now, xyz

    def _goto_orientation(self, xyz: np.ndarray, R: np.ndarray, duration: float) -> None:
        pose = self.get_current_pose()
        pose.translation = np.asarray(xyz, dtype=float).copy()
        pose.rotation = _orthonormalize(np.asarray(R, dtype=float))
        self.robot.goto_pose(
            to_rigid_transform(pose),
            duration=float(duration),
            use_impedance=False,
        )
        self.wait(0.15)

    @staticmethod
    def _tip_deg(R: np.ndarray) -> float:
        tip = R[:3, 2]
        return float(np.degrees(np.arccos(np.clip(-tip[2], -1.0, 1.0))))

    @staticmethod
    def _rotation_lean(closing_xy: np.ndarray, tip_deg: float,
                       lean_xy: np.ndarray) -> np.ndarray:
        tip_rad = np.radians(float(max(tip_deg, 0.0)))
        lean = np.asarray(lean_xy, dtype=float)[:2]
        lean = lean / (np.linalg.norm(lean) + 1e-9)

        y_axis = np.array([closing_xy[0], closing_xy[1], 0.0], dtype=float)
        y_axis = y_axis / (np.linalg.norm(y_axis) + 1e-9)
        down = np.array([0.0, 0.0, -1.0])
        x0 = np.cross(y_axis, down)
        x0 = x0 / (np.linalg.norm(x0) + 1e-9)

        best_R, best_align = None, -1e9
        for signed in (tip_rad, -tip_rad):
            c, s = np.cos(signed), np.sin(signed)
            z_axis = c * down - s * x0
            x_axis = c * x0 + s * down
            z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-9)
            x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-9)
            y = np.cross(z_axis, x_axis)
            y[2] = 0.0
            y = y / (np.linalg.norm(y) + 1e-9)
            x_axis = np.cross(y, z_axis)
            x_axis = x_axis / (np.linalg.norm(x_axis) + 1e-9)
            z_axis = np.cross(x_axis, y)
            z_axis = z_axis / (np.linalg.norm(z_axis) + 1e-9)
            R = np.column_stack([x_axis, y, z_axis])
            align = float(R[:2, 2] @ lean)
            if align > best_align:
                best_align, best_R = align, R
        return best_R
