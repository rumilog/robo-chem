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
                    "error": "Pour tip ok but reset_joints after hold failed",
                    "tip_after": tip_now,
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
