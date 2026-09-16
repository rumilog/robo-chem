"""
``SimFrankaArm`` -- a drop-in replacement for ``frankapy.FrankaArm``.

The skills only ever touch thirteen methods on the arm (get_pose, goto_pose,
goto_joints, get_joints, reset_joints, the four gripper calls, get_gripper_width,
get_gripper_is_grasped, stop_skill, is_skill_done). This class implements that
surface against MuJoCo, so ``SkillsExecutor`` runs unmodified: pick_up, pour and
scoop issue the same Cartesian targets they send to the real robot and you watch
the arm follow them.

Motion is Cartesian, like frankapy's goto_pose: the TCP is interpolated along a
straight line with slerped orientation, and every waypoint is solved by damped
least-squares IK over the seven arm joints only. A target the arm cannot reach
leaves it short, exactly as on hardware, so the skills' own arrival checks fire
instead of being papered over.

Grasping is kinematic by default (``grasp_mode="magnet"``): closing the jaws on
a prop welds it to the hand and reports the prop's width back through
``get_gripper_width``, which is what a real grasp stalling on an object looks
like. ``grasp_mode="physics"`` leaves it to friction instead.
"""

from __future__ import annotations

import gc
import math
import time
from typing import Dict, List, Optional, Tuple

import mujoco
import numpy as np

from .bench import Prop
from .rigid_transform import RigidTransform, install as install_rigid_transform
from .scene import ARM_JOINTS, HOME_JOINTS, SimScene, reset_scene

install_rigid_transform()

FRANKA_TOOL_FRAME = "franka_tool"
WORLD_FRAME = "world"

# Panda joint limits, from the vendor model.
JOINT_LOW = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
JOINT_HIGH = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

MAX_GRIPPER_WIDTH = 0.08
# actuator8 maps ctrl 0..255 onto a 0..40 mm finger travel (0..80 mm opening).
CTRL_PER_METRE = 255.0 / MAX_GRIPPER_WIDTH

# The box, in the TCP frame, a prop's grasp point must fall inside for a
# closing gripper to pick it up. z is negative back towards the fingers.
JAW_BOX = np.array([0.035, 0.05, 0.065])


def _slerp(r_a: np.ndarray, r_b: np.ndarray, t: float) -> np.ndarray:
    """Interpolate between two rotation matrices along the shortest arc."""
    qa, qb = np.zeros(4), np.zeros(4)
    mujoco.mju_mat2Quat(qa, np.ascontiguousarray(r_a).reshape(9))
    mujoco.mju_mat2Quat(qb, np.ascontiguousarray(r_b).reshape(9))
    dot = float(qa @ qb)
    if dot < 0:                       # same rotation, opposite hemisphere
        qb, dot = -qb, -dot
    if dot > 0.9995:                  # nearly parallel: lerp, slerp goes singular
        q = qa + t * (qb - qa)
    else:
        theta = math.acos(np.clip(dot, -1.0, 1.0))
        q = (math.sin((1 - t) * theta) * qa + math.sin(t * theta) * qb) / math.sin(theta)
    q = q / np.linalg.norm(q)
    out = np.zeros(9)
    mujoco.mju_quat2Mat(out, q)
    return out.reshape(3, 3)


def _rotation_error(current: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rotation vector taking ``current`` onto ``target``, in world axes."""
    q_cur, q_tgt, q_err, vel = np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(3)
    mujoco.mju_mat2Quat(q_cur, np.ascontiguousarray(current).reshape(9))
    mujoco.mju_mat2Quat(q_tgt, np.ascontiguousarray(target).reshape(9))
    mujoco.mju_negQuat(q_cur, q_cur)
    mujoco.mju_mulQuat(q_err, q_tgt, q_cur)
    mujoco.mju_quat2Vel(vel, q_err, 1.0)
    return vel


class SimFrankaArm:
    """MuJoCo-backed stand-in for ``frankapy.FrankaArm``."""

    def __init__(
        self,
        scene: SimScene,
        viewer: bool = True,
        realtime: bool = True,
        speed: float = 1.0,
        grasp_mode: str = "magnet",
        verbose: bool = True,
    ):
        """
        Args:
            scene: Compiled scene from :func:`robochem.sim.scene.build_scene`
            viewer: Open the interactive MuJoCo window
            realtime: Play motions at wall-clock speed (off = as fast as possible)
            speed: Multiplier on commanded durations; 2.0 halves every move
            grasp_mode: ``"magnet"`` (kinematic attach) or ``"physics"``
            verbose: Print every command, mirroring frankapy's chatter
        """
        if grasp_mode not in ("magnet", "physics"):
            raise ValueError(f"grasp_mode must be 'magnet' or 'physics', got {grasp_mode!r}")

        self.scene = scene
        self.model = scene.model
        self.data = scene.data
        self.realtime = realtime
        self.speed = float(speed)
        self.grasp_mode = grasp_mode
        self.verbose = verbose

        # Mirrors FakeArm in scripts/test_skills_offline.py so the same
        # assertions can run against the simulator.
        self.commands: List[dict] = []
        self.gripper_commands: List[dict] = []

        self._attached: Dict[str, np.ndarray] = {}   # prop name -> TCP^-1 * prop pose
        self._held_width: Optional[float] = None
        self._viewer = None
        if viewer:
            self._open_viewer()
        self._sync()

    # ---------------------------------------------------------------- viewer

    def _open_viewer(self):
        try:
            import mujoco.viewer

            self._viewer = mujoco.viewer.launch_passive(
                self.model, self.data, show_left_ui=False, show_right_ui=False
            )
            self._viewer.cam.lookat[:] = [0.45, 0.0, 0.15]
            self._viewer.cam.distance = 1.4
            self._viewer.cam.azimuth = 150
            self._viewer.cam.elevation = -25
        except Exception as exc:                      # no display, remote shell
            print(f"[sim] Viewer unavailable ({exc}); running headless")
            self._viewer = None

    def _sync(self):
        if self._viewer is not None and self._viewer.is_running():
            self._viewer.sync()

    def close(self):
        """
        Close the viewer window.

        The handle has to be collected here, not merely closed: left alive
        until interpreter shutdown, GLFW tears its context down in the wrong
        order and the process dies with SIGSEGV after the run has already
        finished, which looks exactly like a crashed experiment.
        """
        if self._viewer is not None:
            viewer, self._viewer = self._viewer, None
            viewer.close()
            del viewer
            gc.collect()
            time.sleep(0.2)

    def hold(self, seconds: float = 1e9):
        """Keep the window open and the physics ticking after a run finishes."""
        if self._viewer is None:
            return
        end = time.time() + seconds
        while time.time() < end and self._viewer.is_running():
            self._step(1)
            time.sleep(self.model.opt.timestep)

    # ------------------------------------------------------------ kinematics

    def _tcp(self) -> Tuple[np.ndarray, np.ndarray]:
        sid = self.scene.tool_site
        return (self.data.site_xpos[sid].copy(),
                self.data.site_xmat[sid].copy().reshape(3, 3))

    def get_pose(self):
        """Current TCP pose as a ``franka_tool`` -> ``world`` RigidTransform."""
        pos, mat = self._tcp()
        return RigidTransform(rotation=mat, translation=pos,
                              from_frame=FRANKA_TOOL_FRAME, to_frame=WORLD_FRAME)

    def get_joints(self) -> np.ndarray:
        return self.data.qpos[:7].copy()

    def _solve_ik(self, target_pos, target_mat, q_seed,
                  iters: int = 40, tol: float = 5e-4) -> np.ndarray:
        """
        Damped least squares IK over the seven arm joints.

        Runs on ``scene.ik_model`` -- the arm alone. Solving on the full scene
        would let the optimiser "reach" a target by sliding a free-floating cup
        through the jacobian of its own free joint.
        """
        model, data = self.scene.ik_model, self.scene.ik_data
        sid = self.scene.ik_tool_site
        q = np.clip(np.asarray(q_seed, float).copy(), JOINT_LOW, JOINT_HIGH)

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        damping = 1e-2
        rest = HOME_JOINTS

        for _ in range(iters):
            data.qpos[:7] = q
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)

            err_p = np.asarray(target_pos, float) - data.site_xpos[sid]
            err_r = _rotation_error(data.site_xmat[sid].reshape(3, 3), target_mat)
            err = np.concatenate([err_p, err_r])
            if np.linalg.norm(err_p) < tol and np.linalg.norm(err_r) < 10 * tol:
                break

            mujoco.mj_jacSite(model, data, jacp, jacr, sid)
            jac = np.vstack([jacp[:, :7], jacr[:, :7]])

            jjt = jac @ jac.T + (damping ** 2) * np.eye(6)
            dq = jac.T @ np.linalg.solve(jjt, err)

            # Nullspace pull toward the home posture keeps elbow flips out of
            # the preview; it costs nothing when the task jacobian is full rank.
            null = (np.eye(7) - jac.T @ np.linalg.solve(jjt, jac)) @ (rest - q)
            dq += 0.1 * null

            step = np.linalg.norm(dq)
            if step > 0.35:
                dq *= 0.35 / step
            q = np.clip(q + dq, JOINT_LOW, JOINT_HIGH)

        return q

    # ------------------------------------------------------------- execution

    def _step(self, n: int = 1):
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)
            self._follow_attached()

    def _follow_attached(self):
        """Carry welded props along with the hand."""
        if not self._attached:
            return
        pos, mat = self._tcp()
        tcp = np.eye(4)
        tcp[:3, :3], tcp[:3, 3] = mat, pos
        for name, rel in self._attached.items():
            world = tcp @ rel
            adr = self.scene.prop_qpos[name]
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, np.ascontiguousarray(world[:3, :3]).reshape(9))
            self.data.qpos[adr:adr + 3] = world[:3, 3]
            self.data.qpos[adr + 3:adr + 7] = quat
            dof = self.model.jnt_dofadr[self.model.body_jntadr[self.scene.prop_bodies[name]]]
            self.data.qvel[dof:dof + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _run(self, duration: float, ctrl_fn):
        """Step physics for ``duration`` seconds, calling ``ctrl_fn(t)`` each step."""
        dt = self.model.opt.timestep
        span = max(duration / max(self.speed, 1e-6), dt)
        steps = max(int(round(span / dt)), 1)
        sync_every = max(int(round(1 / 60 / dt)), 1)
        start = time.time()

        for i in range(steps):
            ctrl_fn(min((i + 1) / steps, 1.0))
            self._step(1)
            if i % sync_every == 0:
                self._sync()
                if self.realtime:
                    lag = (i * dt) - (time.time() - start)
                    if lag > 0:
                        time.sleep(lag)
        self._sync()

    def _settle(self, max_seconds: float = 1.5, tol: float = 2e-4):
        """
        Let the servos converge on the last commanded joint target.

        A move that stopped stepping the instant its trajectory ended would
        report a TCP still millimetres behind the setpoint, and the skills read
        that pose back through their own arrival checks. Settling is timed in
        simulated seconds and is deliberately not scaled by ``speed``: it is
        convergence, not part of the motion being previewed.
        """
        dt = self.model.opt.timestep
        sync_every = max(int(round(1 / 60 / dt)), 1)
        for i in range(int(max_seconds / dt)):
            self._step(1)
            if i % sync_every == 0:
                self._sync()
            if (np.abs(self.data.qvel[:7]).max() < 1e-3
                    and np.abs(self.data.ctrl[:7] - self.data.qpos[:7]).max() < tol):
                break
        self._sync()

    def goto_pose(self, tool_pose, duration: float = 3.0, use_impedance: bool = True,
                  block: bool = True, **kwargs):
        """
        Move the TCP along a straight line to ``tool_pose``.

        Accepts a RigidTransform or a 4x4 matrix, like the skills produce.
        """
        target_pos, target_mat = _as_pose(tool_pose)
        start_pos, start_mat = self._tcp()
        self.commands.append({
            "translation": target_pos.copy(),
            "duration": float(duration),
            "use_impedance": bool(use_impedance),
        })
        if self.verbose:
            print(f"[sim] goto_pose -> {np.round(target_pos, 4).tolist()} "
                  f"({duration:.1f}s)")

        seed = self.get_joints()
        waypoints = max(int(duration * 20), 2)          # ~20 IK solves per second
        plan = []
        for k in range(1, waypoints + 1):
            frac = k / waypoints
            pos = start_pos + (target_pos - start_pos) * frac
            mat = _slerp(start_mat, target_mat, frac)
            seed = self._solve_ik(pos, mat, seed)
            plan.append(seed.copy())
        plan = np.asarray(plan)

        def ctrl(t):
            idx = min(int(t * waypoints), waypoints - 1)
            self.data.ctrl[:7] = plan[idx]

        self._run(duration, ctrl)
        self._settle()
        return True

    def goto_joints(self, joints, duration: float = 3.0, use_impedance: bool = False,
                    block: bool = True, **kwargs):
        """Interpolate the seven joints to ``joints``."""
        target = np.clip(np.asarray(joints, float)[:7], JOINT_LOW, JOINT_HIGH)
        start = self.get_joints()
        self.commands.append({"joints": target.copy(), "duration": float(duration)})
        if self.verbose:
            print(f"[sim] goto_joints ({duration:.1f}s)")

        def ctrl(t):
            self.data.ctrl[:7] = start + (target - start) * t

        self._run(duration, ctrl)
        self._settle()
        return True

    def reset_joints(self, duration: float = 4.0, **kwargs):
        """Drive to frankapy's home posture."""
        if self.verbose:
            print("[sim] reset_joints -> home")
        return self.goto_joints(HOME_JOINTS, duration=duration)

    # --------------------------------------------------------------- gripper

    def get_gripper_width(self) -> float:
        if self._held_width is not None:
            return float(self._held_width)
        return float(self.data.qpos[7] + self.data.qpos[8])

    def get_gripper_is_grasped(self) -> bool:
        if self.grasp_mode == "magnet":
            return bool(self._attached)
        width = self.get_gripper_width()
        return 0.005 < width < 0.075

    def goto_gripper(self, width: float, grasp: bool = False, force: float = None,
                     speed: float = None, block: bool = True, **kwargs):
        """Command a jaw opening. Closing onto a prop picks it up."""
        width = float(np.clip(width, 0.0, MAX_GRIPPER_WIDTH))
        previous = self.get_gripper_width()
        self.gripper_commands.append({"width": width, "grasp": grasp, "force": force})
        if self.verbose:
            print(f"[sim] gripper -> {width * 1000:.1f}mm (grasp={grasp})")

        opening = width > previous + 2e-3
        if opening and self._attached:
            self._release()

        self.data.ctrl[7] = width * CTRL_PER_METRE
        self._run(kwargs.get("duration", 1.0), lambda t: None)

        if not opening and self.grasp_mode == "magnet" and not self._attached:
            self._try_grasp(width)
        return True

    def open_gripper(self, block: bool = True, **kwargs):
        return self.goto_gripper(MAX_GRIPPER_WIDTH, grasp=False, block=block)

    def close_gripper(self, grasp: bool = True, block: bool = True, **kwargs):
        return self.goto_gripper(0.0, grasp=grasp, block=block)

    def stop_gripper(self):
        self.data.ctrl[7] = self.get_gripper_width() * CTRL_PER_METRE

    def stop_skill(self):
        self.data.ctrl[:7] = self.get_joints()

    def is_skill_done(self) -> bool:
        return True

    # -------------------------------------------------------------- grasping

    def _try_grasp(self, commanded_width: float):
        """Attach the prop whose grasp point sits between the closing jaws."""
        pos, mat = self._tcp()
        for prop in self.scene.bench.props:
            bid = self.scene.prop_bodies[prop.name]
            body_mat = self.data.xmat[bid].reshape(3, 3)
            point = self.data.xpos[bid] + body_mat @ np.asarray(prop.grasp_offset)
            local = mat.T @ (point - pos)
            if not np.all(np.abs(local) < JAW_BOX):
                continue
            if commanded_width > prop.grasp_width + 0.012:
                continue          # jaws never came together on it

            tcp = np.eye(4)
            tcp[:3, :3], tcp[:3, 3] = mat, pos
            world = np.eye(4)
            world[:3, :3] = body_mat
            world[:3, 3] = self.data.xpos[bid]
            self._attached[prop.name] = np.linalg.inv(tcp) @ world

            # A real grasp stalls the fingers on the object; the skills read
            # that width back to confirm they are holding something.
            self._held_width = min(prop.grasp_width, MAX_GRIPPER_WIDTH)
            self.data.qpos[7] = self.data.qpos[8] = self._held_width / 2
            mujoco.mj_forward(self.model, self.data)
            if self.verbose:
                print(f"[sim] grasped {prop.name!r} "
                      f"(width {self._held_width * 1000:.1f}mm)")
            return
        if self.verbose:
            print("[sim] jaws closed on nothing")

    def _release(self):
        for name in list(self._attached):
            if self.verbose:
                print(f"[sim] released {name!r}")
        self._attached.clear()
        self._held_width = None

    @property
    def holding(self) -> Optional[str]:
        """Name of the prop currently in the jaws, or None."""
        return next(iter(self._attached), None)

    # ----------------------------------------------------------------- admin

    def reset_scene(self):
        """Put the bench and the arm back to their starting state."""
        self._attached.clear()
        self._held_width = None
        reset_scene(self.scene)
        self._sync()


def _as_pose(pose) -> Tuple[np.ndarray, np.ndarray]:
    """Pull (position, rotation) out of a RigidTransform or a 4x4 matrix."""
    if hasattr(pose, "translation") and hasattr(pose, "rotation"):
        return (np.asarray(pose.translation, float).copy(),
                np.asarray(pose.rotation, float).copy())
    matrix = np.asarray(pose, float)
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected a 4x4 pose or a RigidTransform, got {matrix.shape}")
    return matrix[:3, 3].copy(), matrix[:3, :3].copy()
