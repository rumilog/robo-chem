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
from collections import deque
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

# Wrench filtering: one sample every 10 physics steps (20ms), 5 kept -> a 100ms
# window, which is the order a real F/T reading is filtered over.
WRENCH_EVERY = 10
WRENCH_SAMPLES = 5


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


def _lerp_plan(q_start: np.ndarray, plan: np.ndarray, t: float) -> np.ndarray:
    """
    Joint target at fraction ``t`` of a plan whose k-th sample (1-based) is due
    at t = k/len(plan), ramped linearly from ``q_start``.

    Holding each sample instead (zero-order hold) stepped the servo target by
    up to 0.045 rad every few dozen steps: the wrist hit its 12 Nm torque limit,
    the held tool was kicked, and each kick flipped hundreds of contact states
    in the solver. The endpoint is returned exactly, so the final target -- and
    what _settle converges to -- is unchanged.
    """
    n = len(plan)
    if t >= 1.0:
        return plan[-1].copy()
    s = max(t, 0.0) * n
    k = int(s)
    a = q_start if k == 0 else plan[k - 1]
    return a + (plan[k] - a) * (s - k)


def _hermite_plan(q_start: np.ndarray, plan: np.ndarray):
    """
    A C1 joint reference through ``q_start`` and every plan sample (sample k
    due at u = k/len(plan)), returning ``ref(u) -> (q, dq/du)``.

    Cubic Hermite segments with Catmull-Rom tangents inside and ZERO tangent at
    both ends, so the path starts and ends at rest and its velocity never
    jumps. The velocity is what the feed-forward in follow_pose_path is built
    from; a piecewise-linear reference would make it jump at every sample and
    those jumps saturate the wrist exactly as the old staircase did.
    """
    knots = np.vstack([q_start, plan])
    n = len(knots) - 1
    h = 1.0 / n
    tangents = np.zeros_like(knots)
    tangents[1:-1] = (knots[2:] - knots[:-2]) / (2.0 * h)

    def ref(u: float):
        s = min(max(u, 0.0), 1.0) * n
        k = min(int(s), n - 1)
        x = s - k
        p0, p1 = knots[k], knots[k + 1]
        m0, m1 = tangents[k] * h, tangents[k + 1] * h
        q = ((2 * x**3 - 3 * x**2 + 1) * p0 + (x**3 - 2 * x**2 + x) * m0
             + (-2 * x**3 + 3 * x**2) * p1 + (x**3 - x**2) * m1)
        dq_dx = ((6 * x**2 - 6 * x) * p0 + (3 * x**2 - 4 * x + 1) * m0
                 + (-6 * x**2 + 6 * x) * p1 + (3 * x**2 - 2 * x) * m1)
        return q, dq_dx * n

    return ref


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
        # ~100ms of wrench history, sampled every WRENCH_EVERY steps. Summing
        # contacts on every step would dominate the step cost for no gain.
        self._wrench_hist: deque = deque(maxlen=WRENCH_SAMPLES)
        self._wrench_tick = 0
        self.last_settle: Tuple[int, str] = (0, "none")   # (steps, why it ended)
        self._last_sync = 0.0                              # wall clock of the last frame

        # Velocity feed-forward gain per arm joint. A position servo
        # (kp, kv) with joint damping b trails a ramp by (kv + b)/kp * velocity:
        # 0.1 s here on every joint, so a 1 s scoop roll ran 6-14 deg behind
        # its command, and the tip -- whose height depends most on the tilt
        # near level -- went up to 7 mm below its plan and into the cup floor.
        # Commanding q + gain * qdot cancels that ramp lag, which is what the
        # real controller's feed-forward does.
        kp = self.model.actuator_gainprm[:7, 0]
        kv = -self.model.actuator_biasprm[:7, 2]
        damping = np.array([
            self.model.dof_damping[self.model.jnt_dofadr[
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, j)]]
            for j in ARM_JOINTS])
        self._ff_gain = (kv + damping) / kp

        # Collision bits so a MAGNET-held prop stops touching the hand that
        # carries it. MuJoCo pairs two geoms when (contype1 & conaffinity2) or
        # (contype2 & conaffinity1). Hand and finger geoms move to contype 2 /
        # conaffinity 3, which pairs them with everything exactly as before;
        # a prop, while held, goes to contype 4 / conaffinity 1, which still
        # pairs it with the cup, the table and the other props but not with
        # the hand. Why: the held prop is placed by teleport a step behind the
        # hand, so when the wrist turns fast its handle sinks into the finger
        # pads and the contact shoves the ARM -- at --speed 2 it pushed the arm
        # 17 mm sideways during a settle, onto the rim of the dish.
        self._hand_geoms = [
            g for g in range(self.model.ngeom)
            if self.model.geom_contype[g] or self.model.geom_conaffinity[g]
            if mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                 self.model.geom_bodyid[g])
            in ("hand", "left_finger", "right_finger")]
        for g in self._hand_geoms:
            self.model.geom_contype[g] = 2
            self.model.geom_conaffinity[g] = 3
        self._held_bits: Dict[str, List[Tuple[int, int, int]]] = {}
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
        self._last_sync = time.time()

    # Viewer frames are paced by WALL clock. They used to be paced by sim time
    # (one sync per 1/60 s = 8 steps), which tied the frame rate to the cost of
    # a step: with a powder bed a step costs 20-100 ms, so a dig showed one
    # frame every 0.2-0.8 s. Now a step that crosses a 1/60 s wall boundary is
    # shown, whatever it cost. Headless, _sync does nothing, so nothing changes.
    FRAME_WALL_S = 1 / 60

    def _frame(self):
        if time.time() - self._last_sync >= self.FRAME_WALL_S:
            self._sync()

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

    def settle(self, seconds: float = 0.5):
        """Run physics for a while without a viewer, so free props stop moving."""
        self._step(max(1, int(seconds / self.model.opt.timestep)))

    def hold(self, seconds: float = 1e9):
        """Keep the window open and the physics ticking after a run finishes."""
        if self._viewer is None:
            return
        end = time.time() + seconds
        while time.time() < end and self._viewer.is_running():
            self._step(1)
            self._frame()
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

    def _tool_bodies(self) -> set:
        """Everything rigidly moving with the wrist: hand, fingers, held prop."""
        ids = set()
        for name in ("hand", "left_finger", "right_finger", "mounted_tool"):
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                ids.add(bid)
        for name in self._attached:
            ids.add(self.scene.prop_bodies[name])
        return ids

    def get_ee_force_torque(self) -> np.ndarray:
        """
        Filtered external wrench on the end effector, base frame.

        The hardware reading is already filtered, and a single MuJoCo step is
        not: contact impulses chatter as the solver redistributes them, so an
        instantaneous sample swings by several newtons while the tool sits
        still. This returns the mean of the recent history that ``_step``
        collects, which is the comparable quantity.
        """
        if self._wrench_hist:
            return np.mean(self._wrench_hist, axis=0)
        return self._compute_ee_wrench()

    def _compute_ee_wrench(self) -> np.ndarray:
        """
        One instantaneous sample of the external wrench, base frame.

        Matches ``frankapy.FrankaArm.get_ee_force_torque()``: six floats, force
        then torque, in world coordinates, and signed as the force the world
        exerts ON the robot -- so pressing the tool down onto something reads a
        positive Z.

        A magnet grasp carries the held prop kinematically, so nothing it
        touches is transmitted through the finger joints and a wrist sensor
        would read zero. Instead this sums the contacts on everything moving
        with the wrist, the held prop included, which is what a rigid grasp
        would pass up the arm.
        """
        tool = self._tool_bodies()
        tcp, _ = self._tcp()
        wrench = np.zeros(6)
        buf = np.zeros(6)

        for i in range(self.data.ncon):
            con = self.data.contact[i]
            b1 = self.model.geom_bodyid[con.geom1]
            b2 = self.model.geom_bodyid[con.geom2]
            on1, on2 = b1 in tool, b2 in tool
            if on1 == on2:                 # both ours, or neither: not external
                continue

            mujoco.mj_contactForce(self.model, self.data, i, buf)
            # contact.frame holds the frame axes as rows, so its transpose maps
            # a contact-frame vector into world.
            force = con.frame.reshape(3, 3).T @ buf[:3]
            # mj_contactForce reports the contact force on geom2's body, so it
            # is already the force on us when the tool is geom2; flip it when
            # the tool is geom1. Verified by pressing the stirrer down onto the
            # holder and requiring a positive Z.
            if on1:
                force = -force
            wrench[:3] += force
            wrench[3:] += np.cross(np.asarray(con.pos) - tcp, force)

        return wrench

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
            self._wrench_tick += 1
            if self._wrench_tick % WRENCH_EVERY == 0:
                self._wrench_hist.append(self._compute_ee_wrench())

    def _follow_attached(self):
        """
        Carry welded props along with the hand.

        The pose is set directly, but the velocity is set to match it rather
        than to zero. A body whose position jumps every step while it reports
        standing still is one the solver resolves by flinging whatever rests on
        it: a scoop carried that way reaches the top of the stroke empty every
        single time, however good the stroke was. Measured on the same lift,
        same bed: 0 granules carried with the velocity zeroed, 4 with it set.

        Nothing else notices -- ``_settle`` watches the arm's joints, not the
        prop's -- but anything resting in a held container now behaves.
        """
        if not self._attached:
            return
        dt = self.model.opt.timestep
        pos, mat = self._tcp()
        tcp = np.eye(4)
        tcp[:3, :3], tcp[:3, 3] = mat, pos
        for name, rel in self._attached.items():
            world = tcp @ rel
            adr = self.scene.prop_qpos[name]
            quat = np.zeros(4)
            mujoco.mju_mat2Quat(quat, np.ascontiguousarray(world[:3, :3]).reshape(9))

            was_pos = self.data.qpos[adr:adr + 3].copy()
            was_quat = self.data.qpos[adr + 3:adr + 7].copy()
            self.data.qpos[adr:adr + 3] = world[:3, 3]
            self.data.qpos[adr + 3:adr + 7] = quat

            dof = self.model.jnt_dofadr[self.model.body_jntadr[self.scene.prop_bodies[name]]]
            step = world[:3, 3] - was_pos
            if np.linalg.norm(step) > 0.02:
                # The prop was just snapped into the jaws from where it sat on
                # the table. That is a re-seat, not motion; implying a velocity
                # from it would launch it across the bench.
                self.data.qvel[dof:dof + 6] = 0.0
                continue
            omega = np.zeros(3)
            mujoco.mju_subQuat(omega, quat, was_quat)
            self.data.qvel[dof:dof + 3] = step / dt
            self.data.qvel[dof + 3:dof + 6] = omega / dt
        # Refresh POSES only. Everything read between steps -- the viewer,
        # get_pose, Film/SimVision renders, grains_in_bowl -- needs body, geom,
        # site, camera and light poses, and mj_step re-runs the full forward
        # pass itself from qpos/qvel. A trailing mj_forward here repeated the
        # ~1000-contact constraint solve for nothing: it doubled the cost of
        # every step with a prop in the jaws and changed the trajectory by
        # exactly zero bits (checked over a full 1083-step sweep and settle).
        # data.contact / efc_* / actuator_force are therefore left from
        # mj_step's own pass, one step old; film_sim_skill's ContactWatch
        # refreshes contacts itself for that reason.
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        mujoco.mj_camlight(self.model, self.data)

    def _run(self, duration: float, ctrl_fn):
        """Step physics for ``duration`` seconds, calling ``ctrl_fn(t)`` each step."""
        dt = self.model.opt.timestep
        span = max(duration / max(self.speed, 1e-6), dt)
        steps = max(int(round(span / dt)), 1)
        start = time.time()

        for i in range(steps):
            ctrl_fn(min((i + 1) / steps, 1.0))
            self._step(1)
            if self.realtime:
                # Hold wall = sim when steps are cheaper than realtime; sleep in
                # chunks of at least 4 ms rather than every 2 ms step.
                lag = (i + 1) * dt - (time.time() - start)
                if lag > 0.004:
                    time.sleep(lag)
            self._frame()
        self._sync()

    def _servo_pose_error(self) -> Tuple[float, float]:
        """
        How far the TCP is (metres, radians) from where the servos are driving
        it: forward kinematics of ctrl on the arm-only IK model.
        """
        model, data, sid = self.scene.ik_model, self.scene.ik_data, self.scene.ik_tool_site
        data.qpos[:7] = self.data.ctrl[:7]
        mujoco.mj_kinematics(model, data)
        pos, mat = self._tcp()
        return (float(np.linalg.norm(data.site_xpos[sid] - pos)),
                float(np.linalg.norm(_rotation_error(mat, data.site_xmat[sid].reshape(3, 3)))))

    def _settle(self, max_seconds: float = 1.5, tol: float = 2e-4, window: float = 0.1,
                min_gain: float = 0.2, stall_pos_tol: float = 1e-3,
                stall_rot_tol: float = 5e-3):
        """
        Let the servos converge on the last commanded joint target.

        A move that stopped stepping the instant its trajectory ended would
        report a TCP still millimetres behind the setpoint, and the skills read
        that pose back through their own arrival checks. Settling is timed in
        simulated seconds and is deliberately not scaled by ``speed``: it is
        convergence, not part of the motion being previewed.

        It also stops when waiting has stopped helping. A free servo here is
        overdamped with its slow pole at kp/kv = 10/s on every joint, so its
        joint error falls by about e^-1 per 0.1 s window. If a window's peak
        error is not at least ``min_gain`` below the previous window's AND the
        TCP is already within ``stall_pos_tol`` / ``stall_rot_tol`` of FK(ctrl),
        more time cannot move it. That was the held scoop in a powder bed: the
        plain test never held, and every dig burned the full 1.5 s. In free
        space the plain test always fires first, so behaviour there is
        unchanged; an arm that is blocked or still slewing is far from
        FK(ctrl) and runs to the timeout as before.

        Why it ended is kept in ``last_settle`` as (steps, reason), so a test
        can notice the stall branch starting to fire somewhere new.
        """
        dt = self.model.opt.timestep
        win = max(int(round(window / dt)), 1)
        prev_peak, peak = None, 0.0
        steps, reason = int(max_seconds / dt), "timeout"
        for i in range(int(max_seconds / dt)):
            self._step(1)
            self._frame()
            err = float(np.abs(self.data.ctrl[:7] - self.data.qpos[:7]).max())
            if np.abs(self.data.qvel[:7]).max() < 1e-3 and err < tol:
                steps, reason = i + 1, "converged"
                break
            peak = max(peak, err)
            if (i + 1) % win == 0:
                if prev_peak is not None and peak > (1.0 - min_gain) * prev_peak:
                    dp, dr = self._servo_pose_error()
                    if dp < stall_pos_tol and dr < stall_rot_tol:
                        steps, reason = i + 1, "stalled"
                        break
                prev_peak, peak = peak, 0.0
        self.last_settle = (steps, reason)
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
        q_start = seed.copy()
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
            # Ramp between plan samples rather than holding each one; see
            # _lerp_plan. No feed-forward here: a straight move starts and
            # stops abruptly, and a lead term would overshoot the stop.
            self.data.ctrl[:7] = _lerp_plan(q_start, plan, t)

        self._run(duration, ctrl)
        self._settle()
        return True

    def follow_pose_path(self, points, rotations=None, duration: float = 3.0,
                         **kwargs):
        """
        Run a whole path as ONE motion, the way frankapy's dynamic mode does.

        A chain of ``goto_pose`` calls cannot be smooth here for the same reason
        it cannot be smooth on the robot: each one ends with ``_settle`` waiting
        for the servos to stop, so a subdivided curve becomes a staircase of
        pauses. Filming a 16-point arc made that unmissable -- 2.5s of commanded
        sweep took 26s of simulated time, stopping dead 16 times.

        ``rotations`` may be one 3x3 held for the path, or one per point, which
        is what a scooping wrist needs: the roll has to happen *during* the
        sweep, not between two stops.

        This is the sim's half of :meth:`BaseSkill.stream_pose_path`; the skills
        call that and get whichever exists.
        """
        pts = [np.asarray(p, float) for p in points]
        if len(pts) < 2:
            return False
        mats = rotations if rotations is not None else self._tcp()[1]
        mats = ([np.asarray(m, float) for m in mats]
                if isinstance(mats, (list, tuple)) or getattr(mats, "ndim", 2) == 3
                else [np.asarray(mats, float)] * len(pts))
        if len(mats) != len(pts):
            mats = [mats[0]] * len(pts)

        start_pos, start_mat = self._tcp()
        self.commands.append({"path": len(pts), "duration": float(duration)})
        if self.verbose:
            print(f"[sim] follow_pose_path: {len(pts)} points ({duration:.1f}s)")

        # One joint plan through every point, solved before any of it runs, so
        # the motion never pauses to think.
        seed = self.get_joints()
        q_start = seed.copy()
        plan = []
        per_point = max(int(duration * 20 / max(len(pts) - 1, 1)), 2)
        prev_pos, prev_mat = start_pos, start_mat
        for pos, mat in zip(pts, mats):
            for k in range(1, per_point + 1):
                frac = k / per_point
                seed = self._solve_ik(prev_pos + (pos - prev_pos) * frac,
                                      _slerp(prev_mat, mat, frac), seed)
                plan.append(seed.copy())
            prev_pos, prev_mat = pos, mat
        plan = np.asarray(plan)

        # A smooth reference plus velocity feed-forward, so the arm is where the
        # path says WHEN it says. A path is a shape the skill designed point by
        # point -- scoop's holds the spoon tip at a set height while the wrist
        # rolls -- and a servo trailing it by 0.1 s does not just arrive late:
        # the wrist lags the arm, the tool tilts more than planned, and the
        # shape breaks. From a checkpoint of the scoop roll the tip went 6.7 mm
        # through the cup floor at --speed 5 without this, and stayed within
        # 0.3 mm of plan with it at speeds 2, 3 and 5.
        ref = _hermite_plan(q_start, plan)
        span = max(duration / max(self.speed, 1e-6), self.model.opt.timestep)

        def ctrl(t):
            if t >= 1.0:
                self.data.ctrl[:7] = plan[-1]
                return
            # Min-jerk timing along the path: it starts and ends with zero
            # velocity AND acceleration. With uniform timing the path runs at
            # full speed until its last sample and then stops dead -- the
            # scoop's wrist was still turning at 1.4 rad/s one sample (28 ms)
            # before the end -- and the feed-forward turns that stop into a
            # command step the servos can only answer at their torque limits.
            # At --speed 2 that braking threw the TCP 26 mm sideways into the
            # dish rim; smooth timing is also what frankapy's generators do.
            t = max(t, 0.0)
            u = t ** 3 * (10.0 - 15.0 * t + 6.0 * t ** 2)
            du_dt = 30.0 * t ** 2 * (1.0 - t) ** 2
            q, dq_du = ref(u)
            self.data.ctrl[:7] = q + self._ff_gain * dq_du * du_dt / span

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
        deferred: List[str] = []
        if opening and self._attached:
            # Hand collisions come back only after the jaws are clear; see
            # _release. Restoring them here would jam this very open.
            deferred = self._release(defer_collision=True)

        self.data.ctrl[7] = width * CTRL_PER_METRE
        self._run(kwargs.get("duration", 1.0), lambda t: None)
        for name in deferred:
            self._restore_to_hand(name)

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
            # A fixture is bolted down: there is no freejoint to drive it by,
            # and closing on one should read as "the jaws shut on nothing", not
            # as picking the bench up. The stirrer's head sits directly above
            # its holder, so without this the holder wins the jaw-box test.
            if prop.name not in self.scene.prop_qpos:
                continue
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
            self._exclude_from_hand(prop.name)

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

    def _exclude_from_hand(self, name: str):
        """While held, the prop collides with everything but the hand."""
        body = self.scene.prop_bodies[name]
        saved = []
        for g in range(self.model.ngeom):
            if self.model.geom_bodyid[g] != body:
                continue
            ct, ca = int(self.model.geom_contype[g]), int(self.model.geom_conaffinity[g])
            if not (ct or ca):
                continue          # drawn only, never collides
            saved.append((g, ct, ca))
            self.model.geom_contype[g] = 4
            self.model.geom_conaffinity[g] = 1
        self._held_bits[name] = saved

    def _restore_to_hand(self, name: str):
        for g, ct, ca in self._held_bits.pop(name, []):
            self.model.geom_contype[g] = ct
            self.model.geom_conaffinity[g] = ca

    def _release(self, defer_collision: bool = False) -> List[str]:
        """Let go of whatever is in the jaws.

        A held prop is teleported to follow the TCP with its hand collisions
        switched off, so by the time it is let go its geoms can overlap the
        finger pads -- 13.5mm deep when the stirrer goes back in its holder.
        Turning those contacts back on before the jaws move makes MuJoCo
        resolve that overlap in one step, and the recoil jams the fingers:
        an 80mm open stalled at 52.6mm and the place skill read it as
        "failed to open". With ``defer_collision`` the prop is detached but
        stays uncollidable with the hand, and the caller restores it once the
        jaws are clear. The prop still keeps conaffinity 1 meanwhile, so it
        goes on resting on the holder and the table and cannot fall through.

        Returns the names whose hand collisions are still switched off.
        """
        deferred = []
        for name in list(self._attached):
            if self.verbose:
                print(f"[sim] released {name!r}")
            if defer_collision:
                deferred.append(name)
            else:
                self._restore_to_hand(name)
        self._attached.clear()
        self._held_width = None
        return deferred

    @property
    def holding(self) -> Optional[str]:
        """Name of the prop currently in the jaws, or None."""
        return next(iter(self._attached), None)

    # ----------------------------------------------------------------- admin

    def reset_scene(self):
        """Put the bench and the arm back to their starting state."""
        for name in list(self._attached):
            self._restore_to_hand(name)
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
