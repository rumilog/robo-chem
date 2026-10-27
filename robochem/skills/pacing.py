"""
One timing for every motion, on the robot and in the sim alike.

The sim used to move the way the robot cannot: a ``goto_pose`` ran at constant
speed and stopped dead, a ``reset_joints`` took 4 s at constant speed. frankapy
does neither. Its ``goto_pose`` and ``goto_joints`` are min-jerk, starting and
ending at rest, so over the same duration the middle runs 1.875x faster, and
its ``reset_joints`` takes 5 s that way. The first real pipeline runs
(2026-10-07/08) therefore looked faster than the sim on every lift and every
return home, and the user asked for the robot to move exactly as the sim does.

Both now use min-jerk (``SimFrankaArm`` too), and every duration the skills ask
for is stretched by :data:`MOTION_TIME_SCALE`, so the fastest point of a move
is as fast as the old sim's constant speed and no faster. A trip home takes
:data:`HOME_SECONDS`, which is the old sim's 4 s at constant speed.

Streamed paths (frankapy dynamic mode, ``SimFrankaArm.follow_pose_path``) were
already min-jerk in the sim and are timed the same way on the robot by
``BaseSkill.stream_pose_path``; their durations are not stretched.

On top of that pace, each skill call can run at its own **speed**, 0-100,
which the skill-call planner chooses (the user, 2026-10-27: liquid slow as it
is now, an empty spoon much faster). 0 is the pace above, unchanged, and
the slowest allowed (the user: nothing slower than now); higher is faster,
down to 0.4x the time at 100. It stretches or shortens every motion of the call, streamed paths
included (``BaseSkill.stream_pose_path``). Holds and waits are not motion and
keep their seconds.
"""

import math

#: min-jerk peaks at 1.875x its average speed (15/8): a move given this much
#: longer peaks at the constant speed the old sim showed.
MOTION_TIME_SCALE = 1.875

#: reset_joints: the old sim's 4 s at constant speed, peak-for-peak.
HOME_SECONDS = 4.0 * MOTION_TIME_SCALE

#: The speed every skill was tuned and validated at: exactly the pace above,
#: and the slowest the scale goes.
DEFAULT_SPEED = 0

#: (speed, time multiplier) on the pace above; between anchors the multiplier
#: is interpolated geometrically, so each step of 10 changes speed by the same
#: ratio (about 1.1x). 100 is 2.5x the validated pace: a 3 s goto_pose then takes 2.25 s
#: and the trip home 3 s, a little faster than frankapy's own defaults.
SPEED_ANCHORS = ((0, 1.0), (100, 0.4))

#: A faster speed never shortens a motion below this. Short moves (stir's 1 mm
#: floor probes, a twist's quarter cycle) are already brisk at the validated
#: pace; shortening them further only asks for a velocity discontinuity.
MIN_FAST_SECONDS = 1.0


def speed_time_factor(speed) -> float:
    """The share of the validated time each motion takes at ``speed``."""
    s = min(100.0, max(0.0, float(speed)))
    for (s0, f0), (s1, f1) in zip(SPEED_ANCHORS, SPEED_ANCHORS[1:]):
        if s <= s1:
            u = (s - s0) / (s1 - s0)
            return math.exp((1.0 - u) * math.log(f0) + u * math.log(f1))
    return SPEED_ANCHORS[-1][1]


def parse_speed(value):
    """``value`` as a speed in [0, 100], or None when it is not a number."""
    if value is None or isinstance(value, bool):
        return None
    try:
        s = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(s):
        return None
    return min(100.0, max(0.0, s))


class PacedArm:
    """
    The arm, with every point-to-point duration stretched to the shared pace.

    Wraps a frankapy ``FrankaArm`` or a ``SimFrankaArm``; everything else
    passes straight through. Dynamic (streamed) ``goto_pose`` is left alone:
    its timing is the stream's, which ``motion_seconds`` sets.

    ``speed`` (0-100, :data:`DEFAULT_SPEED` the pace above) is set per skill
    call by ``SkillsExecutor.execute``.
    """

    def __init__(self, arm, scale: float = MOTION_TIME_SCALE,
                 home_seconds: float = HOME_SECONDS):
        self._arm = arm
        self.scale = float(scale)
        self.home_seconds = float(home_seconds)
        self.speed = DEFAULT_SPEED

    def motion_seconds(self, seconds: float) -> float:
        """``seconds`` of motion at the validated pace, at the current speed."""
        s = float(seconds)
        factor = speed_time_factor(self.speed)
        if factor >= 1.0:
            return s * factor
        return max(s * factor, min(s, MIN_FAST_SECONDS))

    @property
    def unpaced(self):
        """The arm underneath."""
        return self._arm

    def goto_pose(self, tool_pose, *args, **kwargs):
        if kwargs.get("dynamic"):
            return self._arm.goto_pose(tool_pose, *args, **kwargs)
        if args:                       # duration given positionally
            args = (self.motion_seconds(float(args[0]) * self.scale),) + tuple(args[1:])
        else:
            kwargs["duration"] = self.motion_seconds(
                float(kwargs.get("duration", 3.0)) * self.scale)
        return self._arm.goto_pose(tool_pose, *args, **kwargs)

    def goto_joints(self, joints, *args, **kwargs):
        if args:
            args = (self.motion_seconds(float(args[0]) * self.scale),) + tuple(args[1:])
        else:
            kwargs["duration"] = self.motion_seconds(
                float(kwargs.get("duration", 5.0)) * self.scale)
        return self._arm.goto_joints(joints, *args, **kwargs)

    def reset_joints(self, *args, **kwargs):
        if args:
            args = (self.motion_seconds(args[0]),) + tuple(args[1:])
        else:
            kwargs["duration"] = self.motion_seconds(
                kwargs.get("duration", self.home_seconds))
        return self._arm.reset_joints(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._arm, name)


def paced(arm):
    """``arm`` wrapped once in :class:`PacedArm` (None stays None)."""
    if arm is None or isinstance(arm, PacedArm):
        return arm
    return PacedArm(arm)
