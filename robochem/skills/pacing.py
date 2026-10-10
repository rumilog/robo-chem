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
"""

#: min-jerk peaks at 1.875x its average speed (15/8): a move given this much
#: longer peaks at the constant speed the old sim showed.
MOTION_TIME_SCALE = 1.875

#: reset_joints: the old sim's 4 s at constant speed, peak-for-peak.
HOME_SECONDS = 4.0 * MOTION_TIME_SCALE


class PacedArm:
    """
    The arm, with every point-to-point duration stretched to the shared pace.

    Wraps a frankapy ``FrankaArm`` or a ``SimFrankaArm``; everything else
    passes straight through. Dynamic (streamed) ``goto_pose`` is left alone:
    its timing is the stream's.
    """

    def __init__(self, arm, scale: float = MOTION_TIME_SCALE,
                 home_seconds: float = HOME_SECONDS):
        self._arm = arm
        self.scale = float(scale)
        self.home_seconds = float(home_seconds)

    @property
    def unpaced(self):
        """The arm underneath."""
        return self._arm

    def goto_pose(self, tool_pose, *args, **kwargs):
        if kwargs.get("dynamic"):
            return self._arm.goto_pose(tool_pose, *args, **kwargs)
        if args:                       # duration given positionally
            args = (float(args[0]) * self.scale,) + tuple(args[1:])
        else:
            kwargs["duration"] = float(kwargs.get("duration", 3.0)) * self.scale
        return self._arm.goto_pose(tool_pose, *args, **kwargs)

    def goto_joints(self, joints, *args, **kwargs):
        if args:
            args = (float(args[0]) * self.scale,) + tuple(args[1:])
        else:
            kwargs["duration"] = float(kwargs.get("duration", 5.0)) * self.scale
        return self._arm.goto_joints(joints, *args, **kwargs)

    def reset_joints(self, *args, **kwargs):
        if not args and "duration" not in kwargs:
            kwargs["duration"] = self.home_seconds
        return self._arm.reset_joints(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._arm, name)


def paced(arm):
    """``arm`` wrapped once in :class:`PacedArm` (None stays None)."""
    if arm is None or isinstance(arm, PacedArm):
        return arm
    return PacedArm(arm)
