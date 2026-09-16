"""
RigidTransform for the simulator.

``base_skill`` builds every pose as an ``autolab_core.RigidTransform`` because
frankapy validates ``from_frame``/``to_frame`` and raises otherwise. autolab_core
is installed only in the frankapy Python 3.8 venv, and the simulator runs in
perception_env (3.10) so it can import mujoco. Rather than duplicate the robot
env, supply the same shim ``scripts/test_skills_offline.py`` already uses and
patch it into ``base_skill`` when the real class is missing.

Importing this module is what makes ``robochem.skills`` usable off the robot PC.
"""

from __future__ import annotations

import numpy as np

try:  # pragma: no cover - depends on which venv is active
    from autolab_core import RigidTransform as _RigidTransform

    IS_SHIM = False
except ImportError:  # pragma: no cover
    _RigidTransform = None
    IS_SHIM = True


class ShimRigidTransform:
    """
    Minimal stand-in for ``autolab_core.RigidTransform``.

    Only the attributes ``base_skill`` and the skills actually touch: the
    rotation/translation pair, the two frame names frankapy validates, and
    ``copy()``.
    """

    def __init__(self, rotation=None, translation=None, from_frame="", to_frame=""):
        self.rotation = np.eye(3) if rotation is None else np.asarray(rotation, float)
        self.translation = (np.zeros(3) if translation is None
                            else np.asarray(translation, dtype=float))
        self.from_frame = from_frame
        self.to_frame = to_frame

    @property
    def matrix(self) -> np.ndarray:
        out = np.eye(4)
        out[:3, :3] = self.rotation
        out[:3, 3] = self.translation
        return out

    @property
    def quaternion(self) -> np.ndarray:
        """wxyz, matching autolab_core's ordering."""
        import mujoco

        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, np.asarray(self.rotation, float).reshape(9))
        return quat

    def copy(self) -> "ShimRigidTransform":
        return ShimRigidTransform(self.rotation.copy(), self.translation.copy(),
                                  self.from_frame, self.to_frame)

    def __repr__(self) -> str:
        return (f"ShimRigidTransform(t={np.round(self.translation, 4).tolist()}, "
                f"{self.from_frame}->{self.to_frame})")


RigidTransform = _RigidTransform if _RigidTransform is not None else ShimRigidTransform


def install() -> type:
    """
    Make ``robochem.skills.base_skill`` able to build poses in this environment.

    ``base_skill`` sets ``RigidTransform = None`` when autolab_core is absent and
    then refuses to convert poses. Point it at whichever class this module
    resolved so the skills run unchanged under the simulator.

    Returns:
        The RigidTransform class now in effect.
    """
    from robochem.skills import base_skill

    if base_skill.RigidTransform is None:
        base_skill.RigidTransform = RigidTransform
    return base_skill.RigidTransform
