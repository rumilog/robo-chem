"""
End-to-end check of the MuJoCo cell -- no robot, no cameras, no ROS.

``smoke_test_motion.py`` and ``smoke_test_skills.py`` need the arm and the cage.
``test_skills_offline.py`` needs neither but drives the skills against a fake arm
whose poses are assigned rather than reached. This one sits in between: the real
skills run against the simulated Panda, so the motions are actually solved and
executed and perception actually reconstructs the props from four rendered
views before each grasp.

Run it headless (the default) in CI, or with --viewer to watch:

    perception_env/bin/python scripts/smoke_test_sim.py
    perception_env/bin/python scripts/smoke_test_sim.py --viewer --speed 2
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robochem.sim import build_cell

# How far a reconstructed centroid may sit from the prop's true position. The
# cage calibrates to a 3-5 mm residual and each camera only sees the near wall,
# so a couple of centimetres is the honest bar, not a millimetre.
CENTROID_TOL = 0.03


def check(label: str, passed: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}" + (f" -- {detail}" if detail else ""))
    return passed


def test_perception(cell) -> bool:
    """Every prop reconstructs, from more than one view, near where it really is."""
    print("\nperception")
    ok = True
    for name in ("plastic beaker", "white paper cup", "citric acid", "larger spoon"):
        located = cell.vision.locate(name, force_refresh=True)
        if located is None:
            ok &= check(f"locate {name!r}", False, "not found")
            continue
        truth = cell.vision.ground_truth(name)
        err = float(np.linalg.norm(located["centroid"][:2] - truth[:2]))
        ok &= check(
            f"locate {name!r}",
            err < CENTROID_TOL and len(located["cameras"]) >= 2,
            f"xy err {err * 1000:.1f} mm from {len(located['cameras'])} cameras",
        )
    return ok


def test_reach(cell) -> bool:
    """Commanded Cartesian targets are actually reached, not merely recorded."""
    print("\nmotion")
    top_down = np.diag([1.0, -1.0, -1.0])
    ok = True
    for target in ([0.46, 0.14, 0.25], [0.58, -0.02, 0.12], [0.40, -0.26, 0.30]):
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = top_down, target
        cell.arm.goto_pose(pose, duration=2.0)
        err = float(np.linalg.norm(cell.arm.get_pose().translation - np.array(target)))
        ok &= check(f"reach {target}", err < 0.005, f"{err * 1000:.2f} mm")
    cell.arm.reset_joints()
    return ok


def test_pick_and_pour(cell) -> bool:
    """The README's validated beaker commands, start to finish."""
    print("\npick_up + pour (plastic beaker -> white paper cup)")
    cell.reset()
    ok = True

    picked, _ = cell.skills.execute(
        "pick_up", {"object_name": "plastic beaker", "z_offset": 0.02, "squeeze": 0.013}
    )
    ok &= check("pick_up reports success", picked)
    ok &= check("gripper is holding the beaker",
                cell.arm.holding == "plastic beaker", str(cell.arm.holding))

    beaker_z = cell.scene.data.xpos[cell.scene.prop_bodies["plastic beaker"]][2]
    ok &= check("beaker left the table", beaker_z > 0.12, f"z={beaker_z:.3f}")

    cell.vision.clear_cache()
    poured, result = cell.skills.execute(
        "pour", {"target_container": "white paper cup", "pour_angle": 90,
                 "hold_duration": 1, "forward_offset": -0.08}
    )
    ok &= check("pour reports success", poured)
    if poured:
        tip = float(result.get("tip_after", 0) or 0)
        ok &= check("tipped to roughly 90 degrees", tip > 80, f"{tip:.1f} deg")
    return ok


def test_spoon_and_scoop(cell) -> bool:
    """The README's validated spoon commands, start to finish."""
    print("\npick_up + scoop (larger spoon -> citric acid)")
    cell.reset()
    ok = True

    picked, _ = cell.skills.execute(
        "pick_up", {"object_name": "larger spoon", "z_offset": 0.0, "grasp_force": 1.0}
    )
    ok &= check("spoon picked up", picked and cell.arm.holding == "larger spoon",
                str(cell.arm.holding))

    cell.vision.clear_cache()
    scooped, result = cell.skills.execute(
        "scoop", {"powder_source": "citric acid", "tool_length": 0.08}
    )
    ok &= check("scoop reports success", scooped)
    if scooped:
        residual = float(result.get("residual_tilt", 99))
        ok &= check("spoon came back level", residual < 5.0, f"{residual:.2f} deg")
    return ok


def test_empty_gripper_refuses_to_scoop(cell) -> bool:
    """Scooping with nothing in the jaws must fail, not mime the motion."""
    print("\nfailure handling")
    cell.reset()
    scooped, result = cell.skills.execute(
        "scoop", {"powder_source": "citric acid", "tool_length": 0.08}
    )
    return check("scoop refuses an empty gripper", not scooped, str(result)[:70])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--viewer", action="store_true", help="Watch it run")
    parser.add_argument("--speed", type=float, default=6.0,
                        help="Playback multiplier (default 6x)")
    parser.add_argument("--granules", action="store_true",
                        help="Loose particles in the reagent cups")
    args = parser.parse_args()

    cell = build_cell(
        viewer=args.viewer,
        realtime=args.viewer,
        speed=args.speed,
        granules=args.granules,
        verbose=False,
        workspace_min=[0.25, -0.40, -0.13],
    )

    try:
        results = {
            "perception": test_perception(cell),
            "motion": test_reach(cell),
            "pick + pour": test_pick_and_pour(cell),
            "pick + scoop": test_spoon_and_scoop(cell),
            "failure handling": test_empty_gripper_refuses_to_scoop(cell),
        }
    finally:
        if args.viewer:
            cell.arm.hold(5)
        cell.close()

    print("\n" + "=" * 52)
    for name, passed in results.items():
        print(f"  {'PASS' if passed else 'FAIL'}  {name}")
    failed = [n for n, p in results.items() if not p]
    print("=" * 52)
    print("all groups passed" if not failed else f"FAILED: {', '.join(failed)}")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
