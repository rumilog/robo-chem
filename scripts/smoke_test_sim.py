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

import mujoco
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

    cup = cell.scene.bench.find("citric acid")
    cell.vision.clear_cache()
    # The printed scoop is CRANKED: the bowl sits forward of and below the
    # jaws, so a bare tool_length aims the arm at the wrong place entirely --
    # with tool_length=0.08 this test collected 0 granules out of 90 and still
    # passed. tool_span and tool_back_reach are what the stroke checks its
    # clearances against; without them it only protects a point.
    scooped, result = cell.skills.execute(
        "scoop", {"powder_source": "citric acid",
                  "tool_offset": [0.025, 0.0, 0.029],
                  "tool_span": 0.0275,
                  "tool_back_reach": 0.030,
                  "bowl_length": 0.0275,
                  "bowl_depth": 0.0115,
                  "bowl_width": 0.0205,
                  # The stroke is sized to the dish, so give it the real inside
                  # radius: perception's "opening radius" reads the outside of
                  # the rim plus its centre error.
                  "container_radius": cup.radius - cup.wall,
                  # ...and its real centre: perception's leans ~16 mm toward the
                  # cameras on the 90 mm dish, which puts the bowl in the wall.
                  "container_center": list(cell.vision.ground_truth("citric acid")[:2])}
    )
    ok &= check("scoop reports success", scooped)
    if scooped:
        # The stroke deliberately finishes NOSE-UP, cupping the load. Coming
        # back level while still over the cup is what tips the powder straight
        # back in, so "did it level" is the wrong question -- negative here
        # means tipped up, which is what carries the scoop out loaded.
        cupped = float(result.get("residual_from_level_deg", 0.0))
        ok &= check("spoon finished cupped (nose-up)", cupped < -5.0,
                    f"{cupped:+.1f} deg from level")
    return ok


def test_scoop_and_dump(cell) -> bool:
    """Scoop, then empty it into the paper cup from just above the rim."""
    print("\npick_up + scoop + dump (larger spoon, citric acid -> white paper cup)")
    cell.reset()
    ok = True

    picked, _ = cell.skills.execute(
        "pick_up", {"object_name": "larger spoon", "z_offset": 0.0, "grasp_force": 1.0}
    )
    ok &= check("spoon picked up", picked and cell.arm.holding == "larger spoon",
                str(cell.arm.holding))
    cup = cell.scene.bench.find("citric acid")
    tool = {"tool_offset": [0.025, 0.0, 0.029], "bowl_length": 0.0275,
            "bowl_depth": 0.0115, "bowl_width": 0.0205}
    cell.vision.clear_cache()
    scooped, _ = cell.skills.execute(
        "scoop", {"powder_source": "citric acid", **tool,
                  "tool_span": 0.0275, "tool_back_reach": 0.030,
                  "container_radius": cup.radius - cup.wall,
                  "container_center": list(cell.vision.ground_truth("citric acid")[:2])}
    )
    ok &= check("scoop reports success", scooped)
    if not scooped:
        return ok

    dumped, result = cell.skills.execute(
        "dump", {"target_container": "white paper cup", **tool})
    ok &= check("dump reports success", dumped, str(result.get("error", ""))[:70])
    if dumped:
        tip = float(result.get("tip_achieved", 0.0))
        ok &= check("tipped to about vertical", tip > 85.0, f"{tip:.1f} deg")
        # The whole point of the rewrite: the bowl stays over the target
        # while it tips, instead of wandering 80 mm toward the base.
        drift = max(result.get("bowl_drift_mm", {}).values(), default=1e9)
        ok &= check("bowl stayed over the cup", drift < 5.0, f"{drift:.1f} mm at worst")
        residual = abs(float(result.get("residual_tilt", 90.0)))
        ok &= check("came back level", residual < 5.0, f"{residual:.1f} deg")
    return ok


def test_scoop_then_stir(cell) -> bool:
    """Scoop and stir back to back in ONE session, with no reset between them.

    Every other group calls cell.reset() first, which wipes exactly the state
    that one skill hands the next: what the jaws are holding, where the arm
    was left, which geoms had their collisions switched off while a prop was
    carried. The gripper bug this suite caught after the bozhang merge only
    appeared when a carried prop was let go, so the spoon is used and put down
    here before the stirrer is picked up -- the handover is the thing under
    test, not the two skills on their own.
    """
    print("\nscoop then stir (no reset between: spoon -> put down -> stirrer)")
    cell.reset()
    ok = True
    tool = {"tool_offset": [0.025, 0.0, 0.029], "bowl_length": 0.0275,
            "bowl_depth": 0.0115, "bowl_width": 0.0205}

    # ---- his scoop -------------------------------------------------------
    picked, _ = cell.skills.execute(
        "pick_up", {"object_name": "larger spoon", "z_offset": 0.0, "grasp_force": 1.0}
    )
    ok &= check("spoon picked up", picked and cell.arm.holding == "larger spoon",
                str(cell.arm.holding))
    if not picked:
        return ok

    cup = cell.scene.bench.find("citric acid")
    cell.vision.clear_cache()
    scooped, _ = cell.skills.execute(
        "scoop", {"powder_source": "citric acid", **tool,
                  "tool_span": 0.0275, "tool_back_reach": 0.030,
                  "container_radius": cup.radius - cup.wall,
                  "container_center": list(cell.vision.ground_truth("citric acid")[:2])}
    )
    ok &= check("scoop reports success", scooped)
    if not scooped:
        return ok

    dumped, _ = cell.skills.execute(
        "dump", {"target_container": "white paper cup", **tool})
    ok &= check("dump reports success", dumped)

    # ---- the handover ----------------------------------------------------
    # Letting the spoon go is the step that used to jam: a carried prop can
    # overlap the finger pads, and turning its hand collisions back on before
    # the jaws move stalls the open.
    ok &= check("spoon released cleanly", cell.arm.open_gripper() is not False)
    ok &= check("jaws opened past 56mm", cell.arm.get_gripper_width() > 0.056,
                f"{cell.arm.get_gripper_width() * 1000:.1f}mm")
    ok &= check("nothing left in the jaws", cell.arm.holding is None,
                str(cell.arm.holding))

    # ---- your stir -------------------------------------------------------
    body = cell.scene.prop_bodies["stirrer"]
    seat = np.array([*cell.scene.bench.find("stirrer holder").pos, 0.100])
    cell.vision.clear_cache()
    picked, _ = cell.skills.execute("pick_up", {"object_name": "stirring rod"})
    ok &= check("stirrer picked up after the scoop",
                picked and cell.arm.holding == "stirrer", str(cell.arm.holding))
    if not picked:
        return ok

    cell.vision.clear_cache()
    stirred, result = cell.skills.execute("stir", {
        "target_container": "plastic beaker",
        "tool_axis": "z",
        "tool_length": 0.081,
        "revolutions": 2,
    })
    ok &= check("stir reports success", stirred, "" if stirred else str(result)[:70])
    if stirred:
        ok &= check("two revolutions completed",
                    result.get("revolutions_completed") == 2.0,
                    str(result.get("revolutions_completed")))

    cell.vision.clear_cache()
    placed, result = cell.skills.execute("place", {
        "target_location": "stirrer holder",
        "on_top": True,
        "release_clearance": 0.015,
        "stop_force_n": 1.0,
    })
    ok &= check("stirrer placed back", placed, "" if placed else str(result)[:70])
    for _ in range(1500):
        mujoco.mj_step(cell.scene.model, cell.scene.data)
    back = cell.scene.data.xpos[body].copy()
    ok &= check("stirrer back in the holder", np.linalg.norm(back - seat) < 0.008,
                f"{np.round(back, 4)} vs seat {np.round(seat, 3)}")
    return ok


def test_stirrer_and_stir(cell) -> bool:
    """
    The stirrer's whole cycle: out of its holder, stir, back into its holder.

    The return is the interesting half. The rod is 66mm long and the bore is
    only 2.5mm wider than it, so the insertion is blind and depends on the
    holder's chamfer catching a rod that arrives off axis -- which is exactly
    what would bite on hardware.
    """
    print("\npick_up + stir + place (stirrer <-> its holder)")
    cell.reset()
    ok = True
    body = cell.scene.prop_bodies["stirrer"]
    seat = np.array([*cell.scene.bench.find("stirrer holder").pos, 0.100])

    def pose():
        R = cell.scene.data.xmat[body].reshape(3, 3)
        return (cell.scene.data.xpos[body].copy(),
                float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1)))))

    start, tilt = pose()
    ok &= check("starts seated in the holder, upright",
                np.linalg.norm(start - seat) < 0.006 and tilt < 2.0,
                f"{np.round(start, 4)}, {tilt:.2f} deg")

    picked, _ = cell.skills.execute("pick_up", {"object_name": "stirring rod"})
    ok &= check("lifted out by its head",
                picked and cell.arm.holding == "stirrer", str(cell.arm.holding))
    ok &= check("gripper closed on the 30mm head",
                abs(cell.arm.get_gripper_width() - 0.030) < 0.002,
                f"{cell.arm.get_gripper_width() * 1000:.1f}mm")
    _, tilt = pose()
    ok &= check("rod hangs vertical after the grasp", tilt < 5.0, f"{tilt:.2f} deg")

    cell.vision.clear_cache()
    stirred, result = cell.skills.execute("stir", {
        "target_container": "plastic beaker",
        "tool_axis": "z",          # gripped end-on, already vertical
        "tool_length": 0.081,      # head centre -> rod tip
        "revolutions": 2,
    })
    ok &= check("stir reports success", stirred, "" if stirred else str(result)[:70])
    if stirred:
        ok &= check("two revolutions completed",
                    result.get("revolutions_completed") == 2.0,
                    str(result.get("revolutions_completed")))
        ok &= check("circle fits the measured rim",
                    result["stir_radius"] <= result["rim_radius"] - 0.010,
                    f"r={result['stir_radius']*1000:.0f}mm in a "
                    f"{result['rim_radius']*1000:.0f}mm opening")

    cell.vision.clear_cache()
    placed, result = cell.skills.execute("place", {
        "target_location": "stirrer holder",
        "on_top": True,            # into the bore, not down beside it
        "release_clearance": 0.015,
        "stop_force_n": 1.0,       # feel for the rim rather than trusting the scan
    })
    ok &= check("place reports success", placed, "" if placed else str(result)[:70])
    if placed:
        ok &= check("stopped on contact, not on the computed height",
                    result.get("seated_by_force") is True,
                    f"{result.get('contact_force_n')} N at z={result.get('contact_z')}")

    for _ in range(1500):          # let it drop the last millimetre and settle
        mujoco.mj_step(cell.scene.model, cell.scene.data)
    back, tilt = pose()
    ok &= check("back in the holder", np.linalg.norm(back - seat) < 0.008,
                f"{np.round(back, 4)} vs seat {np.round(seat, 3)}")
    ok &= check("standing upright again, not dropped on the bench",
                tilt < 5.0, f"{tilt:.2f} deg")
    ok &= check("gripper is empty", cell.arm.holding is None, str(cell.arm.holding))
    return ok


def test_force_guard_refuses_to_drop_into_nothing(cell) -> bool:
    """
    A force-guarded place must keep hold when it never feels anything.

    Probing over open bench and releasing anyway would drop the tool from
    whatever height the scan happened to produce, which is the failure the
    guard exists to prevent.
    """
    print("\nforce guard")
    cell.reset()
    picked, _ = cell.skills.execute("pick_up", {"object_name": "stirring rod"})
    if not picked:
        return check("could pick the stirrer up to test the guard", False)

    # Probe in clear air over the bench. A 3-element target names the surface
    # to release above, and 0.30 is high enough that the stirrer's 81mm of rod
    # never reaches the table -- so the probe genuinely feels nothing.
    cell.vision.clear_cache()
    placed, result = cell.skills.execute("place", {
        "target_location": [0.48, -0.20, 0.30],
        "stop_force_n": 1.0,
        "probe_below": 0.010,
    })
    ok = check("refuses to release having felt nothing", not placed,
               str(result.get("error", ""))[:60])
    ok &= check("still holding the stirrer", cell.arm.holding == "stirrer",
                str(cell.arm.holding))
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
    parser.add_argument("--only", default=None,
                        help="Run just the groups whose name contains this; "
                             "comma-separated for several, "
                             "e.g. --only scoop,stir")
    parser.add_argument("--hold", type=float, default=5.0,
                        help="Seconds to keep the viewer open at the end; "
                             "0 waits until you close the window")
    args = parser.parse_args()

    cell = build_cell(
        viewer=args.viewer,
        realtime=args.viewer,
        speed=args.speed,
        granules=args.granules,
        verbose=False,
        workspace_min=[0.25, -0.40, -0.13],
    )

    groups = {
        "perception": test_perception,
        "motion": test_reach,
        "pick + pour": test_pick_and_pour,
        "pick + scoop": test_spoon_and_scoop,
        "pick + stir": test_stirrer_and_stir,
        "force guard": test_force_guard_refuses_to_drop_into_nothing,
        "pick + scoop + dump": test_scoop_and_dump,
        "scoop then stir": test_scoop_then_stir,
        "failure handling": test_empty_gripper_refuses_to_scoop,
    }
    if args.only:
        terms = [t.strip().lower() for t in args.only.split(",") if t.strip()]
        wanted = {k: v for k, v in groups.items()
                  if any(t in k for t in terms)}
        if not wanted:
            print(f"--only {args.only!r} matches none of: "
                  + ", ".join(groups))
            cell.close()
            return 2
        groups = wanted

    try:
        results = {name: test(cell) for name, test in groups.items()}
    finally:
        if args.viewer:
            cell.arm.hold(args.hold if args.hold > 0 else 1e9)
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
