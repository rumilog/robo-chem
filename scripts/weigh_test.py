"""
Weighing test: put the scoop on the scale, read the scale from the cameras,
put the scoop back where it came from.

    source scripts/env.sh
    scripts/grounding.sh start
    python scripts/weigh_test.py                 # the whole test, arm moves
    python scripts/weigh_test.py --read-only     # just read the scale, no arm

Steps (everything lands in experiments/run_<time>/):
  1. home, locate the scale (SAM prompt --scale-prompt), keep its box per camera
  2. read the empty scale (baseline; it should say 0)
  3. pick_up the scoop (bench-validated grasp, as in the Magic Beaker runs)
  4. set it down on the scale's platform: TCP over the platform centre, shifted
     by how far the scoop's centroid sat from the jaws at the pick, and at the
     platform top + the TCP's height above the table at the pick + a few mm.
     NOT place on_top: that releases the TCP 20 mm over the top, and the bowl
     hangs below the handle, so it would be driven into the platform.
  5. home (arm out of the cameras' way), wait for the scale to settle, read
  6. pick_up the scoop off the scale, place it back at the FIRST pick's site
     (the second pick would otherwise become "where it was picked from")
  7. read once more (should be back to the baseline)

The reading is a model call per camera on a 1080p colour crop of the scale
(robochem/vision/scale_reader.py). Each camera's crop is saved under scale/.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_experiment import _Tee, build_stack  # noqa: E402
from robochem.vision import REAL_TABLE_Z  # noqa: E402
from robochem.vision.scale_reader import (locate_on_platform, platform_from_points,  # noqa: E402
                                          read_scale)


def locate_scale(vision, prompt):
    """(platform centre xy, platform top z, per-camera boxes, platform) or (None, None, {}, None)."""
    try:
        # use_labels=False: "digital scale" otherwise goes through the cup-label
        # reader first (105 s on 2026-11-11 before the direct prompt found it).
        located = vision.locate(prompt, force_refresh=True, use_labels=False)
    except Exception as exc:
        print(f"[Weigh] locating {prompt!r} failed: {type(exc).__name__}: {exc}")
        return None, None, {}, None
    if not located or located.get("points") is None or len(located["points"]) < 20:
        print(f"[Weigh] {prompt!r} not found")
        return None, None, {}, None
    pts = np.asarray(located["points"], dtype=float)
    # The platform is the dominant flat level on top (platform_from_points);
    # the display and base sit lower, anything lying on it higher.
    platform = platform_from_points(pts)
    top_z, centre = platform["top_z"], platform["centre"]
    lo, hi = np.percentile(pts, [2, 98], axis=0)
    scan = (getattr(vision, "scans", {}) or {}).get(vision._scan_key(prompt)) or {}
    boxes = dict(scan.get("boxes") or {})
    print(f"[Weigh] scale: platform centre ({centre[0]:.4f}, {centre[1]:+.4f}), "
          f"top z {top_z:.4f} ({(top_z - REAL_TABLE_Z) * 1000:.0f} mm over the table), "
          f"extent {np.round((hi - lo) * 1000).astype(int).tolist()} mm, "
          f"seen by cams {sorted(boxes)}")
    print(f"[Weigh] platform x {platform['xy_lo'][0]:.3f}..{platform['xy_hi'][0]:.3f}, "
          f"y {platform['xy_lo'][1]:+.3f}..{platform['xy_hi'][1]:+.3f}")
    return centre, top_z, boxes, platform


def regrasp(arm, pos, rot, force) -> bool:
    """
    Take the object back exactly where place let go of it: TCP to ``pos`` with
    the jaws at ``rot``, open on the way down, close, lift. Any miss leaves the
    jaws open above the scale and returns False.
    """
    pos = np.asarray(pos, dtype=float)
    hover = pos.copy()
    hover[2] = max(pos[2] + 0.15, arm.safe_height)
    above = pos.copy()
    above[2] += 0.03
    print(f"[Weigh] re-grasp where it was released: TCP {np.round(pos, 4).tolist()}")
    if not arm.open_gripper():
        print("[Weigh] re-grasp: gripper did not open")
        return False
    for name, target, seconds, tol in (("hover", hover, 4.0, 0.05),
                                       ("above", above, 3.0, 0.05),
                                       ("grasp", pos, 3.0, 0.025)):
        if not arm.goto_pose_rigid(target, rot, duration=seconds):
            print(f"[Weigh] re-grasp: {name} move failed")
            arm.goto_pose_rigid(hover, rot, duration=3.0)
            return False
        arrived, err = arm.reached(target, tol)
        print(f"[Weigh] re-grasp: {name} reached within {err * 1000:.1f} mm")
        if not arrived:
            arm.goto_pose_rigid(hover, rot, duration=3.0)
            return False
    if not arm.close_gripper(force=force, force_limited=True):
        print("[Weigh] re-grasp: close failed")
        arm.open_gripper()
        arm.goto_pose_rigid(hover, rot, duration=3.0)
        return False
    arm.wait(0.4)
    width = arm.held_width()
    print(f"[Weigh] re-grasp: holding at {width * 1000:.1f} mm")
    if width < 0.003:
        print("[Weigh] re-grasp: closed on nothing")
        arm.open_gripper()
        arm.goto_pose_rigid(hover, rot, duration=3.0)
        return False
    lift = pos.copy()
    lift[2] += 0.15
    arm.goto_pose_rigid(lift, rot, duration=3.0)
    if arm.held_width() < 0.003:
        print("[Weigh] re-grasp: lost it on the lift")
        return False
    return True


def pick_from_scale(args, vision, skills, platform, boxes, scale_dir, step, summary, save):
    """
    The object lies on the scale: pick it up and set it back where it lay.

    pick_up finds its object through vision.locate(). For this one call,
    locate() answers the object's name with what lies above the platform
    (locate_on_platform) instead of SAM, which takes the white scale and the
    white scoop as one object. Everything else is pick_up as validated
    (bench grasp defaults, yaw, lift); the put-back is place to the pick site
    with the slow force probe.
    """
    want = vision.canonical_name(args.object)
    original = vision.locate

    def locate(name, *a, **k):
        if vision.canonical_name(name) != want:
            return original(name, *a, **k)
        found = locate_on_platform(vision.object_localizer, platform, boxes,
                                   out_dir=scale_dir, tag="pick")
        summary.setdefault("finds", []).append(found)
        if found is None:
            return None
        pts = np.asarray(found["points"], dtype=float)
        return {"points": pts, "centroid": np.asarray(found["centroid"], dtype=float),
                "dimensions": vision.compute_dimensions(pts),
                "cameras": sorted(found["cameras"]), "rejected_cameras": found["rejected"]}

    vision.locate = locate
    try:
        ok, pick = step("pick_up", {"object_name": args.object})
        if not ok:
            ok, pick = step("pick_up", {"object_name": args.object})
    finally:
        vision.locate = original
    summary["pick"] = pick
    if not ok:
        summary["outcome"] = "pick from the scale failed"
        save()
        return 1

    place = {"target_location": args.object, "release_clearance": 0.02}
    if args.stop_force > 0:
        place.update({"stop_force_n": args.stop_force, "probe_above": 0.0,
                      "probe_below": args.probe_below, "probe_step": args.probe_step,
                      "probe_seconds": args.probe_seconds})
    ok, placed = step("place", place, speed=0)
    summary["place"] = placed
    summary["outcome"] = "done" if ok else "put-back failed"
    save()
    if ok:
        skills.go_home()
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--read-only", action="store_true",
                    help="Locate and read the scale only; the arm does not move")
    ap.add_argument("--pick-from-scale", action="store_true",
                    help="The object is ALREADY on the scale: find it by height, pick it "
                         "up (pick_up, with the found cloud), and set it back on the same "
                         "spot with the slow force probe. No scale reading")
    ap.add_argument("--find-on-scale", action="store_true",
                    help="Locate the scale, then find what lies on its platform by height; "
                         "the arm does not move, nothing is read")
    ap.add_argument("--object", default="scoop", help="What to weigh (bench name or alias)")
    ap.add_argument("--scale-prompt", default="digital scale")
    ap.add_argument("--read-cams", nargs="+", type=int, default=[5, 2, 3, 4],
                    help="Cameras to read the display from, in order of preference; "
                         "the first whose frames all agree ends the read (2026-11-13: "
                         "cam 5 reads it, cam 2 sometimes, cams 3/4 see it blank)")
    ap.add_argument("--all-cams", action="store_true",
                    help="Read every --read-cams camera even after one is unanimous")
    ap.add_argument("--frames", type=int, default=3,
                    help="Frames read per camera; a camera's value needs a majority")
    ap.add_argument("--decimals", type=int, default=1,
                    help="Digits after the point this scale shows (the point itself is "
                         "too small for the cameras); -1 to trust what the model sees")
    ap.add_argument("--speed", type=float, default=0,
                    help="Motion speed for every skill call, 0-100 (robochem.skills.pacing): "
                         "0 = the validated pace (the slowest), 100 = 0.4x the time. "
                         "Waits, the settle time and the force probe's steps do not speed up")
    ap.add_argument("--max-shift", type=float, default=0.04,
                    help="Trust the height-based find of the object on the scale only "
                         "within this distance (m) of where it was released")
    ap.add_argument("--regrasp-force", type=float, default=1.0,
                    help="Grip force (N) when taking it back off the scale; the scoop's "
                         "validated pick uses 1.0")
    ap.add_argument("--settle", type=float, default=3.0,
                    help="Seconds between setting the object down and reading")
    ap.add_argument("--release-clearance", type=float, default=0.008,
                    help="Gap (m) above the computed rest height where the force probe "
                         "starts (or, with --stop-force 0, where the jaws open)")
    ap.add_argument("--stop-force", type=float, default=2.5,
                    help="Feel for the platform: step down slowly and open the jaws "
                         "once the push changes by more than this (N). 2026-11-13: at "
                         "1.0 N a -1.01 N settling wobble 2 mm into the probe read as "
                         "contact and the spoon was dropped from above the platform. "
                         "0 = open at the computed height")
    ap.add_argument("--probe-step", type=float, default=0.001,
                    help="Probe step (m); 1 mm")
    ap.add_argument("--probe-seconds", type=float, default=1.5,
                    help="Seconds per probe step, the force read at its end (~0.7 mm/s)")
    ap.add_argument("--probe-below", type=float, default=0.04,
                    help="How far (m) under the computed height the probe may go")
    ap.add_argument("--place-offset", nargs=2, type=float, default=[0.03, 0.0],
                    metavar=("DX", "DY"),
                    help="Nudge (m) of the set-down spot; +X is away from the base. "
                         "Default +30 mm: at 0 the spoon landed too far back (2026-11-13)")
    ap.add_argument("--workspace-min", nargs=3, type=float, default=[0.25, -0.40, -0.13])
    ap.add_argument("--workspace-max", nargs=3, type=float, default=[0.75, 0.40, 0.70])
    ap.add_argument("--cameras", nargs="+", type=int, default=[2, 3, 4, 5])
    ap.add_argument("--bench", default="magic_beaker")
    ap.add_argument("--log-dir", default="experiments")
    args = ap.parse_args()

    run_dir = Path(args.log_dir) / f"run_{time.strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    sys.stdout = _Tee(sys.stdout, run_dir / "console.log")
    sys.stderr = _Tee(sys.stderr, run_dir / "console.log")
    print(f"[run] weigh test; everything goes in {run_dir}/")
    scale_dir = run_dir / "scale"

    import os
    from robochem.vision.bench_manifest import BENCHES
    stack_args = SimpleNamespace(
        cameras=args.cameras,
        grounding_url=os.environ.get("GROUNDING_URL", "http://127.0.0.1:5005"),
        consensus_tolerance=0.05, max_object_extent=0.35,
        bench=args.bench if args.bench in BENCHES else "none",
        rescan_always=False, dry_run=args.read_only or args.find_on_scale, reset=False,
        workspace_min=args.workspace_min, workspace_max=args.workspace_max,
    )
    _, vision, skills, robot = build_stack(stack_args)
    summary = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "args": vars(args)}

    def save():
        (run_dir / "weigh_summary.json").write_text(
            json.dumps(summary, indent=2, default=str), encoding="utf-8")

    def read(tag):
        print(f"\n--- read: {tag} ---")
        result = read_scale(args.read_cams, boxes, scale_dir, tag, frames=args.frames,
                            stop_when_unanimous=not args.all_cams,
                            decimals=None if args.decimals < 0 else args.decimals)
        print(f"[Weigh] {tag}: value={result['value']} {result['unit'] or ''} "
              f"({result['agree'] or 'no camera read digits'})")
        summary[tag] = result
        save()
        return result

    def step(name, params, speed=None):
        print(f"\n--- {name} {params} ---")
        started = time.time()
        ok, res = skills.execute(name, {**params, "speed": args.speed if speed is None else speed})
        print(f"[Weigh] {name}: {'ok' if ok else 'FAILED'} in {time.time() - started:.1f}s"
              + ("" if ok else f" -- {res.get('error')}"))
        return ok, res

    # 1. Locate the scale from home (arm out of the cage's view).
    if robot is not None:
        robot.open_gripper()
        skills.go_home()
    centre, top_z, boxes, platform = locate_scale(vision, args.scale_prompt)
    summary["scale"] = {"centre": centre, "top_z": top_z, "boxes": boxes}

    if args.pick_from_scale:
        if platform is None:
            return 1
        return pick_from_scale(args, vision, skills, platform, boxes, scale_dir, step,
                               summary, save)

    if args.find_on_scale:
        if platform is None:
            return 1
        found = locate_on_platform(vision.object_localizer, platform, boxes,
                                   out_dir=scale_dir, tag="find")
        summary["on_platform"] = found
        save()
        print(f"\nFind-on-scale: done. Points drawn red in {scale_dir}/find_cam*.jpg")
        return 0 if found else 1

    # 2. Baseline.
    base = read("baseline")
    if args.read_only:
        print(f"\nRead-only: done. Crops in {scale_dir}/")
        return 0 if base["value"] is not None else 1
    if centre is None:
        print("[Weigh] no scale located; not moving the object. Is grounding up "
              "(scripts/grounding.sh status)? Try --scale-prompt 'kitchen scale'")
        return 1

    # 3. Pick the object.
    ok, pick = step("pick_up", {"object_name": args.object})
    if not ok:
        ok, pick = step("pick_up", {"object_name": args.object})   # one retry, as the runs do
    if not ok:
        summary["outcome"] = "first pick failed"
        save()
        return 1
    key = str(skills.held).strip().lower()
    home_site = {"site": skills.pick_sites.get(key), "grasp": skills.pick_grasps.get(key),
                 "z_offset": skills.pick_z_offsets.get(key)}
    grasp = np.asarray(home_site["grasp"] if home_site["grasp"] is not None
                       else np.asarray(pick["grasp_pose"], dtype=float)[:3, 3], dtype=float)
    centroid = np.asarray(pick["centroid"], dtype=float)
    summary["pick"] = {"held": skills.held, **home_site}
    held_name, held_measure = skills.held, dict(skills.held_measure or {})

    # 4. Onto the scale.
    hang = float(grasp[2]) - REAL_TABLE_Z      # TCP height over what it lay on
    xy = centre - (centroid[:2] - grasp[:2]) + np.asarray(args.place_offset, dtype=float)
    release = [float(xy[0]), float(xy[1]), float(top_z + hang + args.release_clearance)]
    print(f"[Weigh] TCP was {hang * 1000:.0f} mm over the table at the pick; centroid "
          f"{np.round((centroid[:2] - grasp[:2]) * 1000).astype(int).tolist()} mm from the "
          f"jaws. Release TCP at {np.round(release, 4).tolist()}")
    place = {"target_location": release, "release_clearance": 0.0}
    if args.stop_force > 0:
        place.update({"stop_force_n": args.stop_force, "probe_above": 0.0,
                      "probe_below": args.probe_below, "probe_step": args.probe_step,
                      "probe_seconds": args.probe_seconds})
    # The validated pace, whatever --speed is: a faster speed shortens each
    # probe step too (to 1.0 s at most), and this descent is meant to be slow.
    ok, placed = step("place", place, speed=0)
    summary["on_scale"] = {"ok": ok, "release_tcp": release, "result": placed}
    arm = skills.skill_instance("place")
    # The jaws' orientation at the release: place keeps it through the retract.
    release_rot = np.asarray(arm.get_current_pose().rotation, dtype=float)
    if not ok:
        if placed.get("still_holding"):
            print("[Weigh] could not set it on the scale; putting it back")
            step("place", {"target_location": args.object})
        summary["outcome"] = "place on scale failed"
        save()
        return 1

    # 5. Arm away, settle, read.
    skills.go_home()
    time.sleep(args.settle)
    loaded = read("loaded")
    if loaded["value"] is not None and base["value"] is not None:
        summary["weight"] = round(loaded["value"] - base["value"], 3)
        print(f"\n[Weigh] {skills.held or args.object}: {summary['weight']} "
              f"{loaded['unit'] or ''} (loaded {loaded['value']} - baseline {base['value']})")

    # 6. Off the scale, back where it was first picked. NOT by pick_up: on the
    # scale, "white plastic tool" segments the scale and the spoon as one
    # white object (2026-11-13: 26 x 21 cm, centroid = the scale's centre), so
    # the grasp landed ~4 cm behind the spoon. Instead: find it by height
    # (what is above the platform), then put the jaws back on it the way they
    # held it -- the same orientation, the same offset from its centroid.
    grasp_pos = np.asarray(placed["place_position"], dtype=float)
    held_offset = centroid[:2] - grasp[:2]          # spoon centroid from the jaws
    expected = grasp_pos[:2] + held_offset
    found = locate_on_platform(vision.object_localizer, platform, boxes,
                               out_dir=scale_dir, tag="on_scale")
    summary["on_platform"] = found
    if found is None:
        print("[Weigh] nothing found on the platform by height; re-grasping where it "
              "was released")
    else:
        shift = found["centroid"][:2] - expected
        print(f"[Weigh] found {np.round(shift * 1000).astype(int).tolist()} mm from where "
              f"it was released (expected centroid {np.round(expected, 4).tolist()})")
        if np.linalg.norm(shift) <= args.max_shift:
            grasp_pos[:2] = found["centroid"][:2] - held_offset
        else:
            print(f"[Weigh] more than {args.max_shift * 1000:.0f} mm off: not trusting it; "
                  f"re-grasping where it was released")
    ok = regrasp(arm, grasp_pos, release_rot, args.regrasp_force)
    if ok:
        skills.held, skills.held_measure = held_name, held_measure
        vision.set_held(held_name)
        vision.forget_scans(None)
    else:
        summary["outcome"] = "re-grasp on the scale failed; object left on the scale"
        save()
        return 1
    key = str(skills.held).strip().lower()
    skills.pick_sites[key] = home_site["site"]
    if home_site["grasp"] is not None:
        skills.pick_grasps[key] = home_site["grasp"]
    skills.pick_z_offsets[key] = home_site["z_offset"] or 0.0
    ok, _ = step("place", {"target_location": args.object})
    if not ok:
        summary["outcome"] = "put-back failed"
        save()
        return 1

    # 7. The scale should be back at the baseline.
    skills.go_home()
    time.sleep(args.settle)
    read("after")
    summary["outcome"] = "done"
    save()
    print(f"\n[Weigh] done. Summary: {run_dir / 'weigh_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
