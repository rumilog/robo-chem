"""
Pick Pipette Skill

Takes the clear transfer pipette out of its holder by the bulb and draws liquid
into it from a chosen receptacle, in one skill:

  1. grasp the bulb from above and squeeze it ALL the way flat (air out)
  2. lift it straight up out of the holder, still squeezed
  3. lower the tip into the receptacle, feeling for the floor
  4. open the jaws only a little: the bulb springs back against them and
     draws liquid up the tip, while the jaws stay on the bulb
  5. lift clear, holding the pipette and what it drew

Afterwards ``dispense`` (mode "dispense") squeezes drops out of it: its
squeezes are measured from the width this skill leaves the jaws at.

Why it is not ``pick_up`` + ``dispense(mode="aspirate")``: pick_up grasps at
the cloud's centroid, which on this pipette is the stem (z 0.110 against a
bulb at 0.112-0.137, scene_captures/pipette_20261021_164147), and closes
gently; aspirate then squeezes 3 mm and releases to where it started. The
full squeeze has to happen at the pick, on the bulb, and the release has to
stop short of the bulb's relaxed width, which only the pick measured.

Perception (2026-10-21, real cell, camera-only): SAM 3 finds "pipette" on 4/4
cameras (0.77; "dropper" 0.82). The plastic is clear, so cams 2 and 5 read
the table THROUGH it (z ~ -0.005) and consensus discards them; cams 3 and 4
agree within 7 mm. The bulb is the top ~25 mm of the cloud, 9-12 mm across.
"""

from typing import Dict, Any, Tuple
import numpy as np

from .base_skill import BaseSkill


class PickPipetteSkill(BaseSkill):
    """
    Pick the pipette up by its bulb, squeezed flat, and draw from a receptacle.

    Pipeline:
    1. Open, home, scan the pipette; grasp point = the bulb (top of the cloud)
    2. Hover, descend around the bulb, touch it lightly to measure its width
    3. Squeeze it flat (force-limited close at squeeze_force)
    4. Lift straight up out of the holder
    5. Home, scan the receptacle, hover the tip over its opening
    6. Feel down to the floor (or dip_depth) in small steps; back off
    7. Release the jaws a little, wait for the liquid to rise, lift clear
    """

    name = "pick_pipette"
    required_params = ["source_container"]

    #: Fingers straight down, closing along base Y (the home orientation).
    TOP_DOWN = np.diag([1.0, -1.0, -1.0])

    @property
    def optional_params(self) -> Dict[str, Any]:
        return {
            # SAM 3 prompt for the pipette. "pipette" and "dropper" both hit
            # 4/4 cameras on 2026-10-21; "pipette bulb" and "bulb" hit none.
            "object_name": "pipette",
            # SAM category when source_container is a written label (as stir).
            "container_category": None,
            # TCP this far below the top of the pipette's cloud. The bulb is
            # the top ~25 mm (9-12 mm across, stem 2-7 mm below it), so 15 mm
            # puts the pads on the bulb's fat middle, clear of the stem.
            "grip_below_top": 0.015,
            # World-XY nudge of the grasp, metres, as pick_up: +X away from
            # the base, +Y to the robot's left.
            "forward_offset": 0.0,
            "lateral_offset": 0.0,
            # Light close first, to measure the bulb's relaxed width (and to
            # tell a miss from a squeezed bulb, which both read narrow).
            "touch_force": 1.0,
            # Then squeeze it flat. Force-limited: the jaws close until the
            # pads push this hard, which a collapsed bulb's two walls resist.
            "squeeze_force": 20.0,
            # A touch narrower than this closed on the stem or on nothing.
            "min_bulb_width": 0.004,
            "max_bulb_width": 0.030,
            # How far to open the jaws after the tip is in the liquid, metres.
            # The bulb re-expands only as far as the jaws let it, so this sets
            # how much is drawn.
            "release_amount": 0.005,
            # ...but never wider than the bulb's relaxed width minus this, so
            # the bulb still presses on both pads and the pipette stays held.
            "hold_margin": 0.003,
            # Seconds with the tip in the liquid after the release.
            "draw_seconds": 3.0,
            # TCP -> pipette tip, metres. NOT MEASURED: estimated from the
            # pipette's top at z 0.137 standing in the stirrer's holder, whose
            # bore floor is near the table. Short is the safe side: the
            # descent stops on contact, so a tip that is really longer meets
            # the floor early and stops there; one that is really shorter would
            # stop short in the air. Measure it and pass it.
            "tool_length": 0.11,
            # Straight up out of the holder, metres. The stem sits ~10 cm deep.
            "lift_height": 0.15,
            "approach_height": 0.08,   # above the bulb top before the grasp
            # Tip this far below the rim, metres. None feels down to the
            # floor instead -- 10 ml in a 52 mm cup stands 3-5 mm deep, which a
            # rim-relative depth does not reach.
            "dip_depth": None,
            # Force-guarded descent, as stir's: stop on a change of this many
            # newtons from a baseline taken at rest above the rim. None
            # descends on position alone (and refuses dip_depth None).
            "stop_force_n": 1.0,
            "probe_above_rim": 0.010,
            # Fine steps over the last fine_zone above the target. Wide on
            # purpose: tool_length is a guess, and a tip that is really 30 mm
            # longer should still meet the floor during a fine step.
            "probe_step": 0.002,
            "coarse_step": 0.005,
            "fine_zone": 0.035,
            "probe_seconds": 0.4,
            # Aim the tip this far past the measured cup bottom, so the force
            # guard, not the arithmetic, decides where the floor is.
            "floor_overshoot": 0.010,
            # Back off this far after contact before releasing: the tip is
            # then ~1 mm off the floor, inside a 3-5 mm deep liquid.
            "contact_backoff": 0.003,
            # The fingertips must stay this far above the rim.
            "hand_clearance": 0.020,
            "reset_before_scan": True,
            "hover_tol": 0.05,
            "grasp_tol": 0.015,
            "descend_tol": 0.03,
        }

    def check_preconditions(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        return self.validate_params(params)

    def execute(self, params: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        params = self.get_params_with_defaults(params)
        name = params["object_name"]
        source = params["source_container"]
        tool_length = float(params["tool_length"])
        R = self.TOP_DOWN

        print(f"[PickPipette] Picking '{name}' by its bulb, then drawing "
              f"from '{source}'")

        # 1. Empty the jaws where they are, then clear the cameras and look.
        if not self.open_gripper():
            return False, {"error": "Failed to open gripper"}
        self.wait(0.4)
        if not self.clear_cameras(params, tag="PickPipette"):
            return False, {"error": "Failed to reset_joints before scanning"}

        self.vision.clear_cache()
        located = self.vision.locate(name, force_refresh=True)
        if located is None:
            return False, {"error": f"Could not find '{name}' in the scene"}
        points = np.asarray(located["points"], dtype=float)
        if len(points) < 20:
            return False, {"error": f"'{name}' has only {len(points)} points"}
        print(f"[PickPipette] Cameras {located.get('cameras')} "
              f"(rejected {located.get('rejected_cameras') or []})")

        # The bulb is the top of the cloud: XY from its top 25 mm, which the
        # stem and anything the mask caught at the holder's mouth stay out of.
        top_z = float(np.percentile(points[:, 2], 99))
        bulb = points[points[:, 2] > top_z - 0.025]
        if len(bulb) < 10:
            bulb = points
        bulb_xy = np.median(bulb[:, :2], axis=0)
        grasp = np.array([bulb_xy[0] + float(params["forward_offset"]),
                          bulb_xy[1] + float(params["lateral_offset"]),
                          top_z - float(params["grip_below_top"])])
        print(f"[PickPipette] Top of '{name}' at z={top_z:.4f}; bulb centre "
              f"{np.round(bulb_xy, 4)}; grasp {np.round(grasp, 4)}")

        # 2. Down around the bulb. The jaws are 80 mm open around a ~11 mm
        # bulb, so XY can be off by centimetres; height cannot be.
        hover = np.array([grasp[0], grasp[1], max(self.safe_height,
                                                  top_z + 0.10)])
        approach = np.array([grasp[0], grasp[1],
                             top_z + float(params["approach_height"])])
        for label, xyz, secs, tol in (("hover", hover, 3.0, params["hover_tol"]),
                                      ("approach", approach, 2.5, params["hover_tol"]),
                                      ("grasp", grasp, 3.0, params["grasp_tol"])):
            print(f"[PickPipette]   {label} {np.round(xyz, 4)}...")
            if not self.goto_pose_rigid(xyz, R, duration=secs):
                return False, {"error": f"Failed to command the {label} pose"}
            arrived, err = self.reached(xyz, float(tol))
            if not arrived:
                return False, {"error": (f"{label.capitalize()} over '{name}' "
                                         f"not reached (off by {err * 1000:.0f}mm)")}
        grasp_tcp_reached = np.asarray(self.get_current_pose().translation,
                                       dtype=float)

        # 3. Touch, then squeeze flat.
        if not self.close_gripper(force=float(params["touch_force"]),
                                  force_limited=True):
            return False, {"error": "Failed to close on the bulb"}
        self.wait(0.3)
        relaxed = self.held_width()
        print(f"[PickPipette] Bulb is {relaxed * 1000:.1f}mm across at "
              f"{float(params['touch_force']):.1f}N")
        if not (float(params["min_bulb_width"]) <= relaxed
                <= float(params["max_bulb_width"])):
            self.open_gripper()
            return False, {
                "error": (f"Closed to {relaxed * 1000:.1f}mm, not a bulb "
                          f"({float(params['min_bulb_width']) * 1000:.0f}-"
                          f"{float(params['max_bulb_width']) * 1000:.0f}mm): "
                          f"missed it, or caught the stem. Adjust grip_below_top "
                          f"or the offsets"),
                "relaxed_width": relaxed,
            }

        print(f"[PickPipette] Squeezing the bulb flat "
              f"({float(params['squeeze_force']):.0f}N)...")
        if not self.close_gripper(force=float(params["squeeze_force"]),
                                  force_limited=True):
            return False, {"error": "Failed to squeeze the bulb"}
        self.wait(0.4)
        squeezed = self.held_width()
        print(f"[PickPipette] Squeezed {relaxed * 1000:.1f} -> "
              f"{squeezed * 1000:.1f}mm")
        if squeezed > 0.6 * relaxed:
            # Not a failure: it still holds, it will just draw less.
            print(f"[PickPipette] WARNING: the bulb only went to "
                  f"{squeezed / relaxed * 100:.0f}% of its width; raise "
                  f"squeeze_force for a fuller draw")

        # 4. Straight up out of the holder, slowly: the stem slides up the bore.
        lifted = grasp_tcp_reached + np.array([0.0, 0.0, float(params["lift_height"])])
        print(f"[PickPipette] Lifting out of the holder to z={lifted[2]:.4f}...")
        if not self.goto_pose_rigid(lifted, R, duration=4.0):
            return False, {"error": "Failed to command the lift"}
        if self._dropped_while_squeezed(squeezed):
            return False, {"error": (f"Pipette lost on the lift: jaws at "
                                     f"{self.held_width() * 1000:.1f}mm, were "
                                     f"{squeezed * 1000:.1f}mm"),
                           "relaxed_width": relaxed, "squeezed_width": squeezed}

        result: Dict[str, Any] = {
            "grasped_object": name,
            "centroid": [float(v) for v in located["centroid"]],
            "pipette_top_z": top_z,
            "grasp_pose": np.vstack([np.column_stack([R, grasp]),
                                     [0, 0, 0, 1]]).tolist(),
            "grasp_tcp_reached": grasp_tcp_reached.tolist(),
            "z_offset": 0.0,
            "relaxed_width": relaxed,
            "squeezed_width": squeezed,
            "tool_length": tool_length,
        }

        # 5. Find the receptacle with the arm out of the way.
        if not self.clear_cameras(params, tag="PickPipette"):
            return False, {**result, "error": "Failed to reset_joints before "
                                              "scanning the source"}
        if self._dropped_while_squeezed(squeezed):
            return False, {**result, "error": "Pipette lost while homing"}
        self.vision.clear_cache()
        cup = self.locate_container(source, force_refresh=True,
                                    category=params.get("container_category"))
        if cup is None:
            return False, {**result, "error": f"Cannot locate source '{source}'"}
        center, rim_z, base_z = cup["rim_center"], cup["top_z"], cup["base_z"]
        print(f"[PickPipette] '{source}' rim centre {np.round(center, 4)}, "
              f"z={rim_z:.4f}, bottom z={base_z:.4f}")

        dip = params.get("dip_depth")
        tip_target = (rim_z - float(dip) if dip is not None
                      else base_z - float(params["floor_overshoot"]))
        target_z = tip_target + tool_length
        lowest = rim_z + float(params["hand_clearance"])
        if target_z < lowest:
            print(f"[PickPipette] The tip target would bring the fingers within "
                  f"{float(params['hand_clearance']) * 1000:.0f}mm of the rim; "
                  f"stopping the TCP at z={lowest:.4f}")
            target_z = lowest

        tip_hover = np.array([center[0], center[1],
                              rim_z + 0.08 + tool_length])
        print(f"[PickPipette] Tip over the opening, TCP {np.round(tip_hover, 4)}...")
        if not self.goto_pose_rigid(tip_hover, R, duration=3.0):
            return False, {**result, "error": "Failed to command the hover"}
        arrived, err = self.reached(tip_hover, float(params["hover_tol"]))
        if not arrived:
            return False, {**result, "error": (f"Hover over '{source}' not "
                                               f"reached (off by {err * 1000:.0f}mm)")}

        # 6. Into the liquid.
        stop_force = params.get("stop_force_n")
        if stop_force is not None and self.ee_wrench() is None:
            stop_force = None
        if stop_force is None and dip is None:
            return False, {**result, "error": (
                "Feeling for the floor needs a force reading (stop_force_n); "
                "pass dip_depth to descend on position alone")}
        if stop_force is not None:
            felt = self._feel_down(center, target_z, rim_z, tool_length, R,
                                   float(stop_force), params)
            result.update({k: v for k, v in felt.items() if k != "error"})
            if felt.get("error"):
                self.goto_pose_rigid(tip_hover, R, duration=3.0)
                return False, {**result, "error": felt["error"]}
            dip_z = felt["dip_z"]
            if dip is None and not felt["stopped_by_force"]:
                # Aimed past the bottom and met nothing: the pipette is shorter
                # than tool_length says, so the tip may be over the liquid.
                print(f"[PickPipette] WARNING: no floor felt "
                      f"{float(params['floor_overshoot']) * 1000:.0f}mm past the "
                      f"cup bottom -- tool_length {tool_length:.3f} is probably "
                      f"longer than the pipette; drawing here anyway")
        else:
            dip_z = target_z
            print(f"[PickPipette] Lowering to z={dip_z:.4f} (position only)...")
            if not self.goto_pose_rigid(np.array([center[0], center[1], dip_z]),
                                        R, duration=3.0):
                return False, {**result, "error": "Failed to command the dip"}
        arrived, err = self.reached(np.array([center[0], center[1], dip_z]),
                                    float(params["descend_tol"]))
        if not arrived:
            self.goto_pose_rigid(tip_hover, R, duration=3.0)
            return False, {**result, "error": (f"Tip did not reach the liquid in "
                                               f"'{source}' (off by "
                                               f"{err * 1000:.0f}mm)")}
        if self._dropped_while_squeezed(squeezed):
            return False, {**result, "error": "Pipette lost on the way down"}

        # 7. Let go a little. Position mode: the jaws stop at release_width
        # and the bulb pushes back against them.
        release_width = min(squeezed + float(params["release_amount"]),
                            relaxed - float(params["hold_margin"]))
        release_width = max(release_width, squeezed)
        print(f"[PickPipette] Releasing {squeezed * 1000:.1f} -> "
              f"{release_width * 1000:.1f}mm (bulb relaxed "
              f"{relaxed * 1000:.1f}mm) to draw...")
        if not self.close_gripper(target_width=release_width,
                                  force_limited=False):
            return False, {**result, "error": "Failed to release the squeeze"}
        # Let the jaws (and the bulb behind them) actually get there before
        # timing the draw: on 2026-10-21 the tip came out of cup A while the
        # bulb was still filling.
        t = 0.0
        while self.held_width() < release_width - 0.0003 and t < 5.0:
            self.wait(0.1)
            t += 0.1
        print(f"[PickPipette] Jaws at {self.held_width() * 1000:.1f}mm; drawing "
              f"for {float(params['draw_seconds']):.1f}s with the tip in...")
        self.wait(float(params["draw_seconds"]))
        held = self.held_width()

        lift = np.array([center[0], center[1], rim_z + 0.08 + tool_length])
        print(f"[PickPipette] Lifting the tip clear of '{source}'...")
        if not self.goto_pose_rigid(lift, R, duration=3.0):
            return False, {**result, "error": "Drew, but failed to lift clear"}

        self.vision.clear_cache()
        print(f"[PickPipette] Done: holding '{name}' at {held * 1000:.1f}mm "
              f"after drawing from '{source}'")
        result.update({
            "source_container": source,
            "release_width": release_width,
            "gripper_width": held,
            "dip_z": float(dip_z),
            "tip_z": float(dip_z - tool_length),
            "rim_z": rim_z,
            # Nothing senses the liquid or the pipette once the jaws are in
            # position mode: a width-held pipette that slid out reads the same.
            "volume_measured": False,
        })
        return True, result

    def _dropped_while_squeezed(self, squeezed: float) -> bool:
        """The squeeze is force-limited, so a lost pipette lets the jaws shut."""
        width = self.held_width()
        if width < max(0.0005, 0.5 * squeezed):
            print(f"[PickPipette] Jaws shut to {width * 1000:.1f}mm from "
                  f"{squeezed * 1000:.1f}mm: the pipette is gone")
            return True
        return False

    def _feel_down(self, center, target_z: float, rim_z: float,
                   tool_length: float, R, stop_force: float,
                   params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Step the tip down to ``target_z`` (TCP), stopping on contact.

        Stir's descent, for a tip on a pipette: the trigger is a change in
        vertical force from a resting baseline, by magnitude (pressing reads
        negative on this arm). Contact with the tip at or above the rim is the
        rim, and refused; contact inside is the floor, and the tip backs off
        ``contact_backoff`` to draw there.
        """
        step = max(0.0005, float(params["probe_step"]))
        coarse = max(step, float(params["coarse_step"]))
        seconds = float(params["probe_seconds"])
        z_from = max(target_z, rim_z + float(params["probe_above_rim"]) + tool_length)
        probe = np.array([center[0], center[1], z_from])

        print(f"[PickPipette] Tip to {float(params['probe_above_rim']) * 1000:.0f}mm "
              f"above the rim (TCP z={z_from:.4f})...")
        if not self.goto_pose_rigid(probe, R, duration=3.0):
            return {"error": "Failed to command the pose above the rim"}
        self.wait(0.3)
        baseline = self.mean_push_up_n(samples=5, gap=0.05)
        print(f"[PickPipette] Feeling down to z={target_z:.4f}, stopping on a "
              f"{stop_force:.1f}N change (baseline {baseline:+.2f}N)")

        fine_from = target_z + float(params["fine_zone"])
        z = z_from
        while z > target_z + 1e-9:
            this_step = coarse if z - coarse >= fine_from else step
            z = max(target_z, z - this_step)
            probe[2] = z
            if not self.goto_pose_rigid(probe, R, duration=seconds):
                return {"error": "Failed to command a descent step"}
            if this_step > step:
                self.wait(0.15)
            delta = self.mean_push_up_n() - baseline
            if abs(delta) <= stop_force:
                continue

            tip_below_rim = rim_z - (z - tool_length)
            print(f"[PickPipette] Contact at z={z:.4f}: {delta:+.2f}N, tip "
                  f"{tip_below_rim * 1000:.0f}mm below the rim")
            felt = {"stopped_by_force": True, "contact_force_n": float(delta),
                    "contact_z": float(z),
                    "contact_tip_below_rim": float(tip_below_rim)}
            if tip_below_rim <= this_step + 0.0005:
                felt["error"] = (f"The tip hit something at or above the rim "
                                 f"({delta:+.2f}N) -- it is landing on the rim, "
                                 f"not in '{params['source_container']}'")
                return felt
            felt["dip_z"] = min(z + float(params["contact_backoff"]), z_from)
            probe[2] = felt["dip_z"]
            if not self.goto_pose_rigid(probe, R, duration=1.0):
                felt["error"] = "Failed to back off after contact"
            return felt

        print(f"[PickPipette] Reached z={target_z:.4f} without contact")
        return {"stopped_by_force": False, "contact_force_n": None,
                "dip_z": float(target_z)}
