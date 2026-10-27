"""
right_spoon: push a scoop that lies on its side over onto its back.

NOT REGISTERED (2026-10-23). Tried on the robot -- a sideways push (slid it
4 cm), then a one-finger press on the crank corner by the bowl -- and it
never righted the scoop; the user stopped the feature ("even by hand it has
to be picked up and set down right"). ``spoon_pose`` is still used by
pick_up's ``right_if_on_side="warn"`` check. See progress.md.

The printed scoop has a cranked handle. Lying flat, mouth up, its handle end
stands ~14 mm higher than the bowl (2026-10-22 capture: handle end p95 23 mm
over the table, bowl 9 mm). On its side that rise turns into a sideways step:
seen from above the tool is a "Z", the handle's line 15-17 mm to one side of
the bowl's, and that side is where the mouth faces (2026-10-23,
``scene_captures/spoon_on_side_20261023_210216``). The bowl stands on its rim
there, 21 mm tall, and is no wider from above than the handle (15 / 15 mm,
against 27 / 19 mm flat).

So, from one scan:
  * flat (mouth up): the wide end is the low one;
  * upside down: the wide end is the high one (not handled here);
  * on its side: both ends equally wide, one standing tall -- the bowl.

On its side, the bowl's top edge is pushed away from the mouth with the closed
jaws, from the mouth side toward the other: the scoop rolls about its handle
and lands on its back, mouth up. Then it is scanned again; a second push is
made if it is still on its side.

Lifting it and rolling the wrist 90 deg was the first idea and does not work
here: the Franka hand is a ~200 mm bar along the jaws' closing axis, and a
90 deg roll stands that bar on end, ~100 mm below the fingertips -- it would
have to let go from over 10 cm up.
"""

from typing import Any, Dict, Optional, Tuple

import numpy as np

from .base_skill import BaseSkill


def spoon_pose(points, table_z: float, end_frac: float = 0.3) -> Dict[str, Any]:
    """
    How a scoop lies, from its cloud: {"pose": flat | upside_down | on_side |
    unknown, ...}. On its side it also gives "bowl_xy" (the standing bowl's
    centre) and "mouth_xy" (the unit direction its mouth faces, or None).
    """
    P = np.asarray(points, dtype=float)
    if len(P) < 50:
        return {"pose": "unknown", "why": "too few points"}
    z = P[:, 2] - float(table_z)
    origin = np.median(P[:, :2], axis=0)
    xy = P[:, :2] - origin
    w, v = np.linalg.eigh(np.cov(xy.T))
    u = v[:, int(np.argmax(w))]
    along = xy @ u
    across = xy @ np.array([-u[1], u[0]])
    lo, hi = np.percentile(along, [2, 98])
    span = hi - lo
    ends = {"A": along > hi - end_frac * span, "B": along < lo + end_frac * span}
    width = {k: float(np.percentile(across[m], 95) - np.percentile(across[m], 5))
             for k, m in ends.items()}
    height = {k: float(np.percentile(z[m], 95)) for k, m in ends.items()}
    info = {"length_mm": round(span * 1000, 1),
            "width_mm": {k: round(x * 1000, 1) for k, x in width.items()},
            "height_mm": {k: round(x * 1000, 1) for k, x in height.items()}}

    tall = max(height, key=height.get)
    low = "B" if tall == "A" else "A"
    wide = max(width, key=width.get)
    narrow = "B" if wide == "A" else "A"
    if width[wide] >= 1.3 * width[narrow]:
        info["pose"] = "flat" if wide == low else "upside_down"
        return info
    if height[tall] - height[low] < 0.008 or height[tall] < 0.015:
        info["pose"] = "unknown"
        info["why"] = "ends alike in width and height"
        return info

    # On its side: the bowl is the tall end. Fit each end's own line; the
    # handle's lies to the mouth's side of the bowl's.
    def segment(m):
        q = xy[m]
        c = q.mean(axis=0)
        ww, vv = np.linalg.eigh(np.cov((q - c).T))
        d = vv[:, int(np.argmax(ww))]
        return c, (d if d @ u > 0 else -d)

    c_bowl, d_bowl = segment(ends[tall])
    c_handle, d_handle = segment(ends[low])
    d = d_bowl + d_handle
    d = d / np.linalg.norm(d)
    n = np.array([-d[1], d[0]])
    step = float((c_handle - c_bowl) @ n)
    zb = z[ends[tall]]
    top = ends[tall] & (z >= np.percentile(zb, 70))
    # The crank's corner next to the bowl: walking from the bowl toward the
    # handle, where the points start to leave the bowl's line for the
    # handle's. On the 2026-10-23 capture it is 21 mm from the bowl's
    # centroid, 13.6 mm tall, and it is the spot the user pressed to right
    # the scoop by hand.
    e = d * np.sign((c_handle - c_bowl) @ d)
    t_all, a_all = (xy - c_bowl) @ e, (xy - c_bowl) @ n
    corner = None
    if abs(step) >= 0.006:
        for b0 in np.arange(0.0, 0.08, 0.003):
            m = (t_all >= b0) & (t_all < b0 + 0.003)
            if m.sum() >= 8 and np.median(a_all[m]) / step > 0.25:
                corner = b0
                break
    if corner is not None:
        m = (t_all >= corner - 0.004) & (t_all < corner + 0.002)
        info["corner_xy"] = (origin + c_bowl + e * corner).tolist()
        info["toward_handle_xy"] = e.tolist()
        info["corner_top"] = float(np.percentile(z[m], 90)) if m.sum() >= 5 else None
    info.update({"pose": "on_side",
                 "bowl_xy": (origin + c_bowl).tolist(),
                 "bowl_top_xy": (origin + xy[top].mean(axis=0)).tolist(),
                 "axis_xy": d.tolist(),
                 "crank_step_mm": round(abs(step) * 1000, 1),
                 "mouth_xy": (n * np.sign(step)).tolist() if abs(step) >= 0.006 else None})
    return info


class RightSpoonSkill(BaseSkill):
    """Push a scoop lying on its side over onto its back, then check."""

    name = "right_spoon"
    required_params = []

    @property
    def optional_params(self) -> Dict[str, Any]:
        return {
            "object_name": "larger spoon",
            # "press": lightly press the standing bowl's top down, a little to
            # its back's side, so it rolls over onto its back. The user's own
            # try (2026-10-23): pushing it sideways only slides it; a very
            # light press on the bowl rights it. "push": the sideways push.
            "method": "press",
            # press: where. "corner": the handle's bend next to the bowl (what
            # the user pressed by hand, 2026-10-23); "bowl": the bowl's top.
            # Falls back to the bowl's top if no corner is found.
            "press_at": "corner",
            # press: each retry follows this much further after contact.
            "press_follow_step": 0.003,
            # press on the corner with ONE finger: the jaws open this wide and
            # turned so the fingers lie one behind the other along the scoop,
            # the near one over the corner and the far one past the handle's
            # end. The closed jaws (~2 x 2 cm) reached the bowl, taller and
            # only 1.5-2 cm away, before the corner (the user, 2026-10-23:
            # "the press lands wrong, the gripper is too big"). None: closed.
            "press_opening": 0.06,
            # press: how far toward the bowl's back from its top the fingers
            # come down, and how much further each retry goes.
            "press_offset": 0.004,
            "press_offset_step": 0.003,
            # press: start this far above the bowl's top, step down this much
            # at a time, and call it contact at this change in force (N);
            # then go press_follow further so it rolls over.
            "press_above": 0.015,
            "press_step": 0.0015,
            "press_force_n": 0.8,
            # 4 mm was too little to roll it over (the user, 2026-10-23:
            # "the press is too weak"); the press is position-controlled, so
            # this is how far the corner is actually pushed down.
            "press_follow": 0.007,
            # How high the fingers meet the bowl: their lowest point at this
            # share of the standing bowl's measured height. Pushed low, the
            # scoop slides instead of tipping: 2026-10-23, TCP 18 mm over the
            # table (fingertips lower still) moved it 4 cm and turned it 30
            # deg, still on its side (the user: "pushing too deep, it slides").
            "push_contact_frac": 0.7,
            # How far the fingertips reach below the TCP, metres. An estimate,
            # not measured on this hand.
            "finger_below_tcp": 0.009,
            # Lowest the fingertips may go over the table, metres.
            # 5 mm: the 13.6 mm corner has to go down 7 mm, and with one
            # finger pressing the other is over bare table.
            "min_finger_height": 0.005,
            # After a push that slid the scoop (moved > slide_moved, still on
            # its side) the next push goes this much higher; after one that
            # did not move it (< still_moved) this much lower.
            "raise_after_slide": 0.004,
            "lower_after_miss": 0.003,
            "slide_moved": 0.010,
            "still_moved": 0.005,
            # Where the push starts, beyond the bowl's centre on its mouth
            # side, and how far past the centre it goes on the other side.
            # Tipping needs the top edge carried only past the bottom edge it
            # pivots on (~1 cm); 30 mm past the centre shoved it along.
            "push_start": 0.035,
            "push_past": 0.010,
            "push_seconds": 4.0,
            # Height over the push to come down from, and to leave at.
            "approach_height": 0.06,
            "attempts": 3,
            "reset_before_scan": True,
        }

    def check_preconditions(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        """The jaws close to push, so they must be empty."""
        try:
            width = self.held_width()
        except Exception:
            return True, "Preconditions met"
        if width > 0.005 and self.gripper_is_grasped():
            return False, (f"The gripper is holding something ({width * 1000:.0f} mm); "
                           f"put it down before righting the scoop")
        return True, "Preconditions met"

    def _press(self, pose, params, table, retry) -> Tuple[bool, str]:
        """Press the scoop down lightly -- on the crank corner by the bowl, or on
        the bowl's top -- until it gives, then a little further."""
        follow = float(params["press_follow"]) + retry * float(params["press_follow_step"])
        if (str(params["press_at"]).lower() == "corner" and pose.get("corner_xy") is not None
                and pose.get("corner_top") is not None):
            xy = np.asarray(pose["corner_xy"], dtype=float)
            part_h = float(pose["corner_top"])
            where = "the crank corner by the bowl"
        else:
            m = np.asarray(pose["mouth_xy"], dtype=float)
            offset = float(params["press_offset"]) + retry * float(params["press_offset_step"])
            xy = np.asarray(pose["bowl_top_xy"], dtype=float) - m * offset
            part_h = max(pose["height_mm"].values()) / 1000.0
            where = f"the bowl's top, {offset * 1000:.0f}mm toward its back"
        below = float(params["finger_below_tcp"])
        z_top = table + part_h + below                    # fingertips at that part's top
        R = self.tool_down_rotation()
        opening = params.get("press_opening")
        one_finger = (opening is not None and where.startswith("the crank")
                      and pose.get("toward_handle_xy") is not None)
        if one_finger:
            # Closing axis (tool Y) along the scoop. Either sign puts a finger
            # on each side of the TCP, so take the smaller wrist turn.
            e = np.asarray(pose["toward_handle_xy"], dtype=float)
            home_y = R[:2, 1]
            turn = float(np.arctan2(home_y[0] * e[1] - home_y[1] * e[0], home_y @ e))
            turn = (turn + np.pi / 2) % np.pi - np.pi / 2
            c, sn = np.cos(turn), np.sin(turn)
            R = np.array([[c, -sn, 0.0], [sn, c, 0.0], [0.0, 0.0, 1.0]]) @ R
            xy = xy + e * float(opening) / 2.0           # the near finger over the corner
            where += f", one finger (jaws {float(opening) * 1000:.0f}mm, wrist {np.degrees(turn):+.0f}°)"
        z = z_top + float(params["press_above"])
        z_floor = table + float(params["min_finger_height"]) + below
        print(f"[RightSpoon] Press on {where} at {np.round(xy, 3)} ({part_h * 1000:.1f}mm "
              f"tall): from {(z - below - table) * 1000:.0f}mm over the table, contact at a "
              f"{float(params['press_force_n']):.1f}N change, then {follow * 1000:.0f}mm more")
        if one_finger:
            self.open_gripper(width=float(opening))
        else:
            self.close_gripper(force=5.0)
        hover = np.array([xy[0], xy[1], z + float(params["approach_height"])])
        if not self.goto_pose_rigid(hover, R, duration=4.0):
            return False, "could not move over the bowl"
        if not self.reached(hover, 0.02)[0]:
            return False, "did not get over the bowl"
        probe = np.array([xy[0], xy[1], z])
        self.goto_pose_rigid(probe, R, duration=2.0)
        if self.ee_wrench() is None:
            return False, "this arm reports no force; not pressing blind"
        self.wait(0.3)
        baseline = self.mean_push_up_n(samples=5, gap=0.05)
        touched = None
        while z >= z_floor - 1e-9:
            probe[2] = z
            self.goto_pose_rigid(probe, R, duration=0.6)
            push = self.mean_push_up_n() - baseline
            if abs(push) > float(params["press_force_n"]):
                touched = z
                print(f"[RightSpoon]   contact at fingertips "
                      f"{(z - below - table) * 1000:.1f}mm over the table ({push:+.2f}N)")
                break
            z -= float(params["press_step"])
        if touched is None:
            print(f"[RightSpoon]   felt nothing down to "
                  f"{float(params['min_finger_height']) * 1000:.0f}mm over the table")
        else:
            probe[2] = max(z_floor, touched - follow)
            self.goto_pose_rigid(probe, R, duration=1.0)
        self.goto_pose_rigid(hover, R, duration=1.5)
        return True, "touched" if touched is not None else "nothing felt"

    def _table_z(self) -> float:
        t = self.config.get("table_z")
        return float(t) if t is not None else 0.0

    def _look(self, name, params) -> Tuple[Optional[dict], str]:
        if not self.clear_cameras(params, tag="RightSpoon"):
            return None, "could not clear the cameras"
        located = self.vision.locate(name, force_refresh=True)
        if located is None or located.get("points") is None:
            return None, f"cannot locate {name!r}"
        pose = spoon_pose(located["points"], self._table_z())
        print(f"[RightSpoon] {name!r}: {pose['pose']} -- widths {pose.get('width_mm')} mm, "
              f"heights {pose.get('height_mm')} mm"
              + (f", crank step {pose['crank_step_mm']} mm, mouth toward "
                 f"{np.round(pose['mouth_xy'], 2) if pose.get('mouth_xy') else 'unknown'}"
                 if pose["pose"] == "on_side" else ""))
        return pose, ""

    def execute(self, params: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        params = self.get_params_with_defaults(params)
        name = params["object_name"]
        table = self._table_z()
        history = []
        bump = 0.0              # height correction from what the last push did
        last_bowl = None
        for attempt in range(int(params["attempts"]) + 1):
            pose, why = self._look(name, params)
            if pose is None:
                return False, {"error": why, "history": history}
            history.append(pose)
            if pose["pose"] == "flat":
                print(f"[RightSpoon] {name!r} lies flat, mouth up"
                      + (f" after {attempt} push(es)" if attempt else " already"))
                return True, {"pose": "flat", "pushes": attempt, "history": history,
                              "height_bump": bump}
            if pose["pose"] != "on_side":
                return False, {"error": f"{name!r} is {pose['pose']}: not something this "
                                        f"skill can fix", "history": history}
            if attempt == int(params["attempts"]):
                break
            if pose.get("mouth_xy") is None:
                return False, {"error": (f"{name!r} is on its side, but which way its mouth "
                                         f"faces is unclear (crank step "
                                         f"{pose['crank_step_mm']} mm)"), "history": history}

            if str(params["method"]).lower() == "press":
                pressed = attempt
                ok, how = self._press(pose, params, table, pressed)
                if not ok:
                    self.go_home()
                    return False, {"error": how, "history": history}
                continue

            bowl = np.asarray(pose["bowl_xy"], dtype=float)
            m = np.asarray(pose["mouth_xy"], dtype=float)
            if last_bowl is not None:
                moved = float(np.linalg.norm(bowl - last_bowl))
                if moved > float(params["slide_moved"]):
                    bump += float(params["raise_after_slide"])
                    print(f"[RightSpoon] The last push slid it {moved * 1000:.0f}mm without "
                          f"tipping it: pushing {float(params['raise_after_slide']) * 1000:.0f}mm "
                          f"higher")
                elif moved < float(params["still_moved"]):
                    bump -= float(params["lower_after_miss"])
                    print(f"[RightSpoon] The last push hardly moved it ({moved * 1000:.0f}mm): "
                          f"pushing {float(params['lower_after_miss']) * 1000:.0f}mm lower")
            last_bowl = bowl
            bowl_h = float(pose["height_mm"][max(pose["height_mm"], key=pose["height_mm"].get)]) / 1000.0
            finger_low = max(float(params["min_finger_height"]),
                             float(params["push_contact_frac"]) * bowl_h + bump)
            z = table + finger_low + float(params["finger_below_tcp"])
            print(f"[RightSpoon] Bowl stands {bowl_h * 1000:.1f}mm tall: fingertips at "
                  f"{finger_low * 1000:.1f}mm over the table (TCP z={z:.4f})")
            up = z + float(params["approach_height"])
            start = bowl + m * float(params["push_start"])
            end = bowl - m * float(params["push_past"])
            R = self.tool_down_rotation()
            print(f"[RightSpoon] Push {attempt + 1}: closed jaws from "
                  f"{np.round(start, 3)} to {np.round(end, 3)} at TCP z={z:.4f}, "
                  f"across the bowl away from its mouth")
            self.close_gripper(force=5.0)
            for target, secs, label in (
                    (np.array([start[0], start[1], up]), 4.0, "over the start"),
                    (np.array([start[0], start[1], z]), 2.5, "down beside the bowl")):
                if not self.goto_pose_rigid(target, R, duration=secs):
                    return False, {"error": f"could not move {label}", "history": history}
                ok, err = self.reached(target, 0.02)
                if not ok:
                    self.go_home()
                    return False, {"error": f"did not get {label} (off by "
                                            f"{err * 1000:.0f}mm)", "history": history}
            self.goto_pose_rigid(np.array([end[0], end[1], z]), R,
                                 duration=float(params["push_seconds"]), use_impedance=True)
            self.goto_pose_rigid(np.array([end[0], end[1], up]), R, duration=2.0)
        return False, {"error": f"{name!r} is still on its side after "
                                f"{int(params['attempts'])} {params['method']}(es)",
                       "history": history}
