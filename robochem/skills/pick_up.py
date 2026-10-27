"""
Pick Up Skill

Picks up an object from the workspace using SAM segmentation,
pointcloud localization, and grasp analysis.
"""

from typing import Dict, Any, Tuple, List
import numpy as np

from .base_skill import BaseSkill


def width_along_closing_axis(points, grasp_pose, slab=0.01,
                             lo_pct=3, hi_pct=97, max_radius=0.06):
    """
    How wide the object is where the fingers will actually meet it.

    Measured along the gripper's closing axis (its local y), over a thin
    horizontal slab at the grasp height. Points farther than ``max_radius``
    from the grasp XY are dropped so a bloated SAM cloud (table, neighbour
    objects) cannot invent a 70mm+ "beaker".
    """
    centre = grasp_pose[:3, 3]
    closing = grasp_pose[:3, 1]

    horizontal = closing[:2]
    norm = np.linalg.norm(horizontal)
    if norm < 1e-6:
        return None
    horizontal = horizontal / norm

    # Keep only a neighbourhood of the grasp — fused clouds often include
    # far-away surface from a fat mask.
    xy_dist = np.linalg.norm(points[:, :2] - centre[:2], axis=1)
    near = points[(xy_dist < max_radius) & (np.abs(points[:, 2] - centre[2]) < slab)]
    if len(near) < 20:
        near = points[np.abs(points[:, 2] - centre[2]) < slab]
    if len(near) < 20:
        return None

    projected = (near[:, :2] - centre[:2]) @ horizontal
    return float(np.percentile(projected, hi_pct)
                 - np.percentile(projected, lo_pct))


def measure_tool_offset(points, grasp_pose, min_span=0.015):
    """
    Where the working end of a held tool sits, relative to the grasp.

    Returns [x, y, z] in the TOOL frame — the vector ``scoop``/``stir`` want as
    ``tool_offset``. Computed from the segmented cloud at the moment of the
    grasp, which is the only place that knows both the tool's shape and where
    the jaws actually closed on it.

    This exists because that offset is NOT a property of the tool alone: shift
    the grasp 10mm along the handle and the bowl is 10mm further away. Tuning
    ``forward_offset`` therefore silently invalidates any hand-measured
    ``tool_offset``, and the scoop then digs next to the powder instead of in
    it. Reporting it here keeps the two in step.

    The working end is taken to be the bulkier end of the tool along the
    gripper's approach-perpendicular axis — for a scoop that is the bowl, which
    is wider and deeper than the handle it hangs off.

    IMPORTANT — the z component is a LOWER BOUND, not a measurement. The cage
    cameras look down, so the underside of a bowl is occluded from all of them
    and the cloud stops at the rim. On the printed scoop this reads ~11mm when
    the floor that actually digs is 28mm below the grasp: use it far enough
    shallow and the scoop hovers over the powder and comes up empty. Take x and
    y from here, and z from the CAD wherever you have it.
    """
    points = np.asarray(points, dtype=float)
    if len(points) < 20:
        return None

    origin = grasp_pose[:3, 3]
    R = grasp_pose[:3, :3]
    local = (points - origin) @ R          # world -> tool frame

    forward = local[:, 0]
    pos, neg = local[forward > min_span], local[forward < -min_span]
    # The bowl end is the one with more material hanging off it.
    if len(pos) == 0 and len(neg) == 0:
        return None
    end = pos if len(pos) >= len(neg) else neg
    if len(end) < 10:
        return None

    # Within that end, the working point is the far, deepest part of the bowl.
    far = end[np.abs(end[:, 0]) > np.percentile(np.abs(end[:, 0]), 70)]
    if len(far) < 5:
        far = end
    return np.array([
        float(np.median(far[:, 0])),
        float(np.median(far[:, 1])),
        float(np.percentile(far[:, 2], 95)),   # +Z is down in the tool frame
    ])


def tool_extent_x(points, grasp_pose):
    """
    Where a held tool's two ends lie along the tool x axis, from the grasp.

    [1st, 99th] percentile of the cloud's x in the TOOL frame. For a spoon
    picked across its handle, x runs along it. The middle of the two ends,
    with the CAD distance from there to the bowl, says where along the handle
    the jaws closed to a millimetre or two -- which the scoop's depth depends
    on steeply (sim, 2026-10-03: x off by 2 mm took a scoop from 1.24 ml to
    0.68). Both ends, not the bowl's alone: the bowl's far edge is thinly
    sampled, and its 98th percentile fell 5 mm short of the real edge, while
    the span's middle was within 1.5 mm.
    """
    points = np.asarray(points, dtype=float)
    if len(points) < 20:
        return None
    local = (points - grasp_pose[:3, 3]) @ grasp_pose[:3, :3]
    return [float(np.percentile(local[:, 0], 1)), float(np.percentile(local[:, 0], 99))]


#: Largest wrist turn about world Z a pick makes. Joint 7 sits at +45 deg at
#: home with a +-166 deg range, so +-120 is clear whichever way it turns.
MAX_GRASP_YAW_DEG = 120.0


def drop_stray_points(points, radius: float = 0.003, min_neighbours: int = 8):
    """
    ``points`` without the sparse ones: fewer than ``min_neighbours`` points
    (itself included) within ``radius``.

    The depth cameras leave arcs of stray points along an object's edges. On
    2026-10-27 they lay beside the red scoop's handle and made it read 22 mm
    wide against the bowl's 27, so a flat scoop was called "on its side" and
    yaw auto refused (bowl and handle alike). Without them the same cloud read
    11 vs 21 mm: flat, wrist +9.6 deg. Only for decisions about the tool's
    shape; the grasp is computed from the cloud as it was. When the filter
    would take away more than half, the cloud is returned whole.
    """
    P = np.asarray(points, dtype=float)
    if len(P) < 50:
        return P
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return P
    counts = np.array([len(n) for n in cKDTree(P).query_ball_point(P, float(radius))])
    kept = P[counts >= int(min_neighbours)]
    return kept if len(kept) >= 0.5 * len(P) else P


def tool_long_axis_yaw(points, end_frac: float = 0.3):
    """
    (yaw, info): the direction, in the table plane, from a long tool's handle
    end to its working end, radians from base +X; or (None, why).

    The long axis is the principal direction of the cloud's XY. Which end is
    the bowl is decided by width: the points in the outer ``end_frac`` of
    each end, their spread across the axis. The printed scoop's bowl is
    20.5 mm across and its handle ~8 mm, so the wider end is the bowl.
    """
    P = np.asarray(points, dtype=float)
    if len(P) < 50:
        return None, "too few points"
    xy = P[:, :2] - np.median(P[:, :2], axis=0)
    w, v = np.linalg.eigh(np.cov(xy.T))
    u = v[:, int(np.argmax(w))]
    if w.max() < 4.0 * max(w.min(), 1e-12):
        return None, f"not elongated (axes {np.sqrt(w.max()) * 1000:.0f}/{np.sqrt(w.min()) * 1000:.0f} mm)"
    along, across = xy @ u, xy @ np.array([-u[1], u[0]])
    lo, hi = np.percentile(along, [2, 98])
    span = hi - lo
    head = across[along > hi - end_frac * span]
    tail = across[along < lo + end_frac * span]
    if len(head) < 10 or len(tail) < 10:
        return None, "an end has too few points"
    wh = float(np.percentile(head, 95) - np.percentile(head, 5))
    wt = float(np.percentile(tail, 95) - np.percentile(tail, 5))
    if wt > wh:
        u, wh, wt = -u, wt, wh
    info = {"length_mm": round(span * 1000, 1), "bowl_width_mm": round(wh * 1000, 1),
            "handle_width_mm": round(wt * 1000, 1)}
    if wh < 1.3 * wt:
        return None, (f"cannot tell the bowl end from the handle "
                      f"({wh * 1000:.0f} vs {wt * 1000:.0f} mm wide)")
    # Where the working end is, the way measure_tool_offset reads it after the
    # pick (the median of the far 30% of the bowl end), and the handle's
    # centre line. A grasp placed from these does not lean with the
    # centroid, which moves with how the cameras see the tool.
    n = np.array([-u[1], u[0]])
    along, across = xy @ u, xy @ n
    bowl_side = along[along > 0.0]
    far = bowl_side[bowl_side >= np.percentile(bowl_side, 70)] if len(bowl_side) >= 10 else bowl_side
    handle = across[along < np.percentile(along, 30)]
    origin = np.median(P[:, :2], axis=0)
    info["working_end_xy"] = (origin + u * float(np.median(far))).tolist()
    info["handle_line_xy"] = (origin + n * float(np.median(handle))).tolist()
    info["axis_xy"] = u.tolist()
    return float(np.arctan2(u[1], u[0])), info


def square_top_yaw(points, band: float = 0.006):
    """
    (yaw, info): how far a square-topped object is turned about Z, radians in
    (-45, 45] deg -- the smallest wrist turn that puts the jaws flat on two
    opposite faces; or (None, why).

    Fitted to the top ``band`` of the cloud (the stirrer's 30 mm cube; the
    rod and holder below are round, so only the cube's turn matters): the
    minimum-area rectangle around those points' XY. A rectangle that is far
    from square, or the wrong size, is refused.
    """
    import cv2
    P = np.asarray(points, dtype=float)
    if len(P) < 30:
        return None, "too few points"
    top = P[P[:, 2] >= np.percentile(P[:, 2], 97) - band]
    if len(top) < 20:
        return None, "too few points on the top face"
    rect = cv2.minAreaRect((top[:, :2] * 1000.0).astype(np.float32))
    w, h = sorted(rect[1])
    info = {"top_mm": [round(w, 1), round(h, 1)], "top_points": int(len(top))}
    if not (20.0 <= h <= 45.0) or h > 1.4 * max(w, 1e-6):
        return None, f"top is not a square of the cube's size ({w:.0f} x {h:.0f} mm)"
    box = cv2.boxPoints(rect)
    edge = box[1] - box[0]
    yaw = float(np.arctan2(edge[1], edge[0]))
    yaw = (yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4        # nearest face
    return yaw, info


def yaw_about_z(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def upright_from_pitched(pose: np.ndarray) -> np.ndarray:
    """
    Same XY/closing as a pitched cup grasp, but tool Z straight down.

    Used for the high hover so we only ask for the 25° wrist tip once the
    arm is already above the cup — much easier for IK near the workspace edge.
    """
    out = pose.copy()
    y = pose[:3, 1].copy()
    y[2] = 0.0
    y = y / (np.linalg.norm(y) + 1e-9)
    z = np.array([0.0, 0.0, -1.0])
    x = np.cross(y, z)
    x = x / (np.linalg.norm(x) + 1e-9)
    y = np.cross(z, x)
    out[:3, 0] = x
    out[:3, 1] = y
    out[:3, 2] = z
    return out


class PickUpSkill(BaseSkill):
    """
    Pick up an object from the workspace.
    
    Pipeline:
    1. SAM3 segment the target object in scene images
    2. Get pointcloud of object from multi-camera fusion
    3. Analyze pointcloud for grasp affordances
    4. Plan approach trajectory (approach from above)
    5. Open gripper
    6. Move to pre-grasp position
    7. Move to grasp position
    8. Close gripper
    9. Lift object
    """
    
    name = "pick_up"
    required_params = ["object_name"]
    
    @property
    def optional_params(self) -> Dict[str, Any]:
        return {
            "grasp_type": "top",  # upright top-down; "side" still available if needed
            "approach_height": 0.10,  # Height above object for approach
            "lift_height": 0.15,  # Height to lift after grasping
            # Max gripper force (N). With force_limited=True the jaws stop as
            # soon as contact force hits this — keep it low for "any resistance".
            "grasp_force": 1.5,
            # Only used when grasp_type="side" (pour-style tipped grasps).
            "pitch_deg": 0.0,
            # Close mode: True = squeeze until contact force (good for rigid
            # objects / flat spoons). False = close to measured diameter minus
            # squeeze (needed for soft paper cups that give no force feedback).
            "force_limited": True,
            # How much narrower than the measured object diameter to close
            # (metres). Only used when force_limited=False.
            "squeeze": 0.004,
            # Below this fraction of the measured object width, treat the
            # grasp as having crushed the object (width-close mode only).
            "crush_fraction": 0.7,
            # World-Z shift of the grasp, metres. Negative lowers the fingers.
            # Keep near 0 for flat objects (spoons); +0.02 floats above them.
            "z_offset": 0.0,
            # World-XY shift of the grasp, metres, same sign convention as
            # pour's site offsets: +X is forward from the base toward the
            # workspace, so NEGATIVE forward_offset grasps closer to the base.
            #
            # Needed for any object whose centroid is not where you want the
            # jaws. The printed scoop is the case in point: its mass is in the
            # bowl, so the computed grasp lands on the crank where the handle
            # bends down — the one spot on the tool where the cross-section is
            # changing and the jaws cannot seat flat.
            "forward_offset": 0.0,
            "lateral_offset": 0.0,
            # Take the object from behind instead of from above, metres; 0
            # comes straight down. The jaws open fully with their closing axis
            # across the slide (base Y), go down this far behind the grasp
            # (toward the base, -X), then slide forward onto the object and
            # close. The user's procedure for the clear cups (2026-10-08):
            # straight down, the 55.5 mm rim leaves ~12 mm a side in the
            # 80 mm opening and a finger could land on it.
            "slide_in": 0.0,
            # Lean the fingers this many degrees toward +X (away from the
            # base) at the grasp, the object still standing upright. pour tips
            # the fingers toward the base, so a cup taken like this is that
            # much further over than the wrist: 10 deg makes a wrist that
            # stalls at 80 deg still pour at 90 (the user, 2026-10-21, for the
            # oil beaker). place sets it back down at this same tilt.
            "tilt_away_deg": 0.0,
            # Turn the wrist about world Z before a top grasp. None: as at
            # home (jaws close along base Y). "auto": along the tool's long
            # axis, tool X toward its working end, so a spoon lying at any
            # angle is taken across its handle and sits in the jaws exactly as
            # one lying straight ahead does -- the skills after the pick home
            # the wrist and see the same tool. forward_offset/lateral_offset
            # then run along/across the handle. A number: that many degrees.
            # Turns past MAX_GRASP_YAW_DEG (a bowl pointing at the base) fall
            # back to no turn.
            "yaw": None,
            # For a long tool picked with yaw "auto": put the jaws this far
            # from its working end, measured along the handle, on the handle's
            # centre line -- instead of the centroid plus forward_offset, which
            # are then not applied. None keeps the centroid. The scoop's
            # forward_offset -0.010 was tuned on a spoon lying straight ahead;
            # turned to it (2026-10-23, +81 and +110 deg) the same offset closed
            # the jaws 8-10 mm nearer the bowl (measured tool offset 32-34 mm
            # against 41-42 when straight).
            "working_end_distance": None,
            # For the scoop: judge from the scan how it lies (no extra scan).
            # "warn": say so if it is on its side and pick anyway. True: press
            # it over first (right_spoon) -- tried on the robot 2026-10-23 and
            # never righted it, so not used. False: no check.
            "right_if_on_side": False,
            # Close to this width and stop, metres (a position command; no
            # squeezing on). None closes per force_limited. For an object whose
            # width at the jaws is known and which a force-limited close
            # crushes: the clear cups.
            "grip_width": None,
            # Clear the cameras before scanning via frankapy reset_joints
            # (home), not a hardcoded XYZ park.
            "reset_before_scan": True,
            # Optional override: if set, refuse grasps whose measured closing
            # width exceeds this (metres). Default None = no extra cap beyond
            # the gripper's physical opening. The old 55mm default was only a
            # bandaid for one bloated SAM mask (~74mm reported on a ~40mm
            # beaker); width measurement now crops to a local neighbourhood.
            "max_object_width": None,
        }
    
    def check_preconditions(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Lightweight gate before execute. Does not scan or refuse on gripper
        width: execute opens the gripper and retracts the arm before looking,
        so a stuck mid-width grasp is cleared rather than blocking the skill.
        """
        return self.validate_params(params)
    
    @staticmethod
    def _grasp_yaw(params, object_pc, grasp_type, slide):
        """(yaw in radians or 0.0, what was decided) for this pick's ``yaw`` param."""
        want = params.get("yaw")
        if want is None or grasp_type == "side" or slide > 0:
            return 0.0, None
        mode = want.strip().lower() if isinstance(want, str) else None
        if mode in ("auto", "long", "square"):
            yaw, info, why = None, None, []
            if mode in ("auto", "long"):
                yaw, info = tool_long_axis_yaw(drop_stray_points(object_pc))
                if yaw is None:
                    why.append(f"long axis: {info}")
            if yaw is None and mode in ("auto", "square"):
                # Not a long tool: a square top (the stirrer's cube), turned to
                # the nearest face.
                yaw, info = square_top_yaw(object_pc)
                if yaw is None:
                    why.append(f"square top: {info}")
                else:
                    info = {**info, "fit": "square top"}
                    print(f"[PickUp] Square top {info['top_mm'][0]:.0f} x "
                          f"{info['top_mm'][1]:.0f} mm ({info['top_points']} points): "
                          f"its faces are turned {np.degrees(yaw):+.0f}°")
            if yaw is None:
                print(f"[PickUp] yaw {mode}: {'; '.join(why)}; grasping as at home")
                return 0.0, {"yaw_auto": why}
        else:
            yaw, info = np.radians(float(want)), {"given_deg": float(want)}
        yaw = float((yaw + np.pi) % (2 * np.pi) - np.pi)
        if abs(np.degrees(yaw)) > MAX_GRASP_YAW_DEG:
            print(f"[PickUp] yaw {np.degrees(yaw):+.0f}° is past "
                  f"±{MAX_GRASP_YAW_DEG:.0f}° (the tool points back at the base); "
                  f"grasping as at home -- turn it around for a clean grasp")
            return 0.0, {**(info if isinstance(info, dict) else {}),
                         "refused_deg": round(float(np.degrees(yaw)), 1)}
        what = ("the jaws onto its faces" if isinstance(info, dict) and
                info.get("fit") == "square top" else
                f"along the tool's long axis, {np.degrees(yaw):+.0f}° from +X")
        print(f"[PickUp] Turning the wrist {np.degrees(yaw):+.0f}° about Z: {what}"
              + (f" ({info['length_mm']:.0f} mm long, bowl {info['bowl_width_mm']:.0f} "
                 f"/ handle {info['handle_width_mm']:.0f} mm wide)"
                 if isinstance(info, dict) and "length_mm" in info else ""))
        return yaw, {**(info if isinstance(info, dict) else {}),
                     "yaw_deg": round(float(np.degrees(yaw)), 1)}

    def execute(self, params: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Execute the pick up skill.
        
        Returns:
            (success, result_dict) where result_dict contains:
            - grasped_object: Name of the object that was grasped
            - grasp_pose: The pose where the grasp was executed
            - grasp_type: The type of grasp used
        """
        params = self.get_params_with_defaults(params)
        object_name = params["object_name"]
        grasp_type = params["grasp_type"]
        approach_height = params["approach_height"]
        lift_height = params["lift_height"]
        grasp_force = params["grasp_force"]
        
        print(f"[PickUp] Starting pick up of '{object_name}'")

        # 1. Drop anything held RIGHT HERE before any arm motion. A leftover
        # goto_pose from a failed previous run used to keep running until that
        # old target was reached, so open_gripper appeared to wait until then
        # and the object never fell at the start.
        print(f"[PickUp] Opening gripper at current pose (drop first)...")
        if not self.open_gripper():
            return False, {"error": "Failed to open gripper"}
        self.wait(0.4)

        # 2. Clear the cage cameras BEFORE scanning. Default: frankapy
        # reset_joints (home). Scanning with the arm over the table occludes
        # the object and pollutes the fused cloud.
        if params.get("reset_before_scan", True):
            print("[PickUp] reset_joints (home) clear of cameras before scan...")
            if not self.go_home():
                self.open_gripper()
                return False, {"error": "Failed to reset_joints before scanning"}
        else:
            retract = np.array(params.get("retract_xyz", [0.35, 0.0, 0.45]), dtype=float)
            print(f"[PickUp] Retracting clear of cameras to {np.round(retract, 3)}...")
            if not self.move_to_position(retract, maintain_orientation=True):
                self.open_gripper()
                return False, {"error": "Failed to retract before scanning"}
        self.wait(0.4)

        # 3. Fresh multi-camera scan with the arm out of the way.
        print(f"[PickUp] Localizing object (all cameras, arm at home)...")
        self.vision.clear_cache()
        located = self.vision.locate(object_name, force_refresh=True)
        
        if located is None:
            return False, {"error": f"Could not segment '{object_name}' in scene"}
        
        object_pc = located["points"]
        centroid = located["centroid"]
        dimensions = located["dimensions"]

        # 3b. A scoop lying on its side cannot be taken across its handle.
        # The scan just made says how it lies; only if it is on its side is it
        # pushed over first (right_spoon) and scanned again.
        righted = None
        mode = params.get("right_if_on_side")
        if mode and object_pc is not None:
            from .right_spoon import RightSpoonSkill, spoon_pose
            table = self.config.get("table_z")
            lies = spoon_pose(drop_stray_points(object_pc),
                              float(table) if table is not None else 0.0)
            print(f"[PickUp] How '{object_name}' lies: {lies['pose']} "
                  f"(end widths {lies.get('width_mm')} mm, heights {lies.get('height_mm')} mm)")
            if lies["pose"] == "on_side" and mode == "warn":
                print(f"[PickUp] WARNING: '{object_name}' lies ON ITS SIDE. Set it flat, "
                      f"mouth up, by hand for a clean grasp; trying the pick anyway")
            elif lies["pose"] == "on_side":
                print(f"[PickUp] '{object_name}' is on its side: righting it before the pick")
                ok, righted = RightSpoonSkill(self.robot, self.vision, self.config).execute(
                    {"object_name": object_name})
                if not ok:
                    return False, {"error": (f"'{object_name}' lies on its side and could not "
                                             f"be righted: {righted.get('error')}"),
                                   "righted": righted}
                self.open_gripper()
                if not self.clear_cameras(params, tag="PickUp"):
                    return False, {"error": "Failed to reset_joints after righting",
                                   "righted": righted}
                self.vision.clear_cache()
                located = self.vision.locate(object_name, force_refresh=True)
                if located is None:
                    return False, {"error": f"Lost '{object_name}' after righting it",
                                   "righted": righted}
                object_pc = located["points"]
                centroid = located["centroid"]
                dimensions = located["dimensions"]
            elif lies["pose"] == "upside_down":
                print(f"[PickUp] WARNING: '{object_name}' looks upside down; picking it "
                      f"anyway (righting handles only a scoop on its side)")

        confidences = located.get("confidences") or {}
        if confidences:
            print(f"[PickUp] Object segmented with confidence: {max(confidences.values()):.2f}")
        cams = located.get("cameras") or []
        print(f"[PickUp] Trusting fused median of cameras {cams} "
              f"(rejected {located.get('rejected_cameras') or []})")
        print(f"[PickUp] Object centroid: {np.round(centroid, 4)}, "
              f"dimensions: {np.round(dimensions, 4)}")
        
        # 4. Upright top-down grasp by default (no pour tip / spout align).
        print(f"[PickUp] Computing grasp pose (type: {grasp_type})...")
        if grasp_type == "side":
            candidates = self.vision.compute_grasp_candidates(
                object_pc,
                object_name,
                grasp_type="side",
                pitch_deg=params["pitch_deg"],
            )
        else:
            # "top", "auto", "handle" → single upright (or handle) pose
            gtype = "top" if grasp_type in ("auto", "top") else grasp_type
            pose = self.vision.compute_grasp_pose(
                object_pc,
                object_name,
                grasp_type=gtype,
                pitch_deg=0.0,
            )
            candidates = [pose] if pose is not None else []

        if not candidates:
            return False, {"error": "Could not compute valid grasp pose"}

        z_offset = params["z_offset"]
        forward_offset = float(params["forward_offset"])
        lateral_offset = float(params["lateral_offset"])
        slide = float(params.get("slide_in") or 0.0)
        yaw, yaw_info = self._grasp_yaw(params, object_pc, grasp_type, slide)
        grasp_pose = None
        for i, cand in enumerate(candidates):
            pose = cand.copy()
            if yaw:
                pose[:3, :3] = yaw_about_z(yaw) @ pose[:3, :3]
            if slide > 0:
                # Jaws across the slide: they close along base Y, so they
                # pass either side of the object as the hand moves along X.
                pose[:3, :3] = np.diag([1.0, -1.0, -1.0])
            tilt_away = float(params.get("tilt_away_deg") or 0.0)
            if tilt_away:
                # About world Y by -tilt: tool Z (fingers) goes from straight
                # down to down-and-forward (+X).
                a = np.radians(tilt_away)
                Ry = np.array([[np.cos(a), 0.0, -np.sin(a)],
                               [0.0, 1.0, 0.0],
                               [np.sin(a), 0.0, np.cos(a)]])
                pose[:3, :3] = Ry @ pose[:3, :3]
            if z_offset:
                pose[2, 3] = float(np.clip(
                    pose[2, 3] + z_offset,
                    self.workspace_min[2],
                    self.workspace_max[2],
                ))
            from_end = params.get("working_end_distance")
            if (yaw and from_end is not None and isinstance(yaw_info, dict)
                    and yaw_info.get("working_end_xy") is not None):
                u = np.asarray(yaw_info["axis_xy"], dtype=float)
                end = np.asarray(yaw_info["working_end_xy"], dtype=float)
                line_pt = np.asarray(yaw_info["handle_line_xy"], dtype=float)
                # On the handle's line, from_end back from the working end.
                along_end = float((end - line_pt) @ u)
                xy = line_pt + u * (along_end - float(from_end))
                was = pose[:2, 3].copy()
                pose[:2, 3] = xy
                pose[:3, 3] = self._clamp_position(pose[:3, 3])
                print(f"[PickUp] Grasp {float(from_end) * 1000:.0f}mm back from the working "
                      f"end, on the handle's line: {np.round(pose[:3, 3], 4)} "
                      f"({np.linalg.norm(xy - was) * 1000:.0f}mm from the centroid; "
                      f"forward/lateral_offset not applied)")
            elif forward_offset or lateral_offset:
                # Along and across the turned grasp; plain base X/Y at no turn.
                shifted = pose[:3, 3] + yaw_about_z(yaw or 0.0) @ np.array(
                    [forward_offset, lateral_offset, 0.0], dtype=float
                )
                pose[:3, 3] = self._clamp_position(shifted)
                axes = ("along the tool" , "across it") if yaw else ("forward(X)", "lateral(Y)")
                print(f"[PickUp] Grasp shifted by {axes[0]}={forward_offset:+.3f}m "
                      f"{axes[1]}={lateral_offset:+.3f}m -> "
                      f"{np.round(pose[:3, 3], 4)}")

            tip = pose[:3, 2]
            tip_from_vert = float(np.degrees(np.arccos(np.clip(-tip[2], -1, 1))))

            hover_pose = pose.copy()
            hover_pose[2, 3] = self.safe_height
            # Keep wrist upright on the way down when this is a top grasp.
            if tip_from_vert < 5.0:
                hover_pose[:3, :3] = pose[:3, :3]
            else:
                hover_pose = upright_from_pitched(pose)
                hover_pose[2, 3] = self.safe_height

            # Sliding in: everything before the grasp happens behind it.
            behind = pose.copy()
            behind[0, 3] -= slide
            approach_pose = behind.copy()
            approach_pose[2, 3] += approach_height
            if slide > 0:
                hover_pose[:3, 3] = [behind[0, 3], behind[1, 3], self.safe_height]

            print(f"[PickUp] Candidate {i + 1}/{len(candidates)}: "
                  f"wrist tip {tip_from_vert:.0f} deg from vertical, "
                  f"grasp {np.round(pose[:3, 3], 4)}"
                  + (f", sliding in from {slide * 1000:.0f}mm behind" if slide > 0 else ""))

            print(f"[PickUp]   hover {np.round(hover_pose[:3, 3], 4)}...")
            if not self.move_to_pose(hover_pose, reach_tol=0.05):
                print(f"[PickUp]   unreachable at hover — trying next")
                continue
            print(f"[PickUp]   approach {np.round(approach_pose[:3, 3], 4)}...")
            if not self.move_to_pose(approach_pose, reach_tol=0.04):
                print(f"[PickUp]   unreachable at approach — trying next")
                continue
            if slide > 0:
                print(f"[PickUp]   down behind it {np.round(behind[:3, 3], 4)}...")
                if not self.move_to_pose(behind, reach_tol=0.03):
                    print(f"[PickUp]   unreachable behind the grasp — trying next")
                    continue
                print(f"[PickUp]   slide forward onto it {np.round(pose[:3, 3], 4)}...")
            else:
                print(f"[PickUp]   grasp {np.round(pose[:3, 3], 4)}...")
            if not self.move_to_pose(pose, speed="slow", reach_tol=0.03):
                print(f"[PickUp]   unreachable at grasp — trying next")
                continue

            grasp_pose = pose
            print(f"[PickUp] Using candidate {i + 1} "
                  f"(wrist tip {tip_from_vert:.0f} deg from vertical)")
            break

        if grasp_pose is None:
            return False, {
                "error": f"No reachable grasp among {len(candidates)} candidates",
            }

        force_limited = bool(params["force_limited"])
        grip_width = params.get("grip_width")
        if grip_width is not None:
            force_limited = False
        # Best-effort width for logging / width-close mode. Flat objects often
        # have no cloud at grasp height — that used to abort the whole pick.
        expected = width_along_closing_axis(object_pc, grasp_pose)
        target_width = None

        if grip_width is not None:
            # A measured width to close to and stop at. The clear cups are
            # tapered, so their cylinder model's width is the rim's, not the
            # width where the jaws close, and the force-limited close kept
            # squeezing them (2026-10-08: held at 37 mm on a 61 mm section).
            target_width = float(grip_width)
            print(f"[PickUp] Closing to {target_width * 1000:.1f}mm and stopping "
                  f"(grip_width)...")
            if not self.close_gripper(
                force=grasp_force, target_width=target_width, force_limited=False
            ):
                return False, {"error": "Failed to close gripper"}
        elif force_limited:
            print(f"[PickUp] Closing until contact "
                  f"(force_limited, max {grasp_force:.1f} N)"
                  + (f"; cloud width ~{expected * 1000:.1f}mm"
                     if expected is not None else
                     "; no cloud width at grasp height"))
            if not self.close_gripper(
                force=grasp_force, force_limited=True
            ):
                return False, {"error": "Failed to close gripper"}
        else:
            # Soft cups: no useful contact force, so aim for diameter − squeeze.
            if expected is None:
                return False, {
                    "error": "Could not measure object width at grasp height; "
                             "refusing to close blind on a crushable object "
                             "(set force_limited=True to close on contact instead)",
                }

            max_w = params.get("max_object_width")
            if max_w is not None and expected > float(max_w):
                self.open_gripper()
                return False, {
                    "error": (
                        f"Measured width {expected * 1000:.1f}mm > "
                        f"max_object_width {float(max_w) * 1000:.0f}mm — "
                        f"segmentation is probably bloated; refusing false grasp"
                    ),
                    "expected_width": expected,
                }

            gripper_max = 0.08
            if expected > gripper_max:
                self.open_gripper()
                return False, {
                    "error": (
                        f"Measured width {expected * 1000:.1f}mm > gripper opening "
                        f"{gripper_max * 1000:.0f}mm"
                    ),
                    "expected_width": expected,
                }

            target_width = max(0.0, expected - params["squeeze"])
            print(f"[PickUp] Object ~{expected * 1000:.1f}mm across; "
                  f"closing to {target_width * 1000:.1f}mm "
                  f"(diameter - {params['squeeze'] * 1000:.0f}mm)...")

            if not self.close_gripper(
                force=grasp_force, target_width=target_width, force_limited=False
            ):
                return False, {"error": "Failed to close gripper"}

        self.wait(0.3)

        # Where the TCP actually is with the jaws closed. move_to_pose accepts
        # the grasp anywhere within 30mm, so the commanded pose is not this:
        # on 2026-10-05 the stirrer's put-back, aimed at the commanded grasp,
        # met the holder 13mm early. Putting a tool back aims at THIS.
        grasp_tcp_reached = np.asarray(self.get_current_pose().translation,
                                       dtype=float)
        off = float(np.linalg.norm(grasp_tcp_reached - grasp_pose[:3, 3]))
        if off > 0.005:
            print(f"[PickUp] Closed at {np.round(grasp_tcp_reached, 4)}, "
                  f"{off * 1000:.1f}mm from the commanded grasp "
                  f"{np.round(grasp_pose[:3, 3], 4)}")

        gripper_width = self.get_gripper_width()
        grasped = self.gripper_is_grasped()
        print(f"[PickUp] Holding at {gripper_width * 1000:.1f}mm "
              f"(gripper_is_grasped={grasped}"
              + (f", target {target_width * 1000:.1f}mm" if target_width is not None
                 else "")
              + ")")

        if gripper_width < 0.003:
            self.open_gripper()
            return False, {
                "error": "Gripper closed on nothing",
                "gripper_width": gripper_width,
                "expected_width": expected,
                "target_width": target_width,
            }
        if (not force_limited and expected is not None
                and gripper_width < expected * params["crush_fraction"]):
            print("[PickUp] Crush detected — opening gripper to release")
            self.open_gripper()
            return False, {
                "error": (f"Gripper closed to {gripper_width * 1000:.1f}mm on a "
                          f"{expected * 1000:.1f}mm object: crushed it"),
                "gripper_width": gripper_width,
                "expected_width": expected,
                "target_width": target_width,
            }
        # Jaws barely left fully open and no grasp contact → empty/missed.
        if gripper_width > 0.07 and not grasped:
            self.open_gripper()
            return False, {
                "error": (
                    f"Gripper still {gripper_width * 1000:.1f}mm open "
                    f"(not grasped) — did not close on the object"
                ),
                "gripper_width": gripper_width,
                "expected_width": expected,
                "target_width": target_width,
            }
        
        # Step 11: Lift object
        print(f"[PickUp] Lifting object...")
        lift_pose = grasp_pose.copy()
        lift_pose[2, 3] += lift_height
        if not self.move_to_pose(lift_pose):
            print("[PickUp] Lift failed — opening gripper to release")
            self.open_gripper()
            return False, {"error": "Failed to lift object"}

        # Re-check after the lift: an object can slip out on the way up.
        # Floor matches the "closed on nothing" check above (0.003m) — the
        # 8mm floor this used to use was well within the range of a thin
        # held object (e.g. a spoon's neck), so a low-force grasp that
        # settled there was misreported as lost even while still holding it.
        lifted_width = self.get_gripper_width()
        if lifted_width < 0.003:
            self.open_gripper()
            return False, {
                "error": (f"Object lost during the lift: gripper went from "
                          f"{gripper_width * 1000:.1f}mm to "
                          f"{lifted_width * 1000:.1f}mm"),
                "gripper_width": lifted_width,
                "expected_width": expected,
            }
        if (not force_limited and expected is not None
                and lifted_width < expected * params["crush_fraction"]):
            print("[PickUp] Crush after lift — opening gripper to release")
            self.open_gripper()
            return False, {
                "error": (f"Object crushed during the lift: "
                          f"{lifted_width * 1000:.1f}mm vs expected "
                          f"{expected * 1000:.1f}mm"),
                "gripper_width": lifted_width,
                "expected_width": expected,
            }

        # Measure the tool from where the jaws closed, not where they were
        # sent: the tool is held THERE. On the robot they close 5-6 mm short
        # of the command along x every time (4.7, 6.3, 4.9 mm, 2026-10-07/08),
        # and an offset taken from the command puts the bowl that much nearer
        # the jaws than it is -- the scoop then digs that much further forward.
        # The sim reaches its command, so there the two are the same.
        held_frame = np.array(grasp_pose, dtype=float).copy()
        held_frame[:3, 3] = grasp_tcp_reached
        suggested = measure_tool_offset(object_pc, held_frame)
        if suggested is not None:
            print(f"[PickUp] Measured tool offset (TCP -> working end, tool "
                  f"frame): [{suggested[0]:.3f}, {suggested[1]:.3f}, "
                  f"{suggested[2]:.3f}] m")
            print(f"[PickUp]   x/y are measured and reliable; it changes "
                  f"whenever the grasp moves.")
            print(f"[PickUp]   z={suggested[2]:.3f} is a LOWER BOUND — the "
                  f"cameras look down, so the underside of a bowl is occluded "
                  f"and the cloud stops at the rim. Use the CAD depth if you "
                  f"have it, or the tool will dig too shallow.")

        print(f"[PickUp] Successfully picked up '{object_name}'")
        
        # The object has moved, so any cached segmentation is now stale.
        self.vision.clear_cache()
        
        result = {
            "grasped_object": object_name,
            "grasp_pose": grasp_pose.tolist(),
            "grasp_tcp_reached": grasp_tcp_reached.tolist(),
            "grasp_type": grasp_type,
            # How far behind it the jaws came down (0: from above). Place
            # leaves the same way, sliding the open jaws back out.
            "slide_in": slide,
            # How far the grasp was moved down (-) from the computed one;
            # a put-back sets the object down that much lower to match.
            "z_offset": float(z_offset or 0.0),
            "gripper_width": lifted_width,
            "expected_width": expected,
            "target_width": target_width,
            "grasp_force": grasp_force,
            "grasp_yaw": yaw_info,
            "righted": righted,
            "suggested_tool_offset": (None if suggested is None
                                      else [float(v) for v in suggested]),
            "tool_extent_x": tool_extent_x(object_pc, held_frame),
            # The held object's own shape, from the cloud it was picked by:
            # pour needs where a held cup's rim and bottom are relative to the
            # jaws to keep its lip over the target (SkillsExecutor fills it in).
            "object_top_z": float(np.percentile(np.asarray(object_pc)[:, 2], 99)),
            "object_bottom_z": float(np.percentile(np.asarray(object_pc)[:, 2], 1)),
            "object_radius": float(np.percentile(np.linalg.norm(
                np.asarray(object_pc)[:, :2] - np.asarray(centroid)[:2], axis=1), 97)),
            "centroid": centroid.tolist(),
            "dimensions": dimensions.tolist() if dimensions is not None else None,
        }
        return True, result
