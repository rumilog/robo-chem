"""
Scoop Skill

Transfers a measured quantity of powder out of a source container with a held
scoop. This is how dry reagents are metered — the profile in robomail_Aliyah's
`config/robot_profile.py` explicitly rules out metering a pour by weight.

``powder_source`` should be the **reagent label** on the paper under the cup
(e.g. ``"citric acid"``, ``"baking soda"``), not a generic ``"white paper cup"``.
Vision segments every white paper cup, reads the labels with a VLM, then scoops
from the matching instance.

The stroke (rewritten 2026-09-21, reshaped 2026-09-23):

    descend ALREADY TILTED (60 deg) into the back of the dish  ->  drive the
    bowl FORWARD while it rolls to level, finishing at the front  ->  lift
    straight up, then turn slightly nose-up to carry the load

The tip is held a fixed height above the dish's INSIDE floor the whole way
(``tip_floor_gap``), so the powder surface never sets how deep it goes; the arm
backs off as the wrist rolls so the tip keeps that height.

**Why this shape.** Powder enters a bowl through its mouth, and only where the
mouth is moving into it. A steep bowl faces its mouth forward, so travelling
forward at depth drives the powder straight in; a level one faces up and just
pushes. The first rewrite pivoted almost in place (8 mm of travel) and, by the
geometric estimate in robochem/sim/powder.py, half-filled the bowl at best.
Before that, a 40 mm push with no regard for where the walls were logged 49785
bowl-to-cup contacts in a 69 mm cup.

**Staying off the dish.** The stroke is placed, not assumed: at every waypoint
each corner of the bowl must stay ``wall_clearance`` inside the dish radius,
and the longest travel that allows is used unless ``sweep_advance`` is given.
That puts the steep entry as far back and the level finish as far forward as
the dish allows. The floor is a hard limit on depth; so is the rim, which the
handle must clear as the bowl levels.

Points kept from the old stroke because each was paid for with a hardware run:
  - reset_joints before scanning, world-frame rim geometry from the pointcloud
  - the descent and sweep run *compliant* (impedance on) because they are
    contact motions; the free-space approach and the lift run stiff and verified
  - the entry tilt is confirmed by measurement before the powder is touched
  - the sweep is discrete blocking waypoints (see P11 in SKILLS_ROADMAP.md)
  - held width is re-checked at every phase boundary

One thing that got *better* rather than being kept. The old code confirmed its
bite angle with ``tool_tip_deg()``, an arccos magnitude that is always >= 0 and
so cannot tell "tilted forward" from "tilted backward" — it shipped a
wrong-direction tilt to the bench reading as a clean success, and needed a
hand-tuned negation to work around. ``tool_long_axis_deg()`` is not
direction-blind: it reads 90° at level, 45° at a 45° nose-down bite and 105° at
15° nose-up, so one measurement now checks magnitude AND sign.
"""

from typing import Dict, Any, Tuple
import numpy as np

from .base_skill import BaseSkill, _orthonormalize, tool_y_delta


class ScoopSkill(BaseSkill):
    """
    Scoop powder from a source container with a held scoop or spoon.

    Pipeline:
    1. Record held width; the scoop must survive the whole motion
    2. Clear the cameras, scan the source, measure rim height / radius
    3. Check the swept envelope fits the opening BEFORE moving
    4. Approach above the powder already at the bite angle, and confirm that
       angle — direction included — while still in free space
    5. Descend straight down into the bed, tilted, compliant
    6. Sweep: roll nose-down -> nose-up about a nearly stationary bowl centre
    7. Lift straight up, keeping the cup angle so the load stays in
    """

    name = "scoop"
    required_params = ["powder_source"]

    @property
    def optional_params(self) -> Dict[str, Any]:
        return {
            # --- how deep: measured from the FLOOR (default) ----------------
            # Height of the head's lowest point above the container's INSIDE
            # floor for the whole dig, metres. The head descends until its tip
            # is this far off the floor, then rolls 45 -> 0 deg holding it
            # there. The powder surface plays no part: how full the cup is
            # changes how much powder is above the head, never where the head
            # goes, so a mis-measured or unknown surface cannot drive it into
            # the floor. Set None to measure depth from the surface instead
            # (scoop_depth / depth_reference below).
            #
            # It must exceed how far the real motion strays below the plan. In
            # simulation, from a checkpoint of this stroke, the tip stays within
            # ~0.3 mm of plan at --speed 2, 3 and 5 now that the streamed path
            # has velocity feed-forward (SimFrankaArm.follow_pose_path); before
            # that the servo lag put it 1.5 mm low at --speed 2 and 6.7 mm
            # through the floor at --speed 5. On the robot, cover depth noise
            # and calibration error too.
            "tip_floor_gap": 0.003,
            # --- or measured from the powder surface ------------------------
            # Only used when tip_floor_gap is None.
            # How far below the powder surface the spoon head goes, metres.
            # This is the key parameter: it is what decides whether powder is
            # actually collected. Everything else about depth — the cup floor,
            # the rim — is a hard limit on it, not a second knob.
            #
            # Measured at the point named by depth_reference. With the default
            # "tip" it is the head's lowest point — the leading edge whenever
            # the head is nose-down or level, which is every angle it takes
            # inside the cup. The log prints the mouth and top depth too.
            "scoop_depth": 0.015,
            # Which point of the head scoop_depth is measured at:
            #   "tip"   — lowest point (default). The depth and the floor limit
            #             then concern the SAME point, so the bite angle cannot
            #             make the head hit the floor: the only condition is
            #             scoop_depth <= bed depth - floor_clearance. The price
            #             is that the opening is not controlled — it sits
            #             (bowl_length/2)*sin(tilt) + bowl_depth*cos(tilt)
            #             above the tip, 17.9mm at 45 deg for the printed
            #             scoop, so check the mouth depth in the log. A 15mm
            #             tip at 45 deg leaves the opening above the surface.
            #   "mouth" — centre of the opening: powder can enter
            #   "top"   — highest point of the head: the whole head is buried
            # "mouth" and "top" need bowl_length and bowl_depth, and are the
            # ones auto_tilt serves: a steep head stands taller.
            "depth_reference": "tip",
            # Where the powder surface actually is, in world z. Leave it None
            # and the surface is taken as the measured top of the container —
            # but that top is the 97th percentile of the CONTAINER's cloud,
            # i.e. the RIM. For a nearly-full cup those coincide; for a
            # part-full one they do not, and the whole stroke then runs in the
            # air above the powder while still reporting success. Pass it when
            # anything knows it (a fill estimate, a taught value, a probe).
            "powder_surface_z": None,

            # --- the bite ------------------------------------------------
            # Nose-down angle the bowl ENTERS the powder at, held through the
            # descent. Steep, so the mouth faces forward into the bed and the
            # forward stroke drives powder straight into it; a shallow bite
            # presents the mouth upward and mostly pushes. The bowl must enter
            # already tilted: dropping in level and tilting afterwards just
            # packs the powder flat.
            "dig_tilt_deg": 60.0,
            # Nose-UP angle the bowl ends at. This is what carries the load —
            # finishing level lets powder slide off the front on the way up.
            #
            # It is reached during the LIFT, not inside the cup. Rolling
            # nose-up while the bowl is still down in a shallow dish swings
            # the HANDLE into the near wall: the handle end sits ~40mm back
            # along the tool, so at -15 deg it reaches 46mm behind the bowl and
            # 34.5mm is all the cup has. Measured, that was 25109 contacts and
            # a cup shoved 7.8mm. Holding the roll until the bowl is clear of
            # the rim costs nothing and is what a hand does anyway.
            "cup_tilt_deg": 15.0,
            # Angle the IN-CUP sweep finishes at. 0 is level, positive is still
            # nose-down. This is the one that has to fit inside the container,
            # so it is separate from cup_tilt_deg: raise it for a shallow dish
            # where even level does not clear, lower it toward a negative value
            # only in a container deep enough to take the handle.
            "sweep_end_tilt_deg": 0.0,
            # How the roll is spread over the stroke. 1.0 (default) rolls in
            # step with the forward travel -- the tilt falls steadily as the
            # bowl moves forward, reaching level at the front. Higher holds
            # the bite angle through the first part of the stroke and rolls
            # late; lower rolls early.
            "roll_late": 1.0,

            # --- the sweep ------------------------------------------------
            # How far the bowl travels FORWARD (along the way it points, away
            # from the base) while it rolls from the bite angle to level.
            # None (default) = as far as the dish allows: the stroke is placed
            # to enter as far BACK as the steep bowl fits and finish as far
            # FORWARD as the level bowl fits, each wall_clearance off the
            # inside wall -- 46 mm in the 86 mm dish. That travel, with the
            # mouth facing forward, is what fills the bowl; the old 8 mm
            # default pivoted almost in place and collected little. A number
            # is used as given, centred in the dish, and still checked.
            "sweep_advance": None,
            # How far the head is allowed to rise across the sweep, metres. 0
            # holds scoop_depth for the whole roll. The rim can still force it
            # up (see handle_clearance) — that is reported, not hidden.
            "sweep_rise": 0.0,
            # Gap kept between the RIM and the point where the handle leaves
            # the bowl's crank, metres.
            #
            # The scoop is a lever: pinning the bowl in place makes the handle
            # swing on an arc instead. Rolling from a 45 deg bite to 15 deg
            # nose-up drops that pivot by 16mm for the printed scoop, which is
            # enough to sweep the handle into the near wall of a 31mm dish —
            # measured as 13760 contacts and a cup shoved 5.5mm. So the bowl is
            # RAISED through the roll by exactly enough to hold the pivot above
            # the rim, which is also what a hand does: it lifts as it turns.
            #
            # Raise this if the crank still clips; lower it (or set 0) for a
            # deep container where the handle has room inside.
            "handle_clearance": 0.005,
            # How far the tool reaches BACK from the TCP along the handle,
            # metres — the distance from the jaws to the far end of the handle.
            #
            # This is what the clearance is actually computed against, because
            # the handle END is what swings lowest and furthest as the wrist
            # rolls; guarding only the TCP is what let v4 clip the wall with
            # the pivot dutifully 5mm above the rim. The printed scoop is about
            # 0.015 from a mid-grip grasp. Left at 0 only the TCP is protected.
            "tool_back_reach": 0.0,
            # Poses the sweep is cut into. In simulation more is smoother and
            # free. ON HARDWARE each is a separate frankapy skill that
            # decelerates to a stop, so a high count turns a smooth roll into a
            # visible staircase (the lesson in P11). Start at 6-8 on the robot.
            "sweep_waypoints": 12,
            # Slow enough to watch the roll: the sweep is min-jerk, so its peak
            # speed is ~1.9x the average (the 100mm dish's 64mm of tip travel
            # and 60 deg of roll over 5s peak near 24mm/s and 22 deg/s).
            "sweep_seconds": 5.0,

            # --- approach and exit ----------------------------------------
            "approach_height": 0.10,
            # Seconds for the tilted move to above the bed, then straight down
            # into it. Each is divided by the arm's speed like any duration.
            "approach_seconds": 4.0,
            "descend_seconds": 3.0,
            "lift_height": 0.12,
            "lift_seconds": 3.5,
            # Roll back to level during the lift. FALSE by default: the cup
            # angle is the only thing holding the powder on a shallow bowl, and
            # levelling while still over the cup drops it straight back in.
            # `dump` homes before it tips, so it does not need a level hand-off.
            "level_on_lift": False,

            # --- the cup floor: first KNOW where it is, then just miss it ---
            # World z of the container's INSIDE floor — the surface the powder
            # rests on. Pass it whenever anything knows it (the simulator's
            # ground truth, a taught value for a standard dish).
            #
            # Perception cannot measure this: under a powder bed the inside
            # floor is invisible, so the lowest points of the container's cloud
            # are its OUTER bottom, by the table. In simulation base_z read
            # 3.0mm while the inside floor is at 4.0mm, and planning against
            # base_z would put the tip 1mm through the floor.
            "container_floor_z": None,
            # When container_floor_z is not given, the inside floor is taken as
            # the measured outer bottom (base_z) plus this. MEASURE IT for the
            # container in use — it is the floor thickness plus whatever the
            # outer bottom estimate reads low. The simulated reagent dish is
            # 4mm.
            "floor_thickness": 0.004,
            # Gap kept between the lowest point of the head and the inside
            # floor. This is NOT a guess about where the floor is — that is
            # container_floor_z / floor_thickness — it only absorbs how far
            # the arm strays from its plan, so the head does not touch.
            "floor_clearance": 0.001,
            # If scoop_depth cannot be reached at dig_tilt_deg without touching
            # the floor — a steep head needs more vertical room than a shallow
            # bed has — lower the bite angle to the steepest one that fits,
            # rather than giving up depth. The depth is the point of the
            # stroke; the angle is a means to it. False fails instead.
            # Only acts for depth_reference "mouth"/"top": with "tip" the
            # depth and the floor are the same point, no angle helps, and the
            # tip is simply stopped at the floor limit with the bite unchanged.
            "auto_tilt": True,

            # --- staying off the wall --------------------------------------
            # The bowl's CIRCUMSCRIBED width, metres — used for the wall
            # envelope, because in a round dish the bowl's corners meet the
            # wall before its leading face does. Left at 0 the check protects
            # a point and the bowl can still clip the wall; it warns when that
            # happens. The printed scoop is 0.0343.
            "tool_span": 0.0,
            # The bowl's outside width across the handle, metres. With the
            # length and depth it is the footprint the stroke is fitted
            # inside the dish with. Derived from tool_span and bowl_length
            # when unset. The printed scoop is 0.0205.
            "bowl_width": None,
            # INSIDE radius of the dish, metres. Pass it when it is known (a
            # taught value; the simulator's truth). Unset, the opening radius
            # perception measured is used, and that reads the OUTSIDE of the
            # rim plus the one-sided centre error -- 39 mm for a 32.5 mm-inside
            # dish in simulation -- which a stroke sized to the room would
            # use to reach the real wall.
            "container_radius": None,
            # World [x, y] of the dish centre, when something knows it better
            # than perception (a taught position; the simulator's truth). The
            # stroke is placed about this point, so an error here moves the
            # bowl straight toward a wall: perception's centre leans toward
            # the cameras that see the dish, by 4 mm for a 69 mm cup and by
            # 16 mm (9.5 mm in x, 13 mm in y) for the 90 mm one in simulation,
            # which put the bowl into the wall. forward_offset / lateral_offset
            # still apply on top.
            "container_center": None,
            # The bowl's length along the handle, metres — NOT the same as
            # tool_span. This one sets how far the leading edge drops below
            # the floor reference when the bowl tilts, so using the
            # circumscribed width here (which is larger) lifts the whole
            # stroke and digs shallow. The printed scoop is 0.0275. Falls
            # back to tool_span when unset.
            "bowl_length": 0.0,
            # Inside depth of the bowl, floor to mouth, metres. Powder does
            # not climb into a bowl whose MOUTH is above the bed: the stroke
            # has to bury the opening, not just the edge. Measured in
            # simulation, a 15mm dig left the mouth 5mm proud of the surface
            # and collected nothing. The printed scoop is 0.0115.
            "bowl_depth": 0.0,
            # DEPRECATED alias: folded into scoop_depth with
            # depth_reference="mouth". Accepted so older commands keep working.
            "immersion": None,
            # Clearance kept between the swept envelope and the measured wall.
            "wall_clearance": 0.008,
            # Refuse rather than scrape if the envelope does not fit. The whole
            # point of this rewrite is that the cup must not be moved; a stroke
            # that does not fit is a stroke that should not run.
            "require_clearance": True,

            # --- siting, same convention as pour / pick_up -----------------
            # +X is forward from the base toward the workspace, +Y the robot's
            # left. The sweep is otherwise centred on the measured rim.
            "forward_offset": 0.0,
            "lateral_offset": 0.0,

            # --- the tool --------------------------------------------------
            # Offset from the gripper TCP to the scoop's BOWL FLOOR, in metres.
            # MEASURE THIS for your scoop. Left at 0 the arm digs with the
            # gripper itself, which is wrong for any tool of real length.
            #
            # A CRANKED tool needs the full vector, not just a depth: pass
            # tool_offset [x, y, z] in the TOOL frame (x along the handle,
            # z down). The printed scoop is about [0.025, 0, 0.029] for a
            # mid-grip grasp. tool_length alone puts the bowl tens of
            # millimetres off target once the bite tilt is applied — it was
            # 0 granules vs 84 in simulation. See BaseSkill.resolve_tool_offset.
            "tool_length": 0.0,
            "tool_offset": None,

            # --- tolerances -------------------------------------------------
            "hover_tol": 0.05,
            # The descent and sweep are contact motions: the powder resists, so
            # the arm legitimately stops short and a tight tolerance would fail
            # every successful scoop. Only a gross miss is a failure.
            "contact_tol": 0.05,
            "tilt_tol_deg": 12.0,
            # frankapy often needs a second attempt before the wrist tracks a
            # commanded orientation; pour and dump retry the same way.
            "tilt_retries": 3,

            "reset_before_scan": True,
            # SAM category to search when the target is identified by a written
            # label. The label path only considers instances of this category,
            # so a labelled CLEAR cup is invisible under the default "white
            # paper cup" and the query quietly finds nothing.
            "container_category": None,
        }

    def check_preconditions(self, params: Dict[str, Any]) -> Tuple[bool, str]:
        valid, msg = self.validate_params(params)
        if not valid:
            return False, msg

        if not self.is_gripper_holding():
            return False, "Gripper is not holding anything. Pick up a scoop/spoon first."

        return True, "Preconditions met"

    # ------------------------------------------------------------------
    # geometry helpers
    # ------------------------------------------------------------------

    def execute(self, params: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        # Which depth keys the CALLER set, before defaults hide the difference:
        # asking for a surface depth explicitly must not be overridden by the
        # floor-gap default.
        asked = {k for k, v in params.items() if v is not None}
        floor_mode = ("tip_floor_gap" in asked
                      or ("tip_floor_gap" not in params
                          and not asked & {"scoop_depth", "depth_reference", "immersion"}))
        params = self.get_params_with_defaults(params)
        source = params["powder_source"]

        scoop_depth = float(params["scoop_depth"])
        dig_tilt_deg = float(params["dig_tilt_deg"])
        cup_tilt_deg = float(params["cup_tilt_deg"])
        sweep_end_tilt_deg = float(params["sweep_end_tilt_deg"])
        back_reach = max(0.0, float(params["tool_back_reach"]))
        roll_late = max(0.1, float(params["roll_late"]))
        sweep_advance = (None if params.get("sweep_advance") is None
                         else max(0.0, float(params["sweep_advance"])))
        sweep_rise = float(params["sweep_rise"])
        waypoints = max(2, int(params["sweep_waypoints"]))
        sweep_seconds = max(0.5, float(params["sweep_seconds"]))
        tool_offset = self.resolve_tool_offset(params)
        tool_span = max(0.0, float(params["tool_span"]))
        bowl_length = max(0.0, float(params["bowl_length"]))
        bowl_depth = max(0.0, float(params["bowl_depth"]))
        if params.get("bowl_width") is not None:
            bowl_width = max(0.0, float(params["bowl_width"]))
        elif tool_span > 0.0 and bowl_length > 0.0 and tool_span > bowl_length:
            # tool_span is the circumscribed width: the diagonal of length x width.
            bowl_width = float(np.sqrt(tool_span ** 2 - bowl_length ** 2))
        else:
            bowl_width = 0.0
        depth_reference = str(params["depth_reference"]).lower()
        if depth_reference not in ("mouth", "top", "tip"):
            return False, {"error": (f"depth_reference must be 'mouth', 'top' or "
                                     f"'tip', got {params['depth_reference']!r}")}
        if params.get("immersion") is not None:
            # The old two-knob scheme (an edge depth plus a separate mouth
            # burial) is one knob now; honour the old name rather than drop it.
            scoop_depth = max(0.0, float(params["immersion"]))
            depth_reference = "mouth"
            print(f"[Scoop] NOTE: 'immersion' is now scoop_depth with "
                  f"depth_reference='mouth'; using scoop_depth="
                  f"{scoop_depth * 1000:.1f}mm at the mouth.")
        if depth_reference in ("mouth", "top") and bowl_depth <= 0.0:
            print(f"[Scoop] NOTE: depth_reference='{depth_reference}' needs "
                  f"bowl_depth; without it the mouth is taken to sit on the "
                  f"head's floor, so the real opening rides shallower than "
                  f"asked. The printed scoop is 0.0115.")
        wall_clearance = float(params["wall_clearance"])
        contact_tol = float(params["contact_tol"])
        tilt_tol = float(params["tilt_tol_deg"])
        retries = max(1, int(params["tilt_retries"]))

        # The push stroke is gone; say so rather than silently ignoring keys
        # that used to change the motion completely.
        for gone in ("dig_advance", "drag_distance", "untilt_over",
                     "push_seconds", "dip_depth", "scoop_distance",
                     "drag_waypoints", "seconds_per_waypoint"):
            if params.get(gone) is not None:
                print(f"[Scoop] NOTE: '{gone}' is no longer used. The stroke is "
                      f"now a roll in place, not a push through the bed — see "
                      f"sweep_advance / cup_tilt_deg / roll_late.")

        print(f"[Scoop] Starting scoop from '{source}'")

        start_width = self.held_width()
        print(f"[Scoop] Holding scoop at {start_width * 1000:.1f}mm")

        # 1. Clear the cameras and scan the source.
        if not self.clear_cameras(params, tag="Scoop"):
            return False, {"error": "Failed to clear the cameras before scanning"}

        self.vision.clear_cache()
        located = self.locate_container(source, force_refresh=True,
                                        category=params.get("container_category"))
        if located is None:
            return False, {"error": f"Cannot locate '{source}'"}

        ok, msg = self.check_still_holding(start_width, tag="Scoop")
        if not ok:
            return False, {"error": f"Lost the scoop before scooping: {msg}"}

        rim_center = located["rim_center"]
        rim_radius = located["rim_radius"]
        base_z = located["base_z"]
        # Print base_z and height too: a container that segments as a few
        # millimetres tall is almost always the PAPER LABEL under it rather
        # than the container itself, and the top alone hides that.
        print(f"[Scoop] '{source}' top z={located['top_z']:.4f}, "
              f"base z={base_z:.4f}, height {located['height'] * 1000:.0f}mm, "
              f"centre {np.round(rim_center, 4)}, "
              f"opening radius {rim_radius * 1000:.0f}mm")
        if located["height"] < 0.02:
            print(f"[Scoop] WARNING: that is only "
                  f"{located['height'] * 1000:.0f}mm tall. A real cup is "
                  f"~90mm. This is probably the paper label, not the "
                  f"container — check the mask before trusting the depth.")

        # 2. Where the powder actually is. The measured top is the RIM unless
        # somebody tells us otherwise, so a part-full cup digs in mid-air
        # without this.
        surface_override = params.get("powder_surface_z")
        if surface_override is not None:
            surface_z = float(surface_override)
            print(f"[Scoop] Powder surface given as z={surface_z:.4f} "
                  f"(measured container top was {located['top_z']:.4f})")
        else:
            surface_z = float(located["top_z"])

        # Where the INSIDE floor is. Know it first; after that the only rule
        # about the floor is not to touch it. base_z cannot stand in for it:
        # under a powder bed the cameras never see the inside floor, so the
        # lowest points of the container's cloud are its outer bottom.
        if params.get("container_floor_z") is not None:
            floor_z = float(params["container_floor_z"])
            floor_source = "container_floor_z"
        else:
            floor_z = base_z + float(params["floor_thickness"])
            floor_source = (f"base_z {base_z:.4f} + floor_thickness "
                            f"{float(params['floor_thickness']) * 1000:.1f}mm")
        floor_clearance = max(0.0, float(params["floor_clearance"]))
        print(f"[Scoop] Inside floor z={floor_z:.4f} ({floor_source}); powder "
              f"surface z={surface_z:.4f}: a {(surface_z - floor_z) * 1000:.1f}mm "
              f"bed")
        if surface_z - floor_z <= floor_clearance:
            return False, {
                "error": (f"No powder to scoop: the surface z={surface_z:.4f} is "
                          f"not above the inside floor z={floor_z:.4f}"),
            }
        if floor_z + floor_clearance < self.workspace_min[2]:
            return False, {
                "error": (f"The inside floor z={floor_z:.3f} is below the "
                          f"workspace floor {self.workspace_min[2]:.3f}"),
            }

        # How deep. From the floor (the default): hold the head's lowest point
        # tip_floor_gap above the inside floor. It is expressed below as the
        # equivalent tip depth under surface_z, and surface_z then cancels out
        # of the plan exactly — the head goes to floor + gap whatever the
        # surface is, or whether it was measured at all.
        tip_floor_gap = None
        if floor_mode and params.get("tip_floor_gap") is not None:
            tip_floor_gap = float(params["tip_floor_gap"])
            if tip_floor_gap < floor_clearance:
                print(f"[Scoop] NOTE: tip_floor_gap {tip_floor_gap * 1000:.1f}mm is "
                      f"under floor_clearance; using "
                      f"{floor_clearance * 1000:.1f}mm.")
                tip_floor_gap = floor_clearance
            depth_reference = "tip"
            scoop_depth = surface_z - (floor_z + tip_floor_gap)
            print(f"[Scoop] Depth from the floor: the head's lowest point stays "
                  f"{tip_floor_gap * 1000:.1f}mm above the inside floor "
                  f"(z={floor_z + tip_floor_gap:.4f}) through the whole dig. The "
                  f"powder surface does not set it.")
        else:
            print(f"[Scoop] Depth from the surface: {scoop_depth * 1000:.1f}mm at "
                  f"the {depth_reference}.")

        # 3. Siting. The sweep is centred on the rim so the bowl pivots about
        # the middle of the cup, which is what keeps it off both walls.
        centre_xy = np.asarray(rim_center, dtype=float)
        if params.get("container_center") is not None:
            given = np.asarray(params["container_center"], dtype=float)[:2]
            print(f"[Scoop] Dish centre given as {np.round(given, 4)}; perception "
                  f"measured {np.round(centre_xy, 4)}, "
                  f"{np.linalg.norm(centre_xy - given) * 1000:.1f}mm away.")
            centre_xy = given
        site = centre_xy + np.array([
            float(params["forward_offset"]), float(params["lateral_offset"])])
        if not np.allclose(site, centre_xy):
            print(f"[Scoop] Sweep site shifted by "
                  f"forward(X)={float(params['forward_offset']):+.3f}m "
                  f"lateral(Y)={float(params['lateral_offset']):+.3f}m: "
                  f"centre {np.round(centre_xy, 4)} -> {np.round(site, 4)}")

        level_rotation = self.tool_down_rotation()

        def rotation_at(angle_deg: float) -> np.ndarray:
            """Wrist rotation for a given nose-down tilt, 0 = level."""
            return _orthonormalize(level_rotation @ tool_y_delta(-float(angle_deg)))

        def long_axis_for(angle_deg: float) -> float:
            """What tool_long_axis_deg() should read at this tilt."""
            return 90.0 - float(angle_deg)

        # 4. Plan the sweep.
        #
        # Angles run from the bite (+dig_tilt, nose down) to sweep_end_tilt,
        # and the head's floor reference creeps forward by sweep_advance. Its
        # HEIGHT comes from the one depth that matters: the depth_reference
        # point of the head goes scoop_depth below the powder surface. Two hard
        # limits then apply, and depth gives way to them — never the reverse:
        #   - the lowest point of the head stays floor_clearance above the
        #     inside floor, so the cup floor is never touched, and
        #   - the back of the handle stays handle_clearance above the rim, so
        #     the handle is never dragged across the near wall. The scoop is a
        #     lever: as the head rolls level about a fixed point, the handle
        #     swings down, and in a shallow dish it would hit the rim.
        # If the floor is what stops the head reaching scoop_depth, that is the
        # bite being too steep for the bed (a steep head stands taller), so
        # auto_tilt lowers the bite until the depth fits.
        rim_top = float(located["top_z"])
        handle_clearance = float(params["handle_clearance"])
        a_off, b_off = float(tool_offset[0]), float(tool_offset[2])
        # The head's own length, not the circumscribed tool_span: they differ
        # by 25% on the printed scoop, and the larger one misplaces the tip.
        dip_length = bowl_length if bowl_length > 0.0 else tool_span
        half_len = dip_length / 2.0
        head_corners = [np.array([sx * half_len, 0.0, sz])
                        for sx in (-1.0, 1.0)
                        for sz in ((0.0, -bowl_depth) if bowl_depth > 0.0 else (0.0,))]
        handle_end = np.array([-(a_off + back_reach), 0.0, -b_off])

        def head_heights(angle_deg: float) -> Tuple[float, float, float, float]:
            """
            Heights of the head's (tip, mouth centre, top, reference point)
            above its floor reference at this tilt. Tool z points down and the
            mouth faces tool -z, so rotating corners through rotation_at gives
            world heights directly; nothing here assumes which way tool x
            points in the world.
            """
            R = rotation_at(angle_deg)
            zs = [float((R @ c)[2]) for c in head_corners]
            tip, top = min(zs), max(zs)
            mouth = float((R @ np.array([0.0, 0.0, -bowl_depth]))[2])
            ref = {"tip": tip, "top": top, "mouth": mouth}[depth_reference]
            return tip, mouth, top, ref

        def handle_above_floor(angle_deg: float) -> float:
            return float((rotation_at(angle_deg) @ handle_end)[2])

        # The dish in the plane: centred on the (trimmed) site, with the inside
        # radius when it is known and the measured opening otherwise.
        if params.get("container_radius") is not None:
            dish_radius = float(params["container_radius"])
            dish_source = "container_radius"
        else:
            dish_radius = float(rim_radius)
            dish_source = "measured opening (pass container_radius if known)"
        reach_limit = dish_radius - wall_clearance
        # "Forward" is the way the level bowl points, flattened -- the stroke
        # runs along the tool, whichever way the scoop was picked up.
        forward = np.asarray(level_rotation[:2, 0], dtype=float)
        forward = forward / (np.linalg.norm(forward) + 1e-12)
        across_axis = np.array([-forward[1], forward[0]])
        bowl_box = [np.array([sx * half_len, sy * bowl_width / 2.0, sz])
                    for sx in (-1.0, 1.0) for sy in (-1.0, 1.0)
                    for sz in ((0.0, -bowl_depth) if bowl_depth > 0.0 else (0.0,))]

        def corners_2d(angle_deg: float):
            """The bowl's corners about its floor reference: (along, across)."""
            R = rotation_at(angle_deg)
            pts = np.array([(R @ c)[:2] for c in bowl_box])
            return pts @ forward, pts @ across_axis

        def room(angles, fracs, advance: float):
            """
            Where the stroke may start (along ``forward``, from the dish
            centre) for this much travel: the interval over which every
            corner at every waypoint stays reach_limit from the centre, or
            None. Each waypoint's corners give a window for its position; the
            start is that window shifted back by the travel done by then.
            """
            lo, hi = -np.inf, np.inf
            for angle, frac in zip(angles, fracs):
                along, across = corners_2d(angle)
                spare = reach_limit ** 2 - across ** 2
                if np.any(spare < 0.0):
                    return None
                half = np.sqrt(spare)
                lo = max(lo, float(np.max(-half - along)) - advance * frac)
                hi = min(hi, float(np.min(half - along)) - advance * frac)
            return (lo, hi) if lo <= hi else None

        def place(angles, fracs):
            """(start, travel) for the stroke, or None if it cannot fit at all."""
            if sweep_advance is not None:
                window = room(angles, fracs, sweep_advance)
                return None if window is None else (0.5 * sum(window), sweep_advance)
            if room(angles, fracs, 0.0) is None:
                return None
            lo_a, hi_a = 0.0, 4.0 * max(reach_limit, 0.0)
            for _ in range(40):          # the longest travel that still fits
                mid = 0.5 * (lo_a + hi_a)
                if room(angles, fracs, mid) is None:
                    hi_a = mid
                else:
                    lo_a = mid
            return 0.5 * sum(room(angles, fracs, lo_a)), lo_a

        # Points along the handle, from the jaws back to its end, relative to
        # the bowl's floor reference in the tool frame.
        handle_line = [np.array([-(a_off + back_reach * k / 6.0), 0.0, -b_off])
                       for k in range(7)]

        def rim_floor_needed(angle_deg: float, xy) -> float:
            """
            Lowest floor-reference height that keeps the handle over the rim,
            for the stretch of handle that actually crosses or leaves the dish.

            Only handle points outside the inside wall count. In a dish big
            enough that the handle stays over the interior, the rim sets no
            limit and the tip keeps its depth; guarding the handle end
            everywhere cost 1.7 mm of depth at the end of every stroke.
            """
            if handle_clearance <= 0.0:
                return -np.inf
            R = rotation_at(angle_deg)
            need = -np.inf
            for p in handle_line:
                off = R @ p
                if np.linalg.norm(np.asarray(xy) + off[:2] - site) > dish_radius - 0.002:
                    need = max(need, rim_top + handle_clearance - float(off[2]))
            return need

        def build(entry_deg: float, depth: float):
            """
            The plan for a given bite and depth. Each entry of `lifts` is how
            much the floor and the rim each pushed that waypoint up from where
            scoop_depth wanted it; `stroke` is (start, travel, fits).
            """
            fracs = [i / waypoints for i in range(waypoints + 1)]
            angles = [entry_deg + (sweep_end_tilt_deg - entry_deg) * f ** roll_late
                      for f in fracs]
            placed = place(angles, fracs)
            start, travel = placed if placed is not None else (0.0, 0.0)
            plan, lifts = [], []
            for frac, angle in zip(fracs, angles):
                tip, _, _, ref = head_heights(angle)
                want = surface_z - depth + sweep_rise * frac - ref
                by_floor = floor_z + floor_clearance - tip
                xy = site + forward * (start + travel * frac)
                by_rim = rim_floor_needed(angle, xy)
                z = max(want, by_floor, by_rim)
                plan.append((angle, np.array([xy[0], xy[1], z])))
                lifts.append((max(0.0, by_floor - want),
                              max(0.0, by_rim - max(want, by_floor))))
            return plan, lifts, (start, travel, placed is not None)

        requested_tilt = dig_tilt_deg
        plan, lifts, stroke = build(dig_tilt_deg, scoop_depth)
        floor_short = max(f for f, _ in lifts)
        if floor_short > 1e-4 and depth_reference == "tip":
            # The depth and the floor limit are the same point here, so no
            # angle can buy depth back: the bed is simply shallower than the
            # request. Keep the bite as asked; build() has already stopped the
            # tip floor_clearance above the floor.
            print(f"[Scoop] The bed is only {(surface_z - floor_z) * 1000:.1f}mm "
                  f"deep: the tip can go {(scoop_depth - floor_short) * 1000:.1f}mm "
                  f"below the surface, not the {scoop_depth * 1000:.1f}mm asked, "
                  f"without coming within {floor_clearance * 1000:.0f}mm of the "
                  f"floor. Keeping {dig_tilt_deg:.0f}° and digging to the floor limit.")
        elif floor_short > 1e-4:
            # The head is too tall at this bite for the bed under it. Find the
            # steepest bite whose whole sweep reaches scoop_depth without the
            # floor stopping it. Scanned, not bisected: with the mouth as the
            # reference the head's height peaks near 50 deg and falls after,
            # so feasibility is not monotonic in the angle.
            fits = None
            angle = dig_tilt_deg
            while angle >= sweep_end_tilt_deg - 1e-9:
                trial, trial_lifts, trial_stroke = build(angle, scoop_depth)
                if max(f for f, _ in trial_lifts) <= 1e-4:
                    fits = (angle, trial, trial_lifts, trial_stroke)
                    break
                angle -= 0.5
            if fits is not None and bool(params["auto_tilt"]):
                dig_tilt_deg, plan, lifts, stroke = fits
                print(f"[Scoop] At {requested_tilt:.0f}° the head is too tall for "
                      f"this bed: reaching {scoop_depth * 1000:.1f}mm below the "
                      f"surface at the {depth_reference} would put its tip "
                      f"{floor_short * 1000:.1f}mm into the floor. Lowering the "
                      f"bite to {dig_tilt_deg:.1f}°, the steepest that reaches the "
                      f"full depth (auto_tilt).")
            elif fits is not None:
                return False, {
                    "error": (f"scoop_depth {scoop_depth * 1000:.1f}mm at the "
                              f"{depth_reference} does not fit at "
                              f"{requested_tilt:.0f}°: the tip would go "
                              f"{floor_short * 1000:.1f}mm into the floor. It fits "
                              f"at {fits[0]:.1f}° or less; lower dig_tilt_deg or "
                              f"enable auto_tilt."),
                    "fits_at_deg": fits[0],
                }
            else:
                # Not even the flattest stroke reaches the depth: the bed is too
                # shallow for this head. The floor wins; say how deep it can go.
                plan, lifts, stroke = build(sweep_end_tilt_deg, scoop_depth)
                dig_tilt_deg = sweep_end_tilt_deg
                reach = scoop_depth - max(f for f, _ in lifts)
                print(f"[Scoop] WARNING: this bed is too shallow for "
                      f"{scoop_depth * 1000:.1f}mm at the {depth_reference} at any "
                      f"bite. Running flat at {dig_tilt_deg:.0f}° reaches only "
                      f"{reach * 1000:.1f}mm without touching the floor.")

        # What the head will actually do, point by point, against the surface
        # and the floor — the numbers that decide whether it collects anything.
        def report(angle, centre):
            tip, mouth, top, _ = head_heights(angle)
            z = float(centre[2])
            return (surface_z - (z + tip), surface_z - (z + mouth),
                    surface_z - (z + top), (z + tip) - floor_z)

        e_tip, e_mouth, e_top, e_gap = report(*plan[0])
        x_tip, x_mouth, x_top, x_gap = report(*plan[-1])
        print(f"[Scoop] Depth below the surface (tip / mouth / top of the head):"
              f" entry at {plan[0][0]:+.0f}° {e_tip * 1000:.1f} / {e_mouth * 1000:.1f}"
              f" / {e_top * 1000:.1f}mm, floor gap {e_gap * 1000:.1f}mm;"
              f" end at {plan[-1][0]:+.0f}° {x_tip * 1000:.1f} / "
              f"{x_mouth * 1000:.1f} / {x_top * 1000:.1f}mm, floor gap "
              f"{x_gap * 1000:.1f}mm. Target {scoop_depth * 1000:.1f}mm at the "
              f"{depth_reference}.")
        rim_short = max(r for _, r in lifts)
        if rim_short > 1e-4:
            at = max(range(len(lifts)), key=lambda k: lifts[k][1])
            print(f"[Scoop] The rim lifts the head by up to {rim_short * 1000:.1f}mm"
                  f" (at {plan[at][0]:+.0f}°): as the head rolls level the handle "
                  f"swings down, and it must stay {handle_clearance * 1000:.0f}mm "
                  f"over the rim at z={rim_top:.4f}. Depth gives way to the wall.")
        depth_reached = min(
            surface_z - (float(c[2]) + head_heights(a)[3]) for a, c in plan)
        dug = max(surface_z - (float(c[2]) + head_heights(a)[0]) for a, c in plan)

        # 5. The stroke has to fit the dish, checked BEFORE anything moves. A
        # stroke that cannot fit without touching the wall must not run:
        # shoving the cup invalidates the scan every following skill uses.
        start, travel, stroke_fits = stroke
        if not stroke_fits:
            msg = (f"The bowl does not fit inside '{source}' at these angles: "
                   f"its footprint needs more than the {dish_radius * 1000:.0f}mm "
                   f"radius ({dish_source}) minus {wall_clearance * 1000:.0f}mm "
                   f"clearance"
                   + (f", with sweep_advance {sweep_advance * 1000:.0f}mm"
                      if sweep_advance is not None else "")
                   + ". Reduce sweep_advance or wall_clearance, or check the "
                     "dish radius.")
            if bool(params["require_clearance"]):
                return False, {"error": msg, "dish_radius": dish_radius,
                               "reach_limit": reach_limit}
            print(f"[Scoop] WARNING: {msg}")

        def along_of(point):
            return float((np.asarray(point[:2]) - site) @ forward)

        def tip_along(angle, centre):
            """Leading bottom edge of the bowl, along forward from the centre."""
            return along_of(centre[:2] + (rotation_at(angle) @ np.array(
                [half_len, 0.0, 0.0]))[:2])

        worst = 0.0
        for angle, centre in plan:
            along, across = corners_2d(angle)
            base = along_of(centre)
            worst = max(worst, float(np.max(np.hypot(base + along, across))))
        tip_travel = tip_along(*plan[-1]) - tip_along(*plan[0])
        print(f"[Scoop] Stroke: enters at {plan[0][0]:.0f}° {start * 1000:+.0f}mm "
              f"from the dish centre and finishes level at "
              f"{(start + travel) * 1000:+.0f}mm -- {travel * 1000:.0f}mm of "
              f"forward travel, the tip {tip_travel * 1000:.0f}mm. The bowl comes "
              f"within {(dish_radius - worst) * 1000:.0f}mm of the "
              f"{dish_radius * 1000:.0f}mm inside wall ({dish_source}).")

        # 6. Approach ALREADY TILTED, directly above the bed. One motion, not a
        # level hover followed by a separate rotation: the wrist has nothing to
        # hit up here, and the old two-step version spent a move getting into a
        # pose it immediately left.
        entry_rotation = rotation_at(dig_tilt_deg)
        approach_tip = np.array([plan[0][1][0], plan[0][1][1],
                                 surface_z + float(params["approach_height"])])
        approach = self.tcp_for_tip(approach_tip, entry_rotation, tool_offset)
        print(f"[Scoop] Approaching tilted {dig_tilt_deg:.0f}° nose-down with "
              f"the bowl at {np.round(approach_tip, 4)} "
              f"(TCP {np.round(approach, 4)})...")
        approach_seconds = max(0.5, float(params["approach_seconds"]))
        if not self.goto_pose_rigid(approach, entry_rotation,
                                    duration=approach_seconds):
            return False, {"error": "Failed to command the tilted approach"}
        arrived, err = self.reached(approach, float(params["hover_tol"]))
        if not arrived:
            return False, {
                "error": (f"Tilted approach above '{source}' unreachable "
                          f"(off by {err * 1000:.0f}mm)"),
            }

        # 7. Confirm the bite angle while still in free space, where backing off
        # is cheap. tool_long_axis_deg is signed, so this catches a wrist that
        # tilted the WRONG WAY — which a bare magnitude check reports as a
        # clean success and which cost a bench run to find.
        want = long_axis_for(dig_tilt_deg)
        measured = self.tool_long_axis_deg()
        for attempt in range(1, retries + 1):
            if abs(measured - want) <= tilt_tol:
                break
            print(f"[Scoop]   tilt attempt {attempt}/{retries}: long axis "
                  f"{measured:.1f}°, want {want:.1f}°")
            # Re-command the same ABSOLUTE pose with a longer duration rather
            # than nudging relatively: relative nudges stack when the wrist is
            # merely lagging, and overshoot.
            self.goto_pose_rigid(approach, entry_rotation,
                                 duration=approach_seconds + 1.0 * attempt)
            self.wait(0.3)
            measured = self.tool_long_axis_deg()
        print(f"[Scoop] Bite angle: long axis {measured:.1f}° "
              f"(want {want:.1f}° for a {dig_tilt_deg:.0f}° nose-down bite)")
        if abs(measured - want) > tilt_tol:
            return False, {
                "error": (f"Entry tilt stalled: long axis reads {measured:.1f}°, "
                          f"wanted {want:.1f}° (tol {tilt_tol:.0f}°). Entering "
                          f"the powder without the bite angle collects nothing; "
                          f"a reading near {180.0 - want:.0f}° means the wrist "
                          f"tilted the wrong way."),
                "long_axis_measured": measured,
                "long_axis_wanted": want,
            }

        # 8. Descend straight down into the bed, holding the bite angle.
        # Vertical, because the tilt is already set: a diagonal entry is what
        # the old stroke used to carry travel, and travel is what hit the wall.
        entry_centre = plan[0][1]
        entry = self.tcp_for_tip(entry_centre, entry_rotation, tool_offset)
        print(f"[Scoop] Descending the bowl floor to z={entry_centre[2]:.4f} "
              f"({scoop_depth * 1000:.0f}mm below the surface, tilted, "
              f"compliant)...")
        if not self.goto_pose_rigid(entry, entry_rotation,
                                    duration=max(0.5, float(params["descend_seconds"])),
                                    use_impedance=True):
            return False, {"error": "Failed to command the descent"}
        arrived, err = self.reached(entry, contact_tol)
        if not arrived:
            print(f"[Scoop] Descent stopped {err * 1000:.0f}mm short — backing out")
            self.goto_pose_rigid(approach, entry_rotation, duration=2.5)
            return False, {
                "error": (f"Could not enter the powder in '{source}' "
                          f"(off by {err * 1000:.0f}mm); the bowl is probably "
                          f"fouling the rim"),
            }

        ok, msg = self.check_still_holding(start_width, tag="Scoop")
        if not ok:
            return False, {"error": f"Lost the scoop entering the powder: {msg}"}

        # 9. The sweep. Roll nose-down -> nose-up about a bowl centre that
        # barely moves. Every waypoint recomputes the TCP from the bowl target,
        # so the ARM backs up as the wrist rolls — that is what keeps the bowl
        # in the same piece of powder instead of swinging it on the arc of
        # |tool_offset|, which for this scoop is ~38mm of unwanted travel.
        print(f"[Scoop] Sweeping {dig_tilt_deg:+.0f}° -> "
              f"{sweep_end_tilt_deg:+.0f}° over {sweep_seconds:.1f}s in "
              f"{waypoints} waypoints; bowl advances "
              f"{travel * 1000:.0f}mm, floor z "
              f"{plan[0][1][2]:.4f} -> {plan[-1][1][2]:.4f}...")
        # ONE continuous motion, not a chain of goto_pose calls.
        #
        # Every goto_pose ends in `_settle`, which waits for the servos to
        # stop — and a bowl buried in powder never stops, so each waypoint
        # burns its full 1.5s timeout. Twelve of them turned a 2s sweep into a
        # visible staircase of pauses that `--speed` could not touch, because
        # settling is deliberately not scaled by it. `arc_scoop` hit exactly
        # this (16 waypoints, 2.5s of sweep costing 26s) and the fix lives in
        # BaseSkill.stream_pose_path: one skill, setpoints streamed, and the
        # roll happening DURING the sweep instead of between two stops.
        rotations = [rotation_at(a) for a, _ in plan[1:]]
        tcps = [self.tcp_for_tip(c, R, tool_offset)
                for (_, c), R in zip(plan[1:], rotations)]
        swept = 0
        streamed, why = self.stream_pose_path(
            tcps, rotations, seconds=sweep_seconds, tag="Scoop",
            cartesian_impedance=True)
        if streamed:
            swept = waypoints
            print(f"[Scoop]   swept as one continuous motion ({why})")
        else:
            # Correct, just not smooth — and slow when the bowl is loaded.
            print(f"[Scoop]   continuous sweep unavailable ({why}); "
                  f"falling back to {waypoints} blocking waypoints")
            per_step = sweep_seconds / waypoints
            for angle, centre in plan[1:]:
                R = rotation_at(angle)
                tcp = self.tcp_for_tip(centre, R, tool_offset)
                if not self.goto_pose_rigid(tcp, R, duration=max(0.15, per_step),
                                            use_impedance=True):
                    print(f"[Scoop]   sweep stopped at {angle:+.0f}°")
                    break
                swept += 1
        final_angle = plan[swept][0] if swept < len(plan) else plan[-1][0]
        print(f"[Scoop] Swept {swept}/{waypoints} waypoints, "
              f"finishing at {final_angle:+.0f}° "
              f"(long axis {self.tool_long_axis_deg():.1f}°)")

        ok, msg = self.check_still_holding(start_width, tag="Scoop")
        if not ok:
            return False, {"error": f"Scoop lost during the sweep: {msg}"}

        # 10. Lift, in two parts:
        #   a) straight up at the sweep's finishing angle, until the handle is
        #      clear of the rim at the angle it is about to roll to, then
        #   b) the rest of the way while rolling nose-up to cup_tilt_deg.
        #
        # The split IS the fix. Part (b) is the motion that swings the handle
        # down and back, and it is only harmless once there is no cup around it
        # — doing it at the bottom of the dish is what put 25109 contacts on
        # the wall and shoved the cup 7.8mm. A hand does the same thing: it
        # clears the rim first and turns the spoon up on the way out.
        final_x = float(plan[swept][1][0] if swept < len(plan) else plan[-1][1][0])
        exit_angle = 0.0 if bool(params["level_on_lift"]) else -abs(cup_tilt_deg)
        # Same two hard limits as the sweep, evaluated at the angle the head is
        # about to roll to: the tip off the floor, the handle over the rim.
        clear_floor = floor_z + floor_clearance - head_heights(exit_angle)[0]
        if handle_clearance > 0.0:
            clear_floor = max(clear_floor, rim_top + handle_clearance
                              - handle_above_floor(exit_angle))
        clear_z = max(clear_floor, float(plan[-1][1][2]) + 0.005)

        stage_rotation = rotation_at(final_angle)
        stage_centre = np.array([final_x, float(site[1]), clear_z])
        stage = self.tcp_for_tip(stage_centre, stage_rotation, tool_offset)
        print(f"[Scoop] Lifting straight out to z={clear_z:.4f} at "
              f"{final_angle:+.0f}° (clearing the rim before rolling up)...")
        if not self.goto_pose_rigid(stage, stage_rotation,
                                    duration=max(0.4, float(params["lift_seconds"]) * 0.4)):
            return False, {"error": "Failed to lift the scoop clear of the rim"}

        lift_rotation = rotation_at(exit_angle)
        lift_centre = np.array([final_x, float(site[1]),
                                surface_z + float(params["lift_height"])])
        lift = self.tcp_for_tip(lift_centre, lift_rotation, tool_offset)
        print(f"[Scoop] Rising to z={lift_centre[2]:.4f} while rolling "
              f"{final_angle:+.0f}° -> {exit_angle:+.0f}° "
              f"({'levelling' if bool(params['level_on_lift']) else 'cupping the load'})...")
        if not self.goto_pose_rigid(lift, lift_rotation,
                                    duration=max(0.5, float(params["lift_seconds"]) * 0.6)):
            return False, {"error": "Failed to lift the scoop clear"}
        arrived, err = self.reached(lift, float(params["hover_tol"]))
        if not arrived:
            return False, {
                "error": f"Scoop did not lift clear (off by {err * 1000:.0f}mm)",
            }

        ok, msg = self.check_still_holding(start_width, tag="Scoop")
        if not ok:
            return False, {"error": f"Scoop lost during the lift: {msg}"}

        # The powder bed changed shape, so the cached source cloud is stale.
        self.vision.clear_cache()

        residual = self.tool_long_axis_deg()
        print(f"[Scoop] Successfully scooped from '{source}' "
              f"(bowl {90.0 - residual:+.0f}° from level)")
        return True, {
            "scooped_from": source,
            # The one depth: what was asked, and the shallowest the head's
            # depth_reference point actually got over the sweep (less than
            # asked only where the floor or the rim would not allow it).
            # "floor": tip held tip_floor_gap above the inside floor;
            # "surface": scoop_depth below the powder surface.
            "depth_mode": "floor" if tip_floor_gap is not None else "surface",
            "tip_floor_gap": tip_floor_gap,
            "scoop_depth": scoop_depth,
            "depth_reference": depth_reference,
            "depth_reached": depth_reached,
            "tip_depth_max": dug,
            "entry_depths_tip_mouth_top": [e_tip, e_mouth, e_top],
            "end_depths_tip_mouth_top": [x_tip, x_mouth, x_top],
            "floor_gap_entry": e_gap,
            "floor_gap_end": x_gap,
            "floor_z": floor_z,
            "floor_source": floor_source,
            "floor_clearance": floor_clearance,
            "rim_lift_max": rim_short,
            "surface_z": surface_z,
            "surface_given": surface_override is not None,
            # The bite actually used; differs from dig_tilt_requested when
            # auto_tilt had to flatten it to reach scoop_depth.
            "dig_tilt_deg": dig_tilt_deg,
            "dig_tilt_requested": requested_tilt,
            "sweep_end_tilt_deg": sweep_end_tilt_deg,
            "cup_tilt_deg": cup_tilt_deg,
            "exit_tilt_deg": exit_angle,
            "handle_clearance": handle_clearance,
            "tool_back_reach": back_reach,
            # Where the stroke ran, along the bowl's forward direction from
            # the dish centre (- = toward the base), and how far it travelled.
            "stroke_start": start,
            "stroke_end": start + travel,
            "stroke_travel": travel,
            "tip_travel": tip_travel,
            "stroke_fits": stroke_fits,
            "bowl_length": bowl_length,
            "bowl_width": bowl_width,
            "bowl_depth": bowl_depth,
            "roll_late": roll_late,
            "sweep_advance_requested": sweep_advance,
            "sweep_rise": sweep_rise,
            "sweep_waypoints": waypoints,
            "swept_waypoints": swept,
            "final_tilt_deg": final_angle,
            # Farthest any bowl corner got from the dish centre, and the
            # radius it was kept inside (dish radius minus wall_clearance).
            "envelope_reach": worst,
            "allowed_reach": reach_limit,
            "dish_radius": dish_radius,
            "dish_radius_source": dish_source,
            "rim_radius": rim_radius,
            "tool_span": tool_span,
            "tool_offset": [float(v) for v in tool_offset],
            "forward_offset": float(params["forward_offset"]),
            "lateral_offset": float(params["lateral_offset"]),
            # Signed, so a wrong-way finish is visible in the log rather than
            # hiding behind a magnitude.
            "residual_from_level_deg": 90.0 - residual,
            # There is no measurement of how much powder is on the scoop. The
            # quantity is nominal, set by depth, bowl size and the roll.
            "quantity_measured": False,
        }
