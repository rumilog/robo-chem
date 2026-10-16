"""
What is on the real bench, and how the real cell finds each thing.

The simulated cell knows its bench exactly (``robochem.sim.bench``) and tells
the scene agent so. The real cell had no such list -- grounding is
open-vocabulary -- and on 2026-10-07 that was the whole difference between a
31-step sim run and a real dry run that planned nothing: with no inventory the
scene agent named 6 of 11 objects, called the scoops and the stirrer
"distractors", folded the beaker into the clear cups, and the planner invented
a pipette.

A manifest gives the real cell the same names the sim uses, so the scene agent
gets the same inventory prompt and the planner the same vocabulary. Behind each
name is what the real perception stack actually needs:

  * a container is found by the handwritten label it stands on (``label``)
    among instances of its SAM ``category`` -- the path validated on the
    robot for the reagent dishes and the B cup (2026-09-25);
  * a tool is found by the SAM ``prompt`` the prompt sweep chose for it, not
    by its name: SAM 3 finds nothing for "spoon" (2026-09-18).

Names the manifest does not know are left alone, so hand-typed ``--skill``
commands keep working on any bench.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .label_resolver import normalize_label


@dataclass(frozen=True)
class BenchObject:
    """One physical object on the bench."""

    #: The name agents and skills use. The same as the sim's prop name.
    name: str
    #: The handwritten label it stands on, as written (case is ignored).
    label: Optional[str] = None
    #: SAM category whose instances are searched for ``label``.
    category: Optional[str] = None
    #: SAM prompt for an object found by appearance instead of by label.
    prompt: Optional[str] = None
    #: Other names that mean this object, including what instructions call it.
    aliases: Tuple[str, ...] = ()
    #: What a person at the bench would add, for the scene agent.
    note: str = ""
    #: A clear container. Depth passes through its wall to the table behind,
    #: so each camera's points lean away from that camera; it is located by
    #: its silhouette instead (``VisionSystem._locate_transparent``), and
    #: handed to the skills as a cylinder of the size below.
    transparent: bool = False
    #: Outside radius at the rim and height, metres. Needed when transparent.
    radius: Optional[float] = None
    height: Optional[float] = None
    #: Wall thickness, metres: the opening is radius - wall.
    wall: float = 0.001
    #: pick_up parameters validated on the robot for this object, as
    #: (name, value) pairs. ``SkillsExecutor`` fills in any the call leaves
    #: out; a planner passes only the object's name.
    pick: Tuple[Tuple[str, object], ...] = ()
    #: scoop parameters for scooping out of this container, likewise.
    scoop: Tuple[Tuple[str, object], ...] = ()


class BenchManifest:
    """A named list of :class:`BenchObject`, looked up by any name or label."""

    def __init__(self, name: str, objects: List[BenchObject],
                 measured_tool_offset: bool = False,
                 skill_defaults: Optional[Dict[str, Dict[str, object]]] = None):
        self.name = name
        self.objects = list(objects)
        #: skill -> parameters for every call of that skill on this bench,
        #: whatever its target. An object's own defaults (pick, scoop) win.
        self.skill_defaults = {k: dict(v) for k, v in (skill_defaults or {}).items()}
        #: Take a held tool's offset from pick_up's measurement (from where
        #: the jaws closed) rather than from the ends of its cloud. Set once
        #: a ruler has confirmed the measurement on this bench.
        self.measured_tool_offset = bool(measured_tool_offset)
        self._by_key: Dict[str, BenchObject] = {}
        for obj in self.objects:
            for key in (obj.name, obj.label) + tuple(obj.aliases):
                if not key:
                    continue
                k = normalize_label(key)
                other = self._by_key.get(k)
                if other is not None and other is not obj:
                    raise ValueError(f"bench {name!r}: {key!r} names both "
                                     f"{other.name!r} and {obj.name!r}")
                self._by_key[k] = obj

    def find(self, name) -> Optional[BenchObject]:
        """The object ``name`` refers to (its name, a label or an alias), or None."""
        if not isinstance(name, str) or not name.strip():
            return None
        return self._by_key.get(normalize_label(name))

    def canonical(self, name):
        """The bench name for ``name``, or ``name`` unchanged if it is not on the bench."""
        obj = self.find(name)
        return obj.name if obj is not None else name

    def known_object_names(self) -> List[str]:
        """Every object's name and label, as the sim's ``known_object_names`` gives them."""
        names = []
        for obj in self.objects:
            names.append(obj.name)
            if obj.label:
                names.append(obj.label.lower())
        return sorted(set(names))

    def defaults(self, skill: str, name) -> Dict[str, object]:
        """Validated parameters for ``skill`` on this bench, and on the object ``name``."""
        out = dict(self.skill_defaults.get(skill, {}))
        obj = self.find(name)
        if obj is None:
            return out
        if skill == "pick_up":
            out.update(dict(obj.pick))
        elif skill == "scoop":
            out.update(dict(obj.scoop))
        return out

    def label_texts(self) -> List[str]:
        """Every handwritten label on the bench, as written (upper case), for the label reader."""
        return [obj.label.upper() for obj in self.objects if obj.label]

    def inventory_notes(self) -> Dict[str, str]:
        """
        One line per object, in the sim's format (``SimVision.inventory_notes``).

        ``comprehend_scene`` reads the label back out of the ``reading "..."``
        phrase for an object the scene agent left out, so the wording matters.
        """
        out = {}
        for obj in self.objects:
            bits = []
            if obj.label:
                label = obj.label.lower()
                bits.append(f'stands on a handwritten label reading "{label}", '
                            f'and "{label}" names it too')
            if obj.aliases:
                bits.append("also called " + ", ".join(f'"{a}"' for a in obj.aliases))
            if obj.note:
                bits.append(obj.note)
            out[obj.name] = "; ".join(bits)
        return out


#: The powder dishes, measured 2026-09-25: 91 mm across the inside, 29 mm
#: tall.
DISH_INSIDE_RADIUS = 0.0455
DISH_HEIGHT = 0.029
#: scoop's z_offset on this bench, metres: the whole dig moved down (-) or up
#: (+). The user saw the bowl ride too high over the bed and asked for it
#: lower (2026-10-08): first -3 mm, then -15 mm after watching that run, then
#: -30 mm ("too shallow, add at least 15 mm") in the 20:50 run, where the
#: offset the pick measured from jaws to bowl was 51 mm against 40 mm in the
#: runs -15 mm suited. At the 60 deg dig, 11 mm too much in that offset puts
#: the bowl ~10 mm high. With an offset measured near 40 mm again, -30 mm
#: may press the bowl into the dish floor. The user then set -25 mm.
#: Change this one number to tune it; a run can also pass "z_offset" in
#: scoop's --params.
#: **Reset to 0 on 2026-10-10** after the cameras were recalibrated (they
#: disagreed by ~36 mm and the table was taken as z 0, not -11.2 mm). Every
#: offset tuned against the old calibration is being re-checked from zero;
#: this one was -0.025. First run at 0 (2026-10-10): the user said ~25 mm too
#: high. 14 mm of that was the floor taken from the dish cloud's base, which
#: read 14 mm above the table (scoop now takes the table by the dish's top);
#: -11 mm was the rest, at the user's word.
#: At -11 mm every sweep that finished ended ~6 mm ABOVE its last setpoint
#: (+6.0, +6.0, +5.7 mm on 2026-10-10/15): the plan ran 6 mm below the dish
#: floor and position control pressed the bowl into it. Two of five sweeps
#: jammed there at the 60 deg entry (libfranka abort, the arm left in the
#: dish). -5 mm puts the end of the sweep on the floor, no deeper: the bowl
#: cannot collect anything below the floor, so it scoops as deep as before.
DISH_SCOOP_Z_OFFSET = -0.005

#: dump's forward_offset on this bench, metres in the base frame (+ away from
#: the arm, - back toward it). The user saw the dump land too far forward and
#: asked for 10 mm back (2026-10-08). A run can pass "forward_offset" in
#: dump's --params instead.
#: **Reset to 0 on 2026-10-10** after the cameras were recalibrated (they
#: disagreed by ~36 mm and the table was taken as z 0, not -11.2 mm). Every
#: offset tuned against the old calibration is being re-checked from zero;
#: this one was -0.010. Part of it may be the streamed carry overshooting
#: (11.9 mm in free air), which recalibration does not change.
DUMP_FORWARD_OFFSET = 0.0

#: dump's min_tip_deg on this bench: the least tip that counts as emptied.
#: The skill's 60 failed a dump into the beaker that stalled at 59.9 deg, which
#: the user watched empty the scoop well (2026-10-08). 55 then failed one at
#: 54.6 deg (2026-10-15: the beaker 0.57 m out caps the tip at 59, and the
#: streamed tip ends ~4 deg short of its command). The user: 54.6 is fine,
#: "don't set it too strict, keeping the experiment going is the most
#: important". 45 leaves room under the reach cap.
DUMP_MIN_TIP_DEG = 45.0

#: stir's circle on this bench, metres. The skill's defaults (15 mm, kept 12 mm
#: off the wall) drew a circle the user found too small (2026-10-08). The arm
#: also streams it under cartesian impedance, which shrinks a small circle.
#: With the 5 mm-radius rod, 20 mm in a 53.5 mm cup leaves the rod about
#: 1.75 mm off the wall at the edge of the circle, if the centre is right.
STIR_RADIUS = 0.020
STIR_WALL_CLEARANCE = 0.006
#: stir's fine_zone on this bench, metres: how far above its aim (5 mm under
#: the cup model's base, with to_floor) the descent switches from 5 mm steps
#: to 1 mm force-read steps. The skill's 15 mm worked while the cups stood on
#: z 0 (2026-10-08: 1 mm steps from TCP z 0.0923, floor felt at 0.0803).
#: With table_z -0.0112 the aim moved 11.2 mm down and the fine zone with it,
#: to start at ~0.0798, i.e. at the floor: the rod hit it on a coarse step,
#: pressed in, and stirred against the bottom, dragging the cup (the user,
#: 2026-10-10). 15 + 11.2 put the 1 mm steps back where they were (0.026).
#: Three stirs of cup B then felt the floor at TCP z 0.0821 every time, and
#: the user found the descent too slow: 0.020 starts the 1 mm steps ~4 mm
#: above that (0.0861), and STIR_COARSE_STEP crosses the open cup in two
#: moves. About 7 steps instead of ~17, each a separate motion plus a force
#: read. The cups are identical and stand on the table, so the floor is felt
#: at the same height in A, B and C.
STIR_FINE_ZONE = 0.020
#: Coarse steps were 20 mm until 2026-10-15: in the pipeline the first one
#: (20 mm in 0.4 s, in free air above cup A) read +1.00 N against the 1.0 N
#: threshold and the stir was refused as "landing on the rim". 10 mm, and the
#: skill now lets the arm settle before reading the force after a coarse step.
STIR_COARSE_STEP = 0.010
#: stir's contact_backoff on this bench, metres: how far above the felt floor
#: the circle runs. The skill's 4 mm (to_floor) still scraped: the circle is
#: streamed under cartesian impedance and its height wanders by millimetres
#: (2026-10-10, contact at TCP z 0.0821, stirred at 0.0861, the cup dragged).
#: The user chose to keep the smooth stream and stir 3 mm higher rather than
#: switch to position-controlled waypoints (smooth: false). In a 4.4 mm-deep
#: 10 ml the tip may now ride near the surface. With position control
#: (STIR_SMOOTH False) 7 mm no longer touched (18:39), and the user asked for
#: 4 mm again. Streamed with z at 1500 N/m (18:54) it still touched lightly
#: (contact at TCP z 0.0801, stirred at 0.0841): 5 mm, with z at 2000.
STIR_CONTACT_BACKOFF = 0.005
#: stir's smooth on this bench. False: the circle is walked as blocking,
#: position-controlled waypoints (waypoints_per_rev 12, 0.4 s each) instead of
#: one streamed impedance motion per revolution. At 7 mm back off the
#: streamed circle still touched the bottom (2026-10-10 18:36: contact at
#: TCP z 0.0821, stirred at 0.0891, revolutions ending 6.3-6.8 mm off), and the
#: user asked for position control. The scoop's position sweep tracked to
#: 0.3 mm.
STIR_SMOOTH = False
#: Then the user asked to keep the smooth stream and fix its height instead:
#: the stream is cartesian impedance at frankapy's 600 N/m, and joint friction
#: changing direction around the circle moves z by millimetres. Streamed
#: again, with z alone stiffened to 1500 N/m (x/y and rotation at frankapy's
#: defaults, so the 50 Hz x/y setpoint steps stay soft). If the arm shakes,
#: stop it and go back to STIR_SMOOTH False.
#: At 1500 (18:54) no shaking, revolutions ended dz +0.4..+0.5 mm (+1.0..+1.2
#: at 600), and the rod still touched lightly: 2000.
STIR_SMOOTH = True
STIR_STREAM_STIFFNESS = [600.0, 600.0, 2000.0, 50.0, 50.0, 50.0]
#: The waypoint circle's pace: the skill's 12 points at 0.4 s each was too
#: slow and stepped for the user (18:39). 16 at 0.25 s: shorter segments, so
#: the stops between them are smaller, and 4.0 s commanded per revolution
#: instead of 4.8. Each blocking goto_pose still adds its own start-up time.
STIR_WAYPOINTS_PER_REV = 16
STIR_SECONDS_PER_WAYPOINT = 0.25


def _dish(name: str, label: str) -> BenchObject:
    # The scoop sizes its stroke to the dish's inside. Measured off the cloud
    # it can come out wide: on the first real pipeline run (2026-10-07) the
    # rim fit was rejected (15% inliers) and the fallback read 61 mm, so the
    # stroke was planned 15 mm past the real wall.
    return BenchObject(name=name, label=label, category="white bowl",
                       aliases=(f"{label} dish", f"{label} bowl"),
                       note="a shallow white dish",
                       scoop=(("container_radius", DISH_INSIDE_RADIUS),
                              ("container_height", DISH_HEIGHT),
                              # Chosen by the user after the air rehearsal
                              # (2026-10-08): streamed, the sweep ended 11.9 mm
                              # past its setpoint and the bowl met the far wall.
                              ("sweep_mode", "position"),
                              ("z_offset", DISH_SCOOP_Z_OFFSET)))


#: The kit's clear cups, measured by the user: 47.3 mm tall (2026-10-03);
#: 73.2 mm across the outside at the rim, their widest, and about 61 mm where
#: the jaws close, 25 mm under the rim (2026-10-08). The 55.5 mm recorded on
#: 2026-10-03 as the outside width was not the rim. Wall 1 mm is not measured.
CLEAR_CUP_RADIUS = 0.0366
CLEAR_CUP_HEIGHT = 0.0473

#: pick_up's lateral_offset for cup A, metres along base y (+ the robot's
#: left). On 2026-10-08 the pick of A (step 26) went in askew and the cup
#: slipped out of the jaws: 55.5 mm of rim in an 80 mm opening leaves about
#: 12 mm a side. The user then read point_at.py over A, B and C: A's
#: perceived centre was 5 mm toward B; B and C were right. A stood at about
#: (0.40, -0.16) and B at (0.40, -0.05), so this moves A's grasp 5 mm away
#: from B. It belongs to that spot: if A is moved, check it again with
#: point_at.py. A run can pass "lateral_offset" in pick_up's --params.
#: **Reset to 0 on 2026-10-10** after the cameras were recalibrated (they
#: disagreed by ~36 mm and the table was taken as z 0, not -11.2 mm). Every
#: offset tuned against the old calibration is being re-checked from zero;
#: this one was -0.005.
CUP_A_LATERAL_OFFSET = 0.0

#: How A, B and C are picked: the user's procedure (2026-10-08). The jaws
#: open fully, go down CLEAR_CUP_SLIDE_IN behind the cup (toward the base),
#: slide forward onto it, and close; place sets it down where it was held and
#: slides the open jaws back out before rising. CLEAR_CUP_GRASP_Z_OFFSET: the
#: user saw the jaws close too high on the cup and asked for 12 mm lower
#: (rim 25 mm above the TCP instead of 13). 60 mm behind keeps the fingers
#: (pads +-10 mm) about 13 mm off the 73.2 mm rim on the way down.
#: CLEAR_CUP_GRASP_Z_OFFSET:
#: **Reset to 0 on 2026-10-10** after the cameras were recalibrated (they
#: disagreed by ~36 mm and the table was taken as z 0, not -11.2 mm). Every
#: offset tuned against the old calibration is being re-checked from zero;
#: it was -0.012. A clear cup is rebuilt standing on VisionSystem.table_z,
#: which was 0 while the table is at -11.2 mm: the cup model sat ~11 mm high,
#: which this offset was compensating.
CLEAR_CUP_SLIDE_IN = 0.06
CLEAR_CUP_GRASP_Z_OFFSET = 0.0
#: Close to this and stop. The force-limited close (1.5 N) kept squeezing:
#: the user saw the cup "squeezed heavily", held at 37 mm on a section the
#: user measured at about 61 mm (2026-10-08). 61 - 3 mm. Wider if it still
#: squeezes, narrower if the cup slips.
CLEAR_CUP_GRIP_WIDTH = 0.058
_CUP_PICK = (("slide_in", CLEAR_CUP_SLIDE_IN), ("z_offset", CLEAR_CUP_GRASP_Z_OFFSET),
             ("grip_width", CLEAR_CUP_GRIP_WIDTH))

#: pick_up's z_offset for the beaker (the "50 ML WATER" cup): the user found
#: the jaws closing too high on it (2026-10-08) and gave no number; 12 mm, as
#: for A-C. Change this one number to tune it.
#: **Reset to 0 on 2026-10-10** after the cameras were recalibrated (they
#: disagreed by ~36 mm and the table was taken as z 0, not -11.2 mm). Every
#: offset tuned against the old calibration is being re-checked from zero;
#: this one was -0.012, likely the same 11 mm as the clear cups'.
BEAKER_GRASP_Z_OFFSET = 0.0


def _cup(name: str, label: str, aliases: Tuple[str, ...],
         radius: float = CLEAR_CUP_RADIUS, height: float = CLEAR_CUP_HEIGHT,
         pick: Tuple[Tuple[str, object], ...] = ()) -> BenchObject:
    return BenchObject(name=name, label=label, category="clear plastic cup",
                       aliases=aliases, note="a clear plastic cup",
                       transparent=True, radius=radius, height=height, pick=pick)


#: The booklet's Magic Beaker (pages 12-13) as set up on the real bench on
#: 2026-10-07, mirroring ``robochem.sim.bench.default_bench`` with two
#: differences the user chose:
#:   * one scoop. The small one was taken off the bench; the big one stands in
#:     for it, so "small scoop" is an alias of "larger spoon" here, and a plan
#:     never picks a tool whose CAD (the small spoon's is a placeholder) does
#:     not match what is in the jaws;
#:   * the beaker is a clear plastic cup on the "50 ML WATER" label, not the
#:     red translucent beaker the "plastic beaker" prompt was swept on.
MAGIC_BEAKER = BenchManifest("magic_beaker", [
    _dish("citric acid cup", "citric acid"),
    _dish("baking soda cup", "baking soda"),
    _dish("red cabbage powder cup", "red cabbage powder"),
    _cup("clear cup a", "a 10 ml water", ("cup a",),
         pick=_CUP_PICK + (("lateral_offset", CUP_A_LATERAL_OFFSET),)),
    _cup("clear cup b", "b 10 ml water", ("cup b",), pick=_CUP_PICK),
    # The same cup as A and B (the user, 2026-10-07).
    _cup("clear cup c", "c empty", ("cup c", "empty clear cup", "empty cup"),
         pick=_CUP_PICK),
    BenchObject(
        name="plastic beaker", label="50 ml water", category="clear plastic cup",
        aliases=("beaker",),
        note=("a clear plastic cup that serves as the beaker; it looks like "
              "the other clear cups and is told apart by its label"),
        # NOT MEASURED: a narrow clear cup, taller than A and B. Silhouette
        # triangulation put its middle 31 mm up (A, B: 22-29 mm), so roughly
        # 60 mm tall; the width is read off the frames.
        transparent=True, radius=0.022, height=0.060,
        # The user: the beaker's grasp "does not go deep enough, too high"
        # (2026-10-08). 12 mm lower, as for cups A-C. The put-back releases
        # as much lower (SkillsExecutor._resolve_pick_site).
        pick=(("z_offset", BEAKER_GRASP_Z_OFFSET),),
    ),
    BenchObject(
        name="larger spoon", prompt="white plastic tool",
        aliases=("big scoop", "big spoon", "large spoon", "scoop",
                 "small scoop", "small spoon", "smaller spoon", "smaller scoop",
                 "white plastic tool"),
        note=("a white plastic scoop lying flat on the bench, and the ONLY "
              "scoop here: use it wherever the instructions call for the big "
              "scoop or the small one"),
        # The robot-validated grasp (2026-09-25, and again 2026-10-07 through
        # this manifest): 10 mm back keeps the jaws off the crank, where the
        # handle's section changes and they cannot seat; 1 N, not the 1.5 N
        # default. The default forward_offset 0 lands on the crank.
        # Reset to 0 on 2026-10-10 with the other offsets, at the user's
        # request, after the recalibration. Back to -0.010 on 2026-10-15: at
        # 0 the jaws closed on the crank (27.1 mm, not grasped), the spoon
        # slid to 8.2 mm in the jaws, and the scoop jammed on the dish floor.
        pick=(("z_offset", 0.0), ("grasp_force", 1.0), ("forward_offset", -0.010)),
    ),
    BenchObject(
        name="stirrer", prompt="white cube",
        aliases=("stirring rod", "stir rod", "stirring stick", "stir stick",
                 "white cube"),
        note=("a rod with a white cube on top, standing in its holder; the "
              "gripper takes it by the cube"),
        # As the robot-run chain picked "white cube" (2026-09-25, 2026-10-05).
        pick=(("z_offset", 0.0), ("grasp_force", 1.0)),
    ),
    BenchObject(
        name="stirrer holder",
        aliases=("holder", "stirrer stand", "stir holder", "tool holder"),
        note=("the dark block the stirrer stands in; the stirrer goes back "
              "into it and nothing else does"),
    ),
],
    # A ruler, scoop in the jaws (2026-10-08): jaws centre to bowl centre
    # 52.9 mm; pick_up printed 0.053. The cloud's ends had been 14-29 mm short
    # on every robot pick, and the scoop dug that much too far forward.
    measured_tool_offset=True,
    skill_defaults={"dump": {"forward_offset": DUMP_FORWARD_OFFSET,
                             "min_tip_deg": DUMP_MIN_TIP_DEG},
                    "stir": {"stir_radius": STIR_RADIUS,
                             "wall_clearance": STIR_WALL_CLEARANCE,
                             "fine_zone": STIR_FINE_ZONE,
                             "coarse_step": STIR_COARSE_STEP,
                             "contact_backoff": STIR_CONTACT_BACKOFF,
                             "smooth": STIR_SMOOTH,
                             "stream_stiffness": STIR_STREAM_STIFFNESS,
                             "waypoints_per_rev": STIR_WAYPOINTS_PER_REV,
                             "seconds_per_waypoint": STIR_SECONDS_PER_WAYPOINT}},
)

#: Benches selectable with ``run_experiment.py --bench``.
BENCHES: Dict[str, BenchManifest] = {MAGIC_BEAKER.name: MAGIC_BEAKER}
