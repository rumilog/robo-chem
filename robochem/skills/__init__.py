"""
RoboChem Skills Library

Each skill represents a self-contained manipulation primitive that the VLM orchestrator
can invoke. Skills follow a consistent interface:
1. Pre-conditions check
2. Visual grounding (object localization)
3. Motion planning
4. Execution
5. Post-conditions check
"""

import numpy as np

from .base_skill import BaseSkill

# Manipulation skills
from .pick_up import PickUpSkill
from .place import PlaceSkill
from .pour import PourSkill
from .scoop import ScoopSkill
from .arc_scoop import ArcScoopSkill
from .dump import DumpSkill
from .dispense import DispenseSkill
from .stir import StirSkill
from .move_to import MoveToSkill
from .gripper import GripperSkill
from .tilt import TiltSkill
from .wait import WaitSkill

# Perception-only skills (no robot movement)
from .analyze_scene import AnalyzeSceneSkill
from .locate_object import LocateObjectSkill
from .check_container import CheckContainerSkill

# Registry mapping skill names to their classes
SKILL_REGISTRY = {
    # Manipulation skills
    "pick_up": PickUpSkill,
    "place": PlaceSkill,
    "pour": PourSkill,
    "scoop": ScoopSkill,
    # A second scooping stroke, curved rather than straight. Deliberately a
    # separate skill: "scoop" is tuned against the bench and stays that way.
    "arc_scoop": ArcScoopSkill,
    "dump": DumpSkill,
    "dispense": DispenseSkill,
    "stir": StirSkill,
    "move_to": MoveToSkill,
    "open_gripper": GripperSkill,
    "close_gripper": GripperSkill,
    "tilt": TiltSkill,
    "wait": WaitSkill,
    
    # Perception skills (VLM can call these to "see" without moving)
    "analyze_scene": AnalyzeSceneSkill,
    "locate_object": LocateObjectSkill,
    "check_container": CheckContainerSkill,
}

def get_skill(skill_name: str) -> type:
    """
    Get skill class by name.
    
    Args:
        skill_name: Name of the skill (e.g., "pick_up", "pour")
        
    Returns:
        The skill class
        
    Raises:
        ValueError: If skill name is not in registry
    """
    if skill_name not in SKILL_REGISTRY:
        available = list(SKILL_REGISTRY.keys())
        raise ValueError(f"Unknown skill: {skill_name}. Available skills: {available}")
    return SKILL_REGISTRY[skill_name]


def list_skills() -> list:
    """Return list of available skill names."""
    return list(SKILL_REGISTRY.keys())


class SkillsExecutor:
    """
    Manages skill execution with shared robot and vision resources.
    """
    
    # Registry aliases that pin a parameter, so the orchestrator can call
    # "open_gripper" directly instead of "gripper" with action="open".
    SKILL_ALIASES = {
        "open_gripper": {"action": "open"},
        "close_gripper": {"action": "close"},
    }
    
    def __init__(self, robot_interface, vision_system, config: dict = None):
        """
        Initialize the skills executor.
        
        Args:
            robot_interface: FrankaArm instance or similar robot interface
            vision_system: Vision system for scene understanding
            config: Optional configuration dictionary
        """
        self.robot = robot_interface
        self.vision = vision_system
        self.config = config or {}
        
        # Pre-instantiate skills for reuse
        self._skill_instances = {}
        #: object name -> where pick_up took it from, for "put it back".
        self.pick_sites = {}
        #: what pick_up last put in the jaws, until place or an open clears it.
        self.held = None
        #: what that pick measured of it (suggested offset, extent along x).
        self.held_measure = {}
        # Where the TCP was when each remembered object was grasped. Putting
        # the TCP back there re-seats a tool exactly as it sat (the stirrer in
        # its holder); the centroid alone is only the mean of what was visible.
        self.pick_grasps = {}
        # The z_offset each remembered pick used. A grasp moved down holds the
        # object higher in the jaws, so its put-back must release lower.
        self.pick_z_offsets = {}

    def list_skills(self) -> list:
        """
        Names the orchestrator may dispatch.

        Includes the aliases, since those are what the planner is told about.
        """
        return list_skills()
        
    def execute(self, skill_name: str, params: dict) -> tuple:
        """
        Execute a skill by name.
        
        Args:
            skill_name: Name of the skill to execute
            params: Parameters to pass to the skill
            
        Returns:
            Tuple of (success: bool, result: dict)
        """
        if skill_name not in self._skill_instances:
            skill_class = get_skill(skill_name)
            # Top-level config (workspace_min, etc.) applies to every skill.
            # Optional nested config[skill_name] can override per skill.
            # Previously only config[skill_name] was passed, so --workspace-min
            # from run_experiment was silently ignored and the 0.015 m floor stuck.
            skill_config = {
                k: v for k, v in self.config.items()
                if k not in SKILL_REGISTRY
            }
            per_skill = self.config.get(skill_name)
            if isinstance(per_skill, dict):
                skill_config.update(per_skill)
            self._skill_instances[skill_name] = skill_class(
                self.robot, self.vision, skill_config
            )
        
        skill = self._skill_instances[skill_name]
        
        params = {**self.SKILL_ALIASES.get(skill_name, {}), **(params or {})}
        params = self._bench_names(params)
        params = self._bench_defaults(skill_name, params)
        params = self._resolve_pick_site(skill_name, params)
        params = self._fill_tool_geometry(skill_name, skill, params)
        if skill_name == "place":
            held = getattr(self, "held_measure", None) or {}
            if params.get("pick_tool_z") is None and held.get("tool_z") is not None:
                params = {**params, "pick_tool_z": list(held["tool_z"])}
            if params.get("slide_out") is None and held.get("slide_in"):
                params = {**params, "slide_out": float(held["slide_in"])}

        # Check preconditions
        can_execute, message = skill.check_preconditions(params)
        if not can_execute:
            return False, {"error": message, "phase": "precondition_check"}
        
        # Execute the skill
        try:
            success, result = skill.execute(params)
            if success:
                self._remember_pick_site(skill_name, params, result)
                self._track_held(skill_name, params, result)
            return success, result
        except Exception as e:
            return False, {"error": str(e), "phase": "execution"}

    #: Params whose value names an object on the bench.
    _OBJECT_PARAMS = ("object_name", "target_location", "target_container",
                      "powder_source", "source_container", "target")

    def _bench_names(self, params: dict) -> dict:
        """
        Rename any alias or label of a bench object to the bench's own name.

        Only a vision system with a bench manifest renames anything. Two things
        depend on it. Tool geometry is looked up by the name a tool was picked
        by: on the 2026-10-07 bench "smaller spoon" is the big scoop, and its
        own CAD entry is a placeholder at 75%. And a pick and its put-back have
        to agree on the key ("scoop" picked, "larger spoon" put back).
        """
        canonical = getattr(self.vision, "canonical_name", None)
        if canonical is None:
            return params
        out = dict(params)
        for key in self._OBJECT_PARAMS:
            value = out.get(key)
            if not isinstance(value, str):
                continue
            name = canonical(value)
            if isinstance(name, str) and name != value:
                print(f"[Skills] {key} {value!r} is {name!r} on this bench")
                out[key] = name
        return out

    def _bench_defaults(self, skill_name: str, params: dict) -> dict:
        """
        Fill in the parameters the bench manifest validated for this object.

        A planner names the object and nothing else. The scoop's grasp was
        made to work on the robot with forward_offset -0.010 and 1 N; the
        defaults (0, 1.5 N) put the jaws on its crank. A dish's inside radius
        is measured, and the scoop's stroke is sized to it. A value the call
        does pass is kept.
        """
        defaults_for = getattr(self.vision, "bench_defaults", None)
        if defaults_for is None:
            return params
        key = {"pick_up": "object_name", "scoop": "powder_source",
               "dump": "target_container", "stir": "target_container"}.get(skill_name)
        name = params.get(key) if key else None
        name = name if isinstance(name, str) else ""
        found = defaults_for(skill_name, name)
        added = {k: v for k, v in (found if isinstance(found, dict) else {}).items()
                 if k not in params}
        if not added:
            return params
        print(f"[Skills] {skill_name} {name!r}: bench-validated "
              + ", ".join(f"{k}={v}" for k, v in added.items()))
        return {**params, **added}

    # ---------------------------------------------------------- pick sites
    #
    # "Put it back where you found it" cannot be answered by perception: by
    # the time it is asked, the object is in the gripper and no camera can see
    # it on the bench. Only the pick knows, so the pick has to say.
    #
    # Holding it here rather than in the skills is deliberate — one executor
    # spans a whole --skill chain, which is the session a hand-off happens in.

    def _remember_pick_site(self, skill_name: str, params: dict, result: dict):
        """After a successful pick_up, note where the object was taken from."""
        if skill_name != "pick_up" or not isinstance(result, dict):
            return
        name = params.get("object_name")
        centroid = result.get("centroid")
        if not name or centroid is None:
            return
        site = list(centroid)
        key = str(name).strip().lower()
        self.pick_sites[key] = site
        # Where the TCP actually closed, not where it was sent: the grasp
        # move is accepted within 30mm. Older results only have the command.
        self.pick_z_offsets[key] = float(result.get("z_offset") or 0.0)
        reached = result.get("grasp_tcp_reached")
        grasp = result.get("grasp_pose")
        if reached is not None:
            self.pick_grasps[key] = [float(v) for v in reached]
        elif grasp is not None:
            self.pick_grasps[key] = [float(v) for v in
                                     np.asarray(grasp, dtype=float)[:3, 3]]
        print(f"[Skills] Noted where '{name}' was picked from: "
              f"[{site[0]:.4f}, {site[1]:.4f}, {site[2]:.4f}] — "
              f"place it back by name")

    # ------------------------------------------------------- tool geometry
    #
    # scoop and dump plan around the bowl -- its size, and where it hangs
    # below the jaws -- and neither a camera nor a planning model can supply
    # that once the tool is held. The pick knows what the tool is, so the
    # executor looks it up (tool_geometry.TOOLS) and fills in what a call left
    # out. A 2026-09-25 agent run in sim passed no bowl size at all; scoop then
    # planned for a point and drove the real 27 mm bowl into the dish wall.

    def _track_held(self, skill_name: str, params: dict, result=None):
        if skill_name == "pick_up":
            self.held = params.get("object_name")
            result = result if isinstance(result, dict) else {}
            # What the pick measured of the tool, for _fill_tool_geometry.
            self.held_measure = {"suggested": result.get("suggested_tool_offset"),
                                 "extent_x": result.get("tool_extent_x"),
                                 # Picked by sliding in from behind: place
                                 # leaves the same way (slide_out).
                                 "slide_in": result.get("slide_in")}
            grasp = result.get("grasp_pose")
            if grasp is not None:
                # The tool's axis as picked: place checks the object is not
                # still tipped (from a pour the robot faulted out of).
                self.held_measure["tool_z"] = [
                    float(v) for v in np.asarray(grasp, dtype=float)[:3, 2]]
            top, bottom = result.get("object_top_z"), result.get("object_bottom_z")
            if grasp is not None and top is not None and bottom is not None:
                tcp_z = float(np.asarray(grasp, dtype=float)[2, 3])
                table = self.config.get("table_z")
                # The cloud's bottom reads 3-6 mm high (the bottom edge is
                # seen only at a grazing angle); standing on a known table,
                # the table is the bottom.
                if table is not None and -0.005 <= bottom - float(table) <= 0.012:
                    bottom = float(table)
                self.held_measure["container"] = {
                    "top_above_tcp": float(top) - tcp_z,
                    "height": float(top) - float(bottom),
                    "radius": float(result.get("object_radius") or 0.0),
                }
        elif skill_name in ("place", "open_gripper") or params.get("action") == "open":
            self.held = None
            self.held_measure = {}
        # A container in the jaws is not on the bench, and must not be voted
        # for there: on 2026-10-08, with the beaker held, its label paper
        # still lay by cup A, one camera read A as "50 ML WATER", and A tied
        # with B and was refused.
        tell = getattr(self.vision, "set_held", None)
        if callable(tell):
            tell(self.held)

    def _offset_from_tip(self, tool: dict, cad: list):
        """
        The tool offset with x read off where the tool's two ends were seen at
        the pick, y and z from the CAD. None without a usable measurement.

        The CAD x assumes one grasp point and the jaws close wherever the
        cloud's centroid put them: 2.8 mm toward the bowl once the sim's depth
        lost its antialiasing, enough to cut a scoop to a fifth (2026-10-03).
        pick_up's own x is the median of the bowl end, 7.7 mm off there. The
        middle of the cloud's span plus the CAD distance from the part's
        middle to its bowl came within 1.5 mm.
        """
        extent = (getattr(self, "held_measure", None) or {}).get("extent_x")
        to_bowl = tool.get("span_mid_to_bowl")
        if not extent or to_bowl is None or abs(cad[0]) < 0.005:
            return None
        length = tool.get("length")
        seen = float(extent[1]) - float(extent[0])
        if length and abs(seen - float(length)) > 0.015:
            # The middle of the cloud is the part's middle only when the cloud
            # is the part. On the robot (2026-10-07/08) the big scoop's cloud
            # ran 111-113 mm against its 69.5 mm, 45 mm of it behind the
            # handle; the middle then put the bowl 26-35 mm from the jaws
            # where pick_up measured 51-55 (and 2026-09-25 validated 51), and
            # the scoop dug 2-3 cm further forward than planned. The sim's
            # cloud ran 64 mm.
            print(f"[Skills] the cloud at the pick ran {seen * 1000:.0f} mm against "
                  f"the part's {float(length) * 1000:.0f} mm: its middle is not the "
                  f"part's, so not placing the bowl from it")
            return None
        x = (extent[0] + extent[1]) / 2 + np.sign(cad[0]) * to_bowl
        if x * cad[0] <= 0 or abs(x - cad[0]) > 0.03:
            return None                 # not the part we know: keep the CAD
        measured = (getattr(self, "held_measure", None) or {}).get("suggested")
        if measured is not None and abs(float(measured[0]) - x) > 0.011:
            # When the cloud is the part, the middle of its ends and pick_up's
            # bowl-end median agree to within 8 mm (sim: 6.2, 8.1, 7.0). On
            # the robot they were 14-29 mm apart every time, the ends always
            # the nearer: 21.6 against 36 on 2026-10-08, with a cloud only
            # 14.5 mm too long, and the scoop dug that much too far forward.
            print(f"[Skills] the tool's ends put the bowl {x * 1000:.1f} mm out, the "
                  f"bowl end of its cloud {float(measured[0]) * 1000:.1f} mm: "
                  f"{abs(float(measured[0]) - x) * 1000:.0f} mm apart, so the ends "
                  f"are not the part's; using the measured bowl")
            self._ends_disagree = True
            return None
        return [float(x), cad[1], cad[2]]

    def _fill_held_container(self, skill_name: str, params: dict) -> dict:
        """
        Give pour the held cup's shape, as pick_up measured it, so it can keep
        the cup's lip just over the target. Without it pour falls back to its
        old fixed site, 15 cm above the target (sim, 2026-10-03: the lip ran
        155-185 mm over the target's rim).
        """
        if skill_name != "pour":
            return params
        shape = (getattr(self, "held_measure", None) or {}).get("container")
        if not shape or shape.get("radius", 0) <= 0:
            return params
        filled = dict(params)
        for key, value in (("source_top_above_tcp", shape["top_above_tcp"]),
                           ("source_height", shape["height"]),
                           ("source_radius", shape["radius"])):
            if filled.get(key) is None:
                filled[key] = value
        return filled

    def _fill_tool_geometry(self, skill_name: str, skill, params: dict) -> dict:
        from .tool_geometry import lookup
        params = self._fill_held_container(skill_name, params)
        tool = lookup(self.held)
        if tool is None:
            return params
        known = set(getattr(skill, "optional_params", {}) or {})
        filled = dict(params)
        added = []
        for key in ("bowl_length", "bowl_width", "bowl_depth",
                    "tool_span", "tool_back_reach"):
            if key in known and not filled.get(key) and tool.get(key) is not None:
                filled[key] = tool[key]
                added.append(f"{key}={tool[key] * 1000:.1f}mm")
        if "tool_offset" in known and tool.get("tool_offset") is not None:
            cad = [float(v) for v in tool["tool_offset"]]
            given = filled.get("tool_offset")
            wrong_end = (given is not None and abs(cad[0]) > 0.005
                         and float(given[0]) * cad[0] < 0)
            off_side = given is not None and abs(float(given[1]) - cad[1]) > 0.02
            measured = (getattr(self, "held_measure", None) or {}).get("suggested")
            copied = (given is not None and measured is not None
                      and np.allclose(np.asarray(given, float), np.asarray(measured, float),
                                      atol=1e-4))
            self._ends_disagree = False
            trust = getattr(self.vision, "trust_measured_tool_offset", None)
            sane = (measured is not None and float(measured[0]) * cad[0] > 0
                    and abs(float(measured[1]) - cad[1]) <= 0.02)
            trusted = bool(callable(trust) and trust() and sane)
            from_tip = None if trusted else self._offset_from_tip(tool, cad)
            if (given is None or copied) and trusted and not filled.get("tool_length"):
                # This bench's measurement is ruler-checked: 0.053 printed,
                # 52.9 mm measured, scoop in the jaws (2026-10-08).
                filled["tool_offset"] = [float(measured[0]), float(measured[1]), cad[2]]
                added.append(f"tool_offset x/y {float(measured[0]) * 1000:.1f}/"
                             f"{float(measured[1]) * 1000:.1f}mm as measured at the pick "
                             f"(ruler-checked on this bench), z {cad[2] * 1000:.1f}mm from CAD")
            elif (given is None or copied) and from_tip is not None and not filled.get("tool_length"):
                # No offset, or pick_up's own copied over by a planner: the
                # bowl's far edge says better where the jaws closed. A value
                # anyone passed on purpose -- the bench-tuned [0.051, ...] --
                # is not this case and is left alone below.
                filled["tool_offset"] = from_tip
                added.append(f"tool_offset x {from_tip[0] * 1000:.1f}mm from where the "
                             f"tool's ends were seen at the pick, y/z from CAD"
                             + (f" (instead of pick_up's {[round(float(v), 4) for v in given]})"
                                if copied else ""))
            elif (given is None and not filled.get("tool_length") and measured is not None
                  and float(measured[0]) * cad[0] > 0
                  and (self._ends_disagree or abs(float(measured[0]) - cad[0]) > 0.015)
                  and abs(float(measured[1]) - cad[1]) <= 0.02):
                # Nothing passed, the ends cannot be trusted, and pick_up's own
                # x is further from the CAD than its error (7.7 mm in sim): the
                # jaws are not where the CAD assumes. Its x and y off this grasp,
                # with the CAD depth, is how the robot-validated chain ran
                # (2026-09-25: [0.051, 0.009, 0.028]); the robot measured
                # 51-55 mm against the CAD's 25.7 on 2026-10-07/08.
                filled["tool_offset"] = [float(measured[0]), float(measured[1]), cad[2]]
                added.append(f"tool_offset x/y {float(measured[0]) * 1000:.1f}/"
                             f"{float(measured[1]) * 1000:.1f}mm as measured at the pick, "
                             f"z {cad[2] * 1000:.1f}mm from CAD")
            elif given is None and not filled.get("tool_length"):
                filled["tool_offset"] = cad
                added.append(f"tool_offset={cad} (CAD)")
            elif wrong_end or off_side:
                # pick_up takes the bulkier end of the cloud for the bowl, and
                # a grasp well off the handle's middle can tip that count to
                # the handle. x then points the wrong way: on 2026-10-02 a
                # measured -30 mm (CAD +25.7) aimed the scoop 55 mm off, and
                # the hand flung two dishes off the bench. Where along the
                # handle the jaws closed moves x by centimetres, never across
                # the grasp, so a flipped sign is a bad measurement, not a grasp.
                filled["tool_offset"] = cad
                why = ("points to the other end of the tool" if wrong_end
                       else f"is {abs(float(given[1]) - cad[1]) * 1000:.0f} mm to one side")
                added.append(f"tool_offset {[round(float(v), 4) for v in given]} {why} "
                             f"-> CAD {cad} (the measurement is discarded)")
            elif given is not None and float(given[2]) < cad[2]:
                # pick_up's z is a lower bound: the underside of the bowl is
                # occluded, so the cloud stops short of it. Too shallow a z
                # puts the bowl lower than planned -- into a floor.
                filled["tool_offset"] = [float(given[0]), float(given[1]), cad[2]]
                added.append(f"tool_offset z {float(given[2]) * 1000:.1f} -> "
                             f"{cad[2] * 1000:.1f}mm (CAD depth; measured is a lower bound)")
        cad_length = tool.get("tool_length")
        given_length = filled.get("tool_length")
        if ("tool_length" in known and cad_length and given_length is not None
                and float(given_length) < 0.8 * float(cad_length)):
            # A measured length is a lower bound for the same reason z is; a
            # planning model passed pick_up's 0.017 for the stirrer, which
            # would have driven the rod 64 mm deeper than intended.
            filled["tool_length"] = float(cad_length)
            added.append(f"tool_length {float(given_length) * 1000:.1f} -> "
                         f"{float(cad_length) * 1000:.1f}mm (CAD; measured is a lower bound)")
        if added:
            print(f"[Skills] '{tool['name']}' is in the gripper; for {skill_name} "
                  f"using its CAD geometry: " + ", ".join(added))
        return filled

    def _resolve_pick_site(self, skill_name: str, params: dict) -> dict:
        """Turn place's ``target_location`` name into the site it was picked from."""
        if skill_name != "place":
            return params
        target = params.get("target_location")
        if not isinstance(target, str):
            return params
        key = target.strip().lower()
        put_back = {}
        held_key = str(self.held).strip().lower() if self.held else None
        if held_key:
            from .tool_geometry import lookup, names_for, _norm
            tool = lookup(self.held)
            own = {_norm(self.held)}
            if tool is not None:
                # A planning model names the tool its own way ("stirring rod"
                # for what was picked as "stirrer"), or names the tool's home
                # ("stirrer holder"). Either means "put it back", and the home
                # is not a spot to set something BESIDE: on 2026-10-02 that
                # dropped the stirrer on the bench and drove the hand into the
                # beaker.
                own |= names_for(tool) | {_norm(h) for h in tool.get("home", ())}
            if _norm(target) in own:
                key = held_key
            elif held_key in self.pick_sites and not params.get("on_top"):
                # Asked to set it down beside another object. On a full bench
                # that spot -- 10 cm from the other object toward the base --
                # is on a neighbour, and another object's remembered pick site
                # is where that object stands now. The Magic Beaker run (sim,
                # 2026-10-02) put the water cup on the beaker's spot and the
                # beaker onto cup B, then into cup A, which went over and
                # spilled. Put it back where it came from instead.
                print(f"[Skills] '{self.held}' goes back where it was picked up, "
                      f"not beside {target!r}: the spot beside another object is "
                      f"not known to be clear")
                key = held_key
        site = self.pick_sites.get(key)
        if site is None:
            return params           # a real scene object: let vision find it
        from .tool_geometry import lookup
        tool = lookup(self.held) if self.held else None
        if tool is not None:
            put_back = tool.get("put_back", {})
        resolved = {**put_back, **dict(params)}
        resolved["target_location"] = list(site)
        grasp = self.pick_grasps.get(key)
        if grasp is not None and resolved.get("pick_grasp_tcp") is None:
            resolved["pick_grasp_tcp"] = list(grasp)
        drop = float(getattr(self, "pick_z_offsets", {}).get(key) or 0.0)
        if drop < 0 and resolved.get("release_clearance") is None:
            # Picked lower (the beaker, 2026-10-08: -12 mm): set it down as
            # much lower, or it falls the difference.
            from .place import PlaceSkill
            usual = float(PlaceSkill.optional_params.fget(None)["release_clearance"])
            resolved["release_clearance"] = max(0.0, usual + drop)
        if put_back:
            print(f"[Skills] putting '{self.held}' back the way it came out: "
                  + ", ".join(f"{k}={v}" for k, v in put_back.items()))
        print(f"[Skills] placing '{self.held}' back at the site it was picked "
              f"from: [{site[0]:.4f}, {site[1]:.4f}, {site[2]:.4f}]")
        return resolved
