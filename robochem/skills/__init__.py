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
        # Where the TCP was when each remembered object was grasped. Putting
        # the TCP back there re-seats a tool exactly as it sat (the stirrer in
        # its holder); the centroid alone is only the mean of what was visible.
        self.pick_grasps = {}
    
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
        params = self._resolve_pick_site(skill_name, params)
        
        # Check preconditions
        can_execute, message = skill.check_preconditions(params)
        if not can_execute:
            return False, {"error": message, "phase": "precondition_check"}
        
        # Execute the skill
        try:
            success, result = skill.execute(params)
            if success:
                self._remember_pick_site(skill_name, params, result)
            return success, result
        except Exception as e:
            return False, {"error": str(e), "phase": "execution"}

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
        grasp = result.get("grasp_pose")
        if grasp is not None:
            self.pick_grasps[key] = [float(v) for v in
                                     np.asarray(grasp, dtype=float)[:3, 3]]
        print(f"[Skills] Noted where '{name}' was picked from: "
              f"[{site[0]:.4f}, {site[1]:.4f}, {site[2]:.4f}] — "
              f"place it back by name")

    def _resolve_pick_site(self, skill_name: str, params: dict) -> dict:
        """Turn place's ``target_location`` name into the site it was picked from."""
        if skill_name != "place":
            return params
        target = params.get("target_location")
        if not isinstance(target, str):
            return params
        site = self.pick_sites.get(target.strip().lower())
        if site is None:
            return params           # a real scene object: let vision find it
        resolved = dict(params)
        resolved["target_location"] = list(site)
        grasp = self.pick_grasps.get(target.strip().lower())
        if grasp is not None and resolved.get("pick_grasp_tcp") is None:
            resolved["pick_grasp_tcp"] = list(grasp)
        print(f"[Skills] '{target}' is the tool in the gripper, so it cannot be "
              f"seen on the bench; placing it back at the site it was picked "
              f"from: [{site[0]:.4f}, {site[1]:.4f}, {site[2]:.4f}]")
        return resolved
