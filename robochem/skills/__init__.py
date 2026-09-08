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

from .base_skill import BaseSkill

# Manipulation skills
from .pick_up import PickUpSkill
from .place import PlaceSkill
from .pour import PourSkill
from .scoop import ScoopSkill
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
            self._skill_instances[skill_name] = skill_class(
                self.robot, self.vision, self.config.get(skill_name, {})
            )
        
        skill = self._skill_instances[skill_name]
        
        params = {**self.SKILL_ALIASES.get(skill_name, {}), **(params or {})}
        
        # Check preconditions
        can_execute, message = skill.check_preconditions(params)
        if not can_execute:
            return False, {"error": message, "phase": "precondition_check"}
        
        # Execute the skill
        try:
            success, result = skill.execute(params)
            return success, result
        except Exception as e:
            return False, {"error": str(e), "phase": "execution"}
