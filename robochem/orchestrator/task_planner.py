"""
Task Planner

Generates high-level task plans from natural language descriptions.
"""

from typing import Dict, List, Any
from openai import OpenAI
import json


class TaskPlanner:
    """
    Generates executable task plans from natural language descriptions.
    
    Uses VLM to decompose high-level goals into sequences of
    robot skills with appropriate parameters.
    """
    
    def __init__(self, model: str = "gpt-4o"):
        """
        Initialize task planner.
        
        Args:
            model: VLM model to use
        """
        self.client = OpenAI()
        self.model = model
    
    def generate_plan(
        self, 
        task: str, 
        scene_state: Dict,
        available_skills: List[str]
    ) -> Dict:
        """
        Generate a task plan.
        
        Args:
            task: Natural language task description
            scene_state: Current scene understanding
            available_skills: List of available skill names
            
        Returns:
            Plan dict with 'steps' list
        """
        prompt = self._build_planning_prompt(task, scene_state, available_skills)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": self._get_system_prompt()
                    },
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            plan = json.loads(response.choices[0].message.content)
            return self._validate_plan(plan, available_skills)
            
        except Exception as e:
            print(f"[TaskPlanner] Error generating plan: {e}")
            return {"steps": [], "error": str(e)}
    
    def _get_system_prompt(self) -> str:
        """Get system prompt for task planning."""
        return """You are a task planner for a robotic chemistry system.

Your job is to decompose high-level chemistry tasks into sequences of 
robot actions that can be executed by a Franka Emika arm with a parallel
plate gripper.

Guidelines:
1. Each step should map to exactly one robot skill
2. Steps should be in logical order with dependencies respected
3. Always pick up tools before using them
4. Always place tools back when done (or when switching tools)
5. For chemistry reactions, allow time for reactions to occur
6. Include verification steps for chemistry outcomes

Output Format:
Return a JSON object with a "steps" array. Each step should have:
- action: The skill name to execute
- params: Parameters for the skill
- expected_outcome: What should happen
- verification_type: "visual", "chemistry", or "none"
"""
    
    def _build_planning_prompt(
        self, 
        task: str, 
        scene_state: Dict,
        available_skills: List[str]
    ) -> str:
        """Build the planning prompt."""
        return f"""Plan the following chemistry task:

TASK: {task}

CURRENT SCENE:
{json.dumps(scene_state, indent=2)}

AVAILABLE SKILLS: {available_skills}

Skill descriptions:
- pick_up: Pick up an object (params: object_name, grasp_type)
- place: Place held object (params: target_location, offset)
- pour: Pour contents into target (params: target_container, pour_angle)
- scoop: Scoop powder/material (params: powder_source, scoop_depth)
- dispense: Dispense drops (params: target_container, num_drops)
- stir: Stir container contents (params: target_container, duration)
- move_to: Move to location (params: target, positioning)
- open_gripper: Open gripper (params: width)
- close_gripper: Close gripper (params: force)

Generate a step-by-step plan to accomplish the task.
Remember: You must pick up tools before using them!"""
    
    def _validate_plan(self, plan: Dict, available_skills: List[str]) -> Dict:
        """Validate and clean up the generated plan."""
        if "steps" not in plan:
            return {"steps": [], "error": "No steps in plan"}
        
        valid_steps = []
        for step in plan["steps"]:
            action = step.get("action", "").lower()
            
            # Map common aliases
            action_map = {
                "pickup": "pick_up",
                "grasp": "pick_up",
                "grab": "pick_up",
                "release": "place",
                "drop": "place",
                "mix": "stir",
                "agitate": "stir",
            }
            action = action_map.get(action, action)
            step["action"] = action
            
            # Ensure params exists
            if "params" not in step:
                step["params"] = {}
            
            valid_steps.append(step)
        
        return {"steps": valid_steps}
    
    def replan_from_failure(
        self,
        original_plan: Dict,
        failed_step: Dict,
        failure_reason: str,
        scene_state: Dict,
        available_skills: List[str]
    ) -> Dict:
        """
        Generate a new plan after a failure.
        
        Args:
            original_plan: The plan that failed
            failed_step: The step that failed
            failure_reason: Why it failed
            scene_state: Current scene state
            available_skills: Available skills
            
        Returns:
            New plan
        """
        prompt = f"""The previous plan failed. Please replan.

ORIGINAL PLAN:
{json.dumps(original_plan, indent=2)}

FAILED STEP:
{json.dumps(failed_step, indent=2)}

FAILURE REASON: {failure_reason}

CURRENT SCENE:
{json.dumps(scene_state, indent=2)}

AVAILABLE SKILLS: {available_skills}

Generate a new plan that avoids the failure and completes the task.
Consider what may have gone wrong and how to recover."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            plan = json.loads(response.choices[0].message.content)
            return self._validate_plan(plan, available_skills)
            
        except Exception as e:
            print(f"[TaskPlanner] Error replanning: {e}")
            return {"steps": [], "error": str(e)}
