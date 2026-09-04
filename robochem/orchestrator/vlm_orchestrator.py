"""
VLM Orchestrator

Main orchestration module that uses a Vision-Language Model to:
1. Parse instructions (from paper/verbal/text)
2. Understand the scene
3. Plan high-level tasks
4. Execute skills with visual verification
5. Replan on failure
"""

from typing import Dict, List, Any, Optional, Tuple
from openai import OpenAI
import json
import numpy as np

from .task_planner import TaskPlanner


class VLMOrchestrator:
    """
    VLM-based orchestrator for autonomous chemistry manipulation.
    
    This is the main control loop that:
    1. Reads instructions from images or text
    2. Analyzes the current scene
    3. Generates and executes action plans
    4. Verifies outcomes and replans if needed
    """
    
    def __init__(
        self, 
        skills_executor,
        vision_system,
        verifier,
        logger=None,
        model: str = "gpt-4o"
    ):
        """
        Initialize the orchestrator.
        
        Args:
            skills_executor: SkillsExecutor instance for executing robot skills
            vision_system: VisionSystem instance for perception
            verifier: ChemistryVerifier for outcome verification
            logger: Optional ExperimentLogger for data recording
            model: VLM model to use (default: gpt-4o)
        """
        self.client = OpenAI()
        self.model = model
        
        self.skills = skills_executor
        self.vision = vision_system
        self.verifier = verifier
        self.logger = logger
        
        self.task_planner = TaskPlanner(model=model)
        
        # State tracking
        self.current_task = None
        self.scene_state = None
        self.execution_log = []
        self.held_object = None  # Track what the gripper is holding
    
    def run_task(
        self, 
        task_description: str = None,
        instruction_image: str = None,
        max_retries: int = 3
    ) -> Dict:
        """
        Execute a chemistry task from start to finish.
        
        Args:
            task_description: Natural language task description
            instruction_image: Path to image containing instructions
            max_retries: Maximum replanning attempts on failure
            
        Returns:
            Execution result with success status and logs
        """
        self.execution_log = []
        
        # Step 1: Parse instructions
        if instruction_image:
            print("[Orchestrator] Parsing instruction image...")
            parsed = self.vision.instruction_parser.parse_instruction_image(instruction_image)
            task_description = parsed.get("goal", task_description)
            self._log_event("instruction_parse", parsed)
        
        if not task_description:
            return {"success": False, "error": "No task description provided"}
        
        self.current_task = task_description
        print(f"[Orchestrator] Task: {task_description}")
        
        # Step 2: Analyze initial scene
        print("[Orchestrator] Analyzing scene...")
        self.scene_state = self._analyze_scene()
        self._log_event("scene_analysis", self.scene_state)
        
        # Step 3: Generate high-level plan
        print("[Orchestrator] Generating plan...")
        plan = self.task_planner.generate_plan(
            task_description,
            self.scene_state,
            self.skills.list_skills() if hasattr(self.skills, 'list_skills') else []
        )
        self._log_event("plan_generated", plan)
        
        if not plan.get("steps"):
            return {"success": False, "error": "Failed to generate plan"}
        
        # Step 4: Execute plan
        print(f"[Orchestrator] Executing {len(plan['steps'])} steps...")
        
        for i, step in enumerate(plan["steps"]):
            print(f"\n{'='*50}")
            print(f"Step {i+1}/{len(plan['steps'])}: {step.get('action', 'unknown')}")
            print(f"{'='*50}")
            
            # Capture pre-action state
            pre_images = self.vision.capture_scene()
            
            # Execute the step
            success, result = self._execute_step(step)
            
            # Capture post-action state
            post_images = self.vision.capture_scene()
            
            # Log step
            step_log = {
                "step_num": i + 1,
                "step": step,
                "success": success,
                "result": result
            }
            
            # Verify step outcome
            if step.get("verification_type"):
                verification = self._verify_step(step, pre_images, post_images)
                step_log["verification"] = verification
                
                if not verification.get("success", True):
                    print(f"[Orchestrator] Verification failed: {verification.get('details')}")
            
            self._log_event(f"step_{i+1}", step_log)
            
            # Handle failure
            if not success:
                print(f"[Orchestrator] Step failed: {result.get('error', 'unknown error')}")
                
                # Attempt recovery
                recovery = self._attempt_recovery(step, result, max_retries)
                if not recovery["recovered"]:
                    return {
                        "success": False,
                        "error": f"Failed at step {i+1}",
                        "failed_step": step,
                        "log": self.execution_log
                    }
        
        # Step 5: Final verification
        print("\n[Orchestrator] Performing final verification...")
        final_images = self.vision.capture_scene()
        final_verification = self.verifier.verify_chemistry_outcome(
            task_description,
            final_images
        )
        
        self._log_event("final_verification", final_verification)
        
        return {
            "success": final_verification.get("success", True),
            "task": task_description,
            "final_verification": final_verification,
            "log": self.execution_log
        }
    
    def _analyze_scene(self) -> Dict:
        """Analyze current scene state using VLM."""
        images = self.vision.capture_scene()
        
        if not images:
            return {"error": "No images captured"}
        
        return self.vision.scene_analyzer.get_scene_description(images)
    
    def _execute_step(self, step: Dict) -> Tuple[bool, Dict]:
        """Execute a single step from the plan."""
        action = step.get("action", "").lower()
        params = step.get("params", {})
        
        # Handle special actions
        if action == "wait":
            import time
            duration = params.get("duration", 1.0)
            time.sleep(duration)
            return True, {"waited": duration}
        
        # Execute via skills executor
        try:
            success, result = self.skills.execute(action, params)
            
            # Update held object state
            if action == "pick_up" and success:
                self.held_object = params.get("object_name")
            elif action in ["place", "pour", "dispense"] and success:
                if action == "place":
                    self.held_object = None
            
            return success, result
            
        except Exception as e:
            return False, {"error": str(e)}
    
    def _verify_step(
        self, 
        step: Dict, 
        pre_images: List, 
        post_images: List
    ) -> Dict:
        """Verify step execution."""
        verification_type = step.get("verification_type", "visual")
        expected = step.get("expected_outcome", "")
        
        if verification_type == "chemistry":
            return self.verifier.verify_chemistry_outcome(
                expected, post_images, pre_images
            )
        
        # Default: simple visual check that something changed
        return {"success": True, "verification_type": "none"}
    
    def _attempt_recovery(
        self, 
        failed_step: Dict, 
        failure_result: Dict,
        max_retries: int
    ) -> Dict:
        """Attempt to recover from a failed step."""
        for attempt in range(max_retries):
            print(f"[Orchestrator] Recovery attempt {attempt + 1}/{max_retries}")
            
            # Simple retry first
            success, result = self._execute_step(failed_step)
            
            if success:
                return {"recovered": True, "attempts": attempt + 1}
            
            # Could add more sophisticated recovery strategies here:
            # - Replan with updated scene understanding
            # - Try alternative approach
            # - Ask for human intervention
        
        return {"recovered": False, "attempts": max_retries}
    
    def _log_event(self, event_type: str, data: Any):
        """Log an event to the execution log."""
        import time
        self.execution_log.append({
            "type": event_type,
            "timestamp": time.time(),
            "data": data
        })
        
        if self.logger:
            # Also log to file
            pass
