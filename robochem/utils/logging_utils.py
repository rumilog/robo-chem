"""
Logging Utilities

Experiment logging and data recording.
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import numpy as np
from PIL import Image


class ExperimentLogger:
    """
    Logs experiment data including images, actions, and results.
    
    Directory structure:
    experiment_dir/
    ├── metadata.json           # Experiment metadata
    ├── log.json                # Full execution log
    ├── step_0/
    │   ├── pre_action.png      # Pre-action image
    │   ├── post_action.png     # Post-action image
    │   ├── action.json         # Action details
    │   └── result.json         # Step result
    ├── step_1/
    │   └── ...
    └── final/
        ├── result.json         # Final verification result
        └── scene.png           # Final scene image
    """
    
    def __init__(self, base_dir: str, experiment_name: str = None):
        """
        Initialize experiment logger.
        
        Args:
            base_dir: Base directory for experiments
            experiment_name: Optional experiment name (auto-generated if not provided)
        """
        self.base_dir = base_dir
        
        # Generate experiment name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if experiment_name:
            self.experiment_name = f"{experiment_name}_{timestamp}"
        else:
            self.experiment_name = f"experiment_{timestamp}"
        
        self.experiment_dir = os.path.join(base_dir, self.experiment_name)
        
        # Create directories
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # Initialize log
        self.log = {
            "experiment_name": self.experiment_name,
            "start_time": datetime.now().isoformat(),
            "steps": [],
            "metadata": {}
        }
        
        self.current_step = 0
    
    def log_metadata(self, metadata: Dict[str, Any]):
        """
        Log experiment metadata.
        
        Args:
            metadata: Dict with experiment metadata (task, config, etc.)
        """
        self.log["metadata"] = metadata
        self._save_log()
        
        # Save metadata separately
        metadata_path = os.path.join(self.experiment_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    def start_step(self, step_name: str, action: Dict[str, Any]) -> str:
        """
        Start logging a new step.
        
        Args:
            step_name: Name/description of the step
            action: Action parameters
            
        Returns:
            Step directory path
        """
        step_dir = os.path.join(self.experiment_dir, f"step_{self.current_step}")
        os.makedirs(step_dir, exist_ok=True)
        
        step_data = {
            "step_num": self.current_step,
            "step_name": step_name,
            "action": action,
            "start_time": datetime.now().isoformat(),
            "result": None,
            "images": {}
        }
        
        # Save action
        action_path = os.path.join(step_dir, "action.json")
        with open(action_path, 'w') as f:
            json.dump(action, f, indent=2, default=str)
        
        self.log["steps"].append(step_data)
        self._save_log()
        
        return step_dir
    
    def log_image(
        self, 
        image: np.ndarray, 
        name: str, 
        step_dir: str = None
    ) -> str:
        """
        Save an image to the current step directory.
        
        Args:
            image: Image array (HxWx3)
            name: Image name (e.g., "pre_action", "post_action")
            step_dir: Step directory (uses current step if not provided)
            
        Returns:
            Path to saved image
        """
        if step_dir is None:
            step_dir = os.path.join(self.experiment_dir, f"step_{self.current_step}")
        
        os.makedirs(step_dir, exist_ok=True)
        
        image_path = os.path.join(step_dir, f"{name}.png")
        
        # Convert to PIL and save
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        
        pil_image = Image.fromarray(image)
        pil_image.save(image_path)
        
        # Update log
        if self.log["steps"] and self.current_step < len(self.log["steps"]):
            self.log["steps"][self.current_step]["images"][name] = image_path
        
        return image_path
    
    def end_step(self, success: bool, result: Dict[str, Any]):
        """
        End the current step and log results.
        
        Args:
            success: Whether step succeeded
            result: Step result data
        """
        if self.current_step < len(self.log["steps"]):
            self.log["steps"][self.current_step]["result"] = result
            self.log["steps"][self.current_step]["success"] = success
            self.log["steps"][self.current_step]["end_time"] = datetime.now().isoformat()
        
        # Save result to file
        step_dir = os.path.join(self.experiment_dir, f"step_{self.current_step}")
        result_path = os.path.join(step_dir, "result.json")
        
        with open(result_path, 'w') as f:
            json.dump({"success": success, **result}, f, indent=2, default=str)
        
        self.current_step += 1
        self._save_log()
    
    def log_verification(self, verification_result: Dict[str, Any], final_image: np.ndarray = None):
        """
        Log final verification result.
        
        Args:
            verification_result: Verification data
            final_image: Optional final scene image
        """
        final_dir = os.path.join(self.experiment_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        
        # Save verification result
        result_path = os.path.join(final_dir, "result.json")
        with open(result_path, 'w') as f:
            json.dump(verification_result, f, indent=2, default=str)
        
        # Save final image if provided
        if final_image is not None:
            self.log_image(final_image, "scene", final_dir)
        
        # Update main log
        self.log["final_verification"] = verification_result
        self.log["end_time"] = datetime.now().isoformat()
        self._save_log()
    
    def finish(self, success: bool):
        """
        Finish experiment and save final log.
        
        Args:
            success: Overall experiment success
        """
        self.log["success"] = success
        self.log["end_time"] = datetime.now().isoformat()
        
        # Calculate duration
        start = datetime.fromisoformat(self.log["start_time"])
        end = datetime.fromisoformat(self.log["end_time"])
        self.log["duration_seconds"] = (end - start).total_seconds()
        
        self._save_log()
        print(f"[ExperimentLogger] Experiment saved to: {self.experiment_dir}")
    
    def _save_log(self):
        """Save current log to file."""
        log_path = os.path.join(self.experiment_dir, "log.json")
        with open(log_path, 'w') as f:
            json.dump(self.log, f, indent=2, default=str)
    
    @staticmethod
    def load_experiment(experiment_dir: str) -> Dict:
        """
        Load a saved experiment.
        
        Args:
            experiment_dir: Path to experiment directory
            
        Returns:
            Experiment log dictionary
        """
        log_path = os.path.join(experiment_dir, "log.json")
        with open(log_path, 'r') as f:
            return json.load(f)
