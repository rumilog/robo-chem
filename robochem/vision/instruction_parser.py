"""
Instruction Parser

Reads and parses instructions from images (paper/screen).
Uses OCR + VLM to extract structured task information.
"""

from typing import Dict, List, Optional
from openai import OpenAI
import base64
import json
import cv2
import numpy as np


class InstructionParser:
    """
    Parses chemistry instructions from images.
    
    Supports:
    - Printed instructions with text and diagrams
    - Handwritten notes
    - Lab protocol sheets
    - Chemistry kit instruction cards
    """
    
    def __init__(self, api_key: str = None):
        """
        Initialize instruction parser.
        
        Args:
            api_key: OpenAI API key (uses env var if not provided)
        """
        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
    
    def parse_instruction_image(self, image_path: str) -> Dict:
        """
        Parse an image containing chemistry instructions.
        
        Args:
            image_path: Path to instruction image
            
        Returns:
            Structured instruction data:
            {
                "goal": str,  # Overall task goal
                "steps": List[str],  # Ordered steps
                "reagents": List[str],  # Required reagents
                "equipment": List[str],  # Required equipment
                "safety_notes": List[str],  # Safety warnings
                "expected_outcome": str,  # Expected result
                "quantities": Dict[str, str],  # Reagent quantities if specified
            }
        """
        base64_image = self._encode_image_file(image_path)
        return self._parse_image_content(base64_image)
    
    def parse_instruction_frame(self, image: np.ndarray) -> Dict:
        """
        Parse a camera frame showing instructions.
        
        Args:
            image: numpy array (HxWx3, RGB or BGR)
            
        Returns:
            Structured instruction data
        """
        # Ensure RGB
        if image.shape[2] == 3:
            # Assume BGR if coming from OpenCV
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Encode to base64
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        base64_image = base64.b64encode(buffer).decode('utf-8')
        
        return self._parse_image_content(base64_image)
    
    def _parse_image_content(self, base64_image: str) -> Dict:
        """
        Parse instruction content from base64 encoded image.
        
        Args:
            base64_image: Base64 encoded image string
            
        Returns:
            Structured instruction data
        """
        prompt = """You are analyzing an image containing chemistry experiment instructions.

Extract the following information in a structured format:

1. GOAL: The overall objective/goal of the experiment (what should be achieved)

2. STEPS: The step-by-step procedure as an ordered list. Each step should be:
   - Actionable (something the robot can do)
   - Specific about which reagents/tools to use
   - In sequential order

3. REAGENTS: All chemicals, substances, or materials mentioned
   - Include any quantities if specified

4. EQUIPMENT: All tools, containers, or equipment needed
   - Beakers, cups, pipettes, spoons, etc.

5. SAFETY_NOTES: Any safety warnings or precautions mentioned
   - "None" if not applicable

6. EXPECTED_OUTCOME: What the result should look like
   - Color change, fizzing, volume change, etc.

7. QUANTITIES: Specific amounts for each reagent (if mentioned)
   - e.g., {"sodium bicarbonate": "1 teaspoon", "water": "50ml"}

Return as JSON with these exact keys:
goal, steps, reagents, equipment, safety_notes, expected_outcome, quantities

If any information is not present in the image, use null for that field.
For steps, try to infer a logical order even if not explicitly numbered."""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are an expert at reading and interpreting chemistry experiment instructions. Extract information precisely and completely."
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url", 
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}",
                                    "detail": "high"
                                }
                            }
                        ]
                    }
                ],
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Normalize the result
            return self._normalize_result(result)
            
        except Exception as e:
            print(f"[InstructionParser] Error parsing instructions: {e}")
            return {
                "goal": None,
                "steps": [],
                "reagents": [],
                "equipment": [],
                "safety_notes": [],
                "expected_outcome": None,
                "quantities": {},
                "error": str(e)
            }
    
    def _normalize_result(self, result: Dict) -> Dict:
        """
        Normalize parsed result to consistent format.
        
        Args:
            result: Raw parsed result
            
        Returns:
            Normalized result with all required keys
        """
        normalized = {
            "goal": result.get("goal"),
            "steps": result.get("steps", []) or [],
            "reagents": result.get("reagents", []) or [],
            "equipment": result.get("equipment", []) or [],
            "safety_notes": result.get("safety_notes", []) or [],
            "expected_outcome": result.get("expected_outcome"),
            "quantities": result.get("quantities", {}) or {}
        }
        
        # Ensure lists are actually lists
        for key in ["steps", "reagents", "equipment", "safety_notes"]:
            if not isinstance(normalized[key], list):
                normalized[key] = [normalized[key]] if normalized[key] else []
        
        # Ensure quantities is a dict
        if not isinstance(normalized["quantities"], dict):
            normalized["quantities"] = {}
        
        return normalized
    
    def _encode_image_file(self, image_path: str) -> str:
        """
        Encode image file to base64.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Base64 encoded string
        """
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    def extract_task_from_text(self, text: str) -> Dict:
        """
        Extract structured task from natural language text.
        
        Useful for verbal or typed instructions rather than images.
        
        Args:
            text: Natural language task description
            
        Returns:
            Structured instruction data
        """
        prompt = f"""Parse this chemistry task description into structured format:

"{text}"

Extract:
1. goal: The main objective
2. steps: Ordered list of actions needed
3. reagents: Chemicals/materials involved
4. equipment: Tools needed
5. expected_outcome: What should happen

Return as JSON with keys: goal, steps, reagents, equipment, expected_outcome"""

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Parse chemistry tasks into structured format."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            return self._normalize_result(result)
            
        except Exception as e:
            print(f"[InstructionParser] Error parsing text: {e}")
            return {
                "goal": text,  # Use original text as goal
                "steps": [],
                "reagents": [],
                "equipment": [],
                "expected_outcome": None,
                "error": str(e)
            }
