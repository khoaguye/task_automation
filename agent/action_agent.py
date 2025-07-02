from __future__ import annotations

"""ActionAgent – converts a high‑level step into a concrete driver command."""

from dataclasses import dataclass
from typing import Any, Dict, List
import json
from .base_agent import BaseLLMAgent
from pathlib import Path

@dataclass
class ActionAgent(BaseLLMAgent):
    """Produces a low‑level driver JSON command from a step description."""

    def decide(self, step: Dict[str, Any], entities: List[Dict[str, Any]], screenshots: List[Path | str] | None = None) -> str:
        #indexed = [{"id": i, **e} for i, e in enumerate(entities)]
        ent_snip = json.dumps(entities)
        #print(f"{indexed[:10]}")  # Debug: print first 1000 chars of entities
        sys = (
            self.system_prompt
            or "You are an action agent to plan one action to execute the step given, you returns ONE JSON OBJECT only; no extra text."
        )
        user_prompt = f"""
             Your goal is to: {step["reason"]}, you need to execute the step: {step["step"]} to have the expected result like {step["result"]}, 

            UI elements (each has id, content, bbox):
            {ent_snip}

            The screenshot of the current screen is provided to help you decide what action to take at what element.

            Select **one** element (by id).  Copy its content and bbox verbatim.
    
            {{
            "id": int,                             // id of the element to interact with, e.g. 0, 1, 2...
            "content": string,                      // name of the UI element to interact with which extract the content from entities.             
            "action": string,                       // action to perform on the element, e.g. click", "double_click", "right_click", "hover", "type", "scroll", "wait"
            "text": string,                         // this is empty when action is not type.
            "reason": string,                       // why this action is necessary to achieve the goal.
            }}

            Return only the JSON object, no markdown fences.
            """.strip()

        if screenshots:
            raw = self.call(
                user_text=user_prompt,
                screenshots=screenshots,      # pass list directly
                system_prompt=sys,
                
            )
        else:
            raw = self.call(
                user_text=user_prompt,
                screenshots=None,             # text-only path
                system_prompt=sys,
                
            )
        #"bbox": [float, float, float, float],   // exactly to the bbox of the element that match with the id [x1, y1, x2, y2].
        print("Step from action_agent file", {step["step"]})
        
        bbox = []
        for element in entities:
            if element["id"] == raw.get("id") and element["content"] == raw.get("content"):
                bbox = element["bbox"]
                raw["bbox"] = bbox
                break
        else:
                raise ValueError("Attribute is not compatible.")
        print(f"ActionAgent raw output: {raw}")
        try:
            #print(type(json.loads(raw)))
            return raw
        except json.JSONDecodeError as exc:
            raise ValueError(f"Action output not valid JSON: {raw}") from exc

    def repair_action(self, step: Dict[str, Any], explanation: str, fix: str,  entities: List[Dict[str, Any]], screenshots: List[Path | str] | None = None) -> Dict[str, Any]:
        """Repair the action if it failed."""
        sys = (
            self.system_prompt
            or "You are an action agent to repair the action to execute the step given base on the explaination and the suggestion of how to fix the error, you returns ONE JSON OBJECT only; no extra text."
        )
        user_prompt = f"""
            You are given a step: {step["step"]} to achieve the goal: {step["reason"]}, 
            but the action you took failed, you need to repair the action to achieve the goal.
            The explaination of the failure is: {explanation}, and the suggestion of how to fix the error is: {fix}.
            UI elements (each has id, content, bbox):  
            {json.dumps(entities)}   

            The screenshot of the current screen is provided to help you decide what action to take at what element.

            Select **one** element (by id).  Copy its content and bbox verbatim.
    
            {{
            "id": int,                             // id of the element to interact with, e.g. 0, 1, 2...
            "content": string,                      // name of the UI element to interact with which extract the content from entities.             
            "action": string,                       // action to perform on the element, e.g. click", "double_click", "right_click", "hover", "type", "scroll", "wait"
            "text": string,                         // this is empty when action is not type.
            "reason": string,                       // why this action is necessary to achieve the goal.
            }}

            Return only the JSON object, no markdown fences.
        """.strip()

        if screenshots:
            raw = self.call(
                user_text=user_prompt,
                screenshots=screenshots,
                system_prompt=sys,
                
            )
        else:
            raw = self.call(
                user_text=user_prompt,
                screenshots=None,
                system_prompt=sys,
                
            )
        print("Step from action_agent file", {step["step"]})
        print(f"ActionAgent raw output: {raw}")
        bbox = []
        for element in entities:
            if element["id"] == raw.get("id") and element["content"] == raw.get("content"):
                bbox = element["bbox"]
                raw["bbox"] = bbox
                break
        else:
                raise ValueError("Attribute is not compatible.")
        try:
            #print(type(json.loads(raw)))
            return raw
        except json.JSONDecodeError as exc:
            raise ValueError(f"Action output not valid JSON: {raw}") from exc



# step ={'order': 2, 'step': 'Type openswim shokz pro into the search bar.', 'result': 'The search term is entered into the search bar', 'reason': 'Entering the specific search term is necessary to find the desired product on Amazon.', 'status': 'todo'}
# out_path = Path("../parser/parsed_output_20250624_114202.json")            
# with out_path.open() as f:
#     data = json.load(f) 
#     entities = data["elements"] 
# screenshots = [Path("../screenshots/screen_20250624_114132_20250624_114137.png")] 
# print (ActionAgent().decide(step, entities))



"""
input:
step = {
        "order": int,          // 1-based index of the step
        "step": string,       // what action the user/agent should perform to achieve the goal
        "reason": string,     // why this step is necessary to reach the goal
        "UI_elements": List[Dict[str, Any]], // list of UI elements to interact with, each element has "content" and "bbox" keys}

output: 
action = {    
        "content": string,    // name of the UI element to interact with which extract the content from entities 
        "bbox": [float, float, float, float], // bounding box coordinates of the element in the format [x1, y1, x2, y2] which extract the bbox from entities            
        "action": string, // action to perform on the element, e.g. click", "double_click", "right_click", "hover", "type", "scroll", "wait"
        "text": string, // this is empty when action is not type
        "reason": string, // why this action is necessary to achieve the goal 
    }
"""
