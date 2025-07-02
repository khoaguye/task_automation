from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from .base_agent import BaseLLMAgent


@dataclass
class PlannerAgent(BaseLLMAgent):
    """Produces or repairs a JSON array of steps to achieve the user goal."""

    def plan(
        self,
        app_name: str,
        user_task: str,
        entities: List[Dict[str, Any]],
        *,
        screenshots: List[Path | str] | None = None,   # ← accept list
    ) -> List[Dict[str, Any]]:
        """Return a list of step dictionaries."""

        sys = (
            self.system_prompt
            or "You are a task-planner agent. Your task is to analyze the screenshots and determine the plan to take based on the user's task request. You will be shown a single screenshot of a phone screen along with information about interactive elements. "
        )

        user_prompt = f"""
            You are using the **{app_name}** app. Your goal is: {user_task}

            Screen entities extracted from the current UI:
            {json.dumps(entities)}

            Produce a JSON array. Each element must follow this exact schema:
            {{
            "step_idx": int,       // 1-based index of the step
            "step":      string,   // what action the user/agent should perform to achieve the goal
            "result":    string,   // what the user/agent expects to happen after this step
            "reason":    string,   // why this step is necessary to reach the goal
            "status":    "todo/ acting/ success/ action_problem/ plan_problem/ done/ fail"
            }}

            Return only the JSON array—no extra commentary and no markdown fences.
            """.strip()

 
        if screenshots:
            result = self.call(
                user_text=user_prompt,
                screenshots=screenshots,      # pass list directly
                system_prompt=sys,
                
            )
        else:
            result = self.call(
                user_text=user_prompt,
                screenshots=None,             # text-only path
                system_prompt=sys,
                
            )
        print("Planner_result", result)

        # -------- normalise output ------------------------------------
        if isinstance(result, list):
            return result
        if isinstance(result, dict) and "steps" in result:
            return result["steps"]

        # 2) the common “single-step dict” case → wrap it
        if isinstance(result, dict) and {"step_idx", "step", "result"}.issubset(result):
            return [result]                 

        raise ValueError(f"Planner output not valid JSON: {result!r}")

    # ------------------------------------------------------------------ #
    def repair_plan(
        self,
        app_name: str,
        user_task: str,
        step_idx: int,
        explanation: str, 
        fix: str,
        entities: List[Dict[str, Any]],
        current_plan: List[Dict[str, Any]],
        *,
        screenshots: List[Path | str] | None = None,  
    ) -> List[Dict[str, Any]]:
        """Repair an existing plan from `step_idx` onward."""

        sys = (
            self.system_prompt or "You are a task-planner agent repairing an existing plan. Return ONLY JSON."
        )

        user_prompt = f"""
            You are using the **{app_name}** app. Your goal is: {user_task}
            The current plan is not working, and you need to repair it.
            Explanation of the problem: {explanation}
            Suggested fix: {fix}
            Current step index to repair from: {step_idx}
            Current plan:
            {json.dumps(current_plan)}

            Current UI entities:
            {json.dumps(entities)}

            Repair the plan by adding/modifying steps. Follow the same JSON schema.
            """.strip()

        if screenshots:
            result = self.call(
                user_text=user_prompt,
                screenshots=screenshots,
                system_prompt=sys,
               
            )
        else:
            result = self.call(
                user_text=user_prompt,
                screenshots=None,
                system_prompt=sys,
                
            )

        print(f"PlannerAgent repair output: {result}")
        if isinstance(result, dict) and "steps" in result:
            return result["steps"]            # type: ignore[return-value]
        if isinstance(result, list):
            return result                     # type: ignore[return-value]

        raise ValueError(f"Planner repair output not valid JSON: {result!r}")
        


# if __name__ == "__main__":
#     sample_entities_path = Path("../parser/parsed_output_20250624_114202.json")
#     with sample_entities_path.open() as f:
#         ents = json.load(f).get("elements", [])

#     before_png = Path("../screenshots/screen_20250624_114132_20250624_114137.png")   
#     print(
#         PlannerAgent().plan(
#             app_name="amazon",
#             user_task="search for 'openswim shokz pro', fine the chepest one and add it to the cart",
#             entities=ents,
#             screenshots=[before_png],              
#         )
#     )












"""
Input: 
    1. Task
    2. Image parsing recognition: OmniParser_usage/api.py
output:
    1. A Json with step by step instructions to complete the task
    ex:
    Step = {
        {
            "number": 1,
            "step": "go to the search bar"
            "reseason": "The search bar is where you can enter the name of the product you want to search for.",
            "status": "todo/success"
         ""        
        },
        {
            "number": 2,
            "step": "type 'The Pragmatic Programmer'",
            "reason": "This is the name of the book you want to search for.",
            "status": "todo/success"
        },
        {
            "number": 3,
            "step": "click the search button",
            "reason": "Clicking the search button will initiate the search for the book.",
            "status": "todo/success"
        }
    
    }
"""