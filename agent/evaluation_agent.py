import json
from dataclasses import dataclass
from typing import Any, Dict, List, Literal
from pathlib import Path
from .base_agent import BaseLLMAgent


def diff_ui(before: List[dict], after: List[dict]) -> Dict[str, list]:
            added   = [e for e in after if e not in before]
            removed = [e for e in before if e not in after]
            # you can also compare bbox overlap or text changes
            return {"added": added, "removed": removed}


@dataclass
class EvaluationAgent(BaseLLMAgent):
    """Check whether a UI action achieved its goal and explain why."""

    
    def evaluate(
        self,
        plan: List[Dict[str, Any]],
        step_idx: int,
        action: Dict[str, Any],
        ui_before: List[Dict[str, Any]],
        ui_after: List[Dict[str, Any]],
        screenshots: List[Path | str] | None = None
    ) -> Dict[str, Any]:

        expected = plan[step_idx].get("goal", "")
        ui_delta = diff_ui(ui_before, ui_after)


        sys = (
            self.system_prompt
            or "You are an evaluation agent, you are provided with 2 screenshots and its elements, one is the current screen, another is the screen after execute the action. "
                "You are also given a plan, and the action that exectuted from the plan you need to evaluate whether the next screen is on the right track to achieved its goal. "
                "Always follow the checklist below\n"
                "Checklist:\n"
                "1. Understand the entire tentative plan"
                "2. Identify the current step and its action.\n"
                "3. Locate affected element(s), in ui_before.\n"
                "4. Locate same or new element(s) in ui_after.\n"
                "5. Compare outcome to expected, also look at the title, image and content,then use provided hints as the guide for better judgment.\n"
                "6. Decide result: success | fail | task_completion.\n"
                "Heuristics:\n"
                "• click  → element becomes focused / page navigates / element disappears.\n"
                "• type   → text field now contains expected text.\n"
                "• scroll → viewport y-offset changed OR new elements visible.\n"
        )
# You should evaluate the action based on the current tentative plan and the UI state before and after the action was executed
#         to see if the action achieved its goal.
        user_prompt = f"""
        You are provided with some hints to help you evaluate the action:
        - If the action was a scroll or swipe, but the screen did not change, the action likely failed
        - If the action is a click check if the expected element is present in the UI after the action
        - If the action was to click on a text field, if it has the letter "I" in the begining of text box, the action likely succeeded
        - If the action was to type text, check if the text field has the expected text in it
        
        You are give the {json.dumps(plan)} as the current tentative plan,
        the action is {json.dumps(action)},
        the current goal of this action if {json.dumps(expected)}
        the UI state before the action is {json.dumps(ui_before)},
        and the UI state after the action is {json.dumps(ui_after)}.
        The different between 2 Ui is {json.dumps(ui_delta)}
        If the result is fail, you should also explain why the action failed and what could be done to fix it.
        You should return a JSON object with the following keys

        {{
          evaluation_criteria: string,      //describing the criteria you used to evaluate the action
          result: string,                   //"success" when the outcome match with the expected output or meet with the given hint, "fail" when the outcome doesn't meet the expectation, or "task_completion" when the task is success and there are no other step in the plan.
          explanation: string,              //explaning why you reached this conclusion.
          fix: string, optional,            //explaning how to fix the action if it failed or empty string if the action succeeded.
          request: string, optional         //refresh or back to the previous step if the action failed, or empty string if the action succeeded. 

        }}
        Do not return any other keys or values, and do not include any markdown fences.
       """
        
        if screenshots:
            raw = self.call(
                user_text=user_prompt,
                screenshots=screenshots,     
                system_prompt=sys,   
            )

        else:
            raw = self.call(
                user_text=user_prompt,
                screenshots=None,             # text-only path
                system_prompt=sys,
            )   
        try:
            return raw
        except json.JSONDecodeError as exc:
            raise ValueError(f"Evaluation output not valid JSON: {raw}") from exc

# plan = [{"step_idx": 1, "step": "Tap on the search bar labeled 'Search Amazon'", "result": "The search bar becomes active and ready for input.", "reason": "Activating the search bar is necessary to enter the desired search term.", "status": "todo"}, {"step_idx": 2, "step": "Type 'openswim shokz pro' into the search bar", "result": "The text 'openswim shokz pro' appears in the search bar.", "reason": "Entering the search term is essential to find the desired product.", "status": "todo"}, {"step_idx": 3, "step": "Press the search button or hit enter", "result": "Search results related to 'openswim shokz pro' are displayed.", "reason": "Submitting the search term is required to view the relevant products.", "status": "todo"}]       
# step = {"id": 0, "content": "Search Amazon", "bbox": [0.2691798806190491, 0.025198938325047493, 0.34259259700775146, 0.0517241396009922], "action": "click", "text": "", "reason": "Activating the search bar is necessary to enter the desired search term."}
# ui_before = Path("../parser/parsed_1.json") 
# ui_after = Path("../parser/parsed_2.json")           
# with ui_before.open() as f:
#     data = json.load(f) 
#     entities_1 = data["elements"] 
# with ui_after.open() as f:  
#     data = json.load(f) 
#     entities_2 = data["elements"]

# screenshots = [Path("../screenshots/screen_1.png", Path("../screenshots/screen_2.png"))] 
# print (EvaluationAgent().evaluate(plan,1,step, entities_1, entities_2, screenshots=screenshots))


