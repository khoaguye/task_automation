from __future__ import annotations
import json, time
from typing import List, Dict, Literal, Optional
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph               
from IPython.display import Image, display

# Agents     
from .browser_agent    import BrowserAgent   
from .planner_agent    import PlannerAgent
from .action_agent     import ActionAgent
from .evaluation_agent import EvaluationAgent
from run_logger import log_agent
from pathlib import Path

browser   = BrowserAgent()
planner   = PlannerAgent()
actor     = ActionAgent()
evaluator = EvaluationAgent()

class CycleState(BaseModel):
    task: str
    app_name: str = ""
    plan: List[Dict]        = Field(default_factory=list)
    step_idx: int           = 0

    img_before: Path | None = None
    img_after:  Path | None = None
    ui_before: List[Dict]   = Field(default_factory=list)
    ui_after:  List[Dict]   = Field(default_factory=list)

    action: Optional[Dict]  = None
    retries: int            = 0
    status: Literal[
        "todo", "acting", "success",
        "action_problem", "plan_problem",
        "done", "fail"
    ] = "todo"     
    explanation: str = ""
    fix: str = ""
    request: str = ""

# ── Node implementations ─────────────────────────────────────────────────────
def perceive_before(state: CycleState) -> CycleState:
    img = browser.take_screenshot()
    state.ui_before = browser._run_omniparser(img).get("elements", [])
    state.img_before = img
    return state

def perceive_after(state: CycleState) -> CycleState:
    
    img = browser.take_screenshot()
    state.ui_after = browser._run_omniparser(img).get("elements", [])
    state.img_after = img
    return state

def tentative_plan(state: CycleState) -> CycleState:
    """Generate or repair a plan."""
    print("call here 1")
    if not state.plan:
        state.plan = planner.plan(
            app_name = state.app_name,
            user_task = state.task,
            entities  = state.ui_before,
            screenshots = [state.img_before] if state.img_before else None
     
        )
        state.step_idx = 0
        #log_agent("planner", state.step_idx, state.plan, str(state.img_before), str(state.img_after))
    elif state.status == "plan_problem" :
        browser.refresh_or_go_back(state.request)  # refresh or go back to the previous step
        state.plan = planner.repair_plan(
            app_name     = state.app_name,
            user_task    = state.task,
            step_idx     = state.step_idx,
            entities     = state.ui_before,
            explanation  = state.explanation,
            fix          = state.fix,
            current_plan = state.plan,
            screenshots  = [state.img_before, state.img_after]
        )
    log_agent("planner", state.step_idx, state.plan, str(state.img_before), str(state.img_after))
    state.status = "acting"
    return state

def decide_action(state: CycleState) -> CycleState:
    """Pick the next low-level action for the current step."""
    # if state.step_idx >= len(state.plan):
    #     state.status = "done"
    #     return state
    step = state.plan[state.step_idx]
    print("step from graph_builder", step)
    if state.status == "action_problem":
        #move the screen back to the previous state then fix the action
        browser.refresh_or_go_back(state.request)  # refresh or go back to the previous step
        state.action = actor.repair_action(step, state.explanation, state.fix ,state.ui_before, screenshots = [state.img_before])  #Create the action agent to repair the action
        state.status = "acting"
        return state
    else:
        state.action = actor.decide(step, state.ui_before, screenshots = [state.img_before] if state.img_before else None)
    
    if state.action is None:
        state.status = "action_problem"
    else:
        state.status = "acting"
    return state

def execute_action(state: CycleState) -> CycleState:
    try:
        browser.execute_action(state.action)
        log_agent("action", state.step_idx, state.action, str(state.img_before), str(state.img_after))
        # After execution we immediately capture ui_after in next node
    except Exception as exc:
        print(f"Action execution failed: {exc}")
        state.status = "action_problem"
    return state

def evaluate_action(state: CycleState) -> CycleState:
    """Judge whether the action achieved the step’s intent."""
    evaluation = evaluator.evaluate(
        plan       = state.plan,
        step_idx   = state.step_idx,
        action     = state.action,
        ui_before  = state.ui_before,
        ui_after   = state.ui_after,
        screenshots = [state.img_before, state.img_after] if state.img_after else None
    )
    
    match evaluation["result"]:
        case "success":
            state.status = "success"
        case "task_completion":
            state.status = "done"
        case "fail":
            state.status = "action_problem"
            state.explanation = evaluation.get("explanation", "")
            state.fix = evaluation.get("fix", "")
            state.request = evaluation.get("request", "")
    log_agent("evaluation", state.step_idx, evaluation, str(state.img_before), str(state.img_after))

    
    return state

def update_state(state: CycleState) -> CycleState:
    """Logic"""
    if state.step_idx >= len(state.plan):
            state.status = "done"

    if state.status == "success":
        state.plan[state.step_idx]["status"] = "success"
        state.step_idx += 1
        state.retries  = 0

        # Prepare for next loop: ui_after ➞ ui_before
        state.ui_before, state.ui_after = state.ui_after, []
        state.img_before, state.img_after = state.img_after, None  # reset after use
    elif state.status == "action_problem":
        state.retries += 1
        if state.retries > 3:
            state.status = "plan_problem"
        # the ui_before is the same, and ui_after is empty
        state.ui_after = []
        state.img_after = None 

    elif state.status == "plan_problem":
        state.retries = 0   # reset, will branch back to `plan`

    return state

#build the graph 
g = StateGraph(CycleState)

g.add_node("perceive_before", perceive_before)
g.add_node("plan_task",       tentative_plan)
g.add_node("decide_action",   decide_action)
g.add_node("execute_action",  execute_action)
g.add_node("perceive_after",  perceive_after)
g.add_node("evaluate_action", evaluate_action)
g.add_node("update_state",    update_state)

g.set_entry_point("perceive_before")

# straight-line backbone
g.add_edge("perceive_before", "plan_task")
g.add_edge("plan_task",        "decide_action")
g.add_edge("decide_action",   "execute_action")
g.add_edge("execute_action",  "perceive_after")
g.add_edge("perceive_after",  "evaluate_action")
g.add_edge("evaluate_action", "update_state")

# conditional routing after bookkeeping

def router(state: CycleState):
    return {
        "success":        "decide_action",
        "fail":           "decide_action",  # retry the same step 
        "acting":         "decide_action",  # first run or normal loop-through
        "action_problem": "decide_action",  # retry the same step
        "plan_problem":   "plan_task",           # re-plan
        "todo":           "plan_task",           # first call
        "done":           "__END__",
        
    }.get(state.status, "__END__")          # safeguard for unknown status

g.add_conditional_edges("update_state", router)

workflow = g.compile()




