# main.py
from __future__ import annotations
import json, pathlib, argparse, datetime as dt
from typing import Tuple

# ───────── project imports ─────────
      
from agent.graph_builder import workflow, CycleState, browser  
from agent.browser_agent import BrowserAgent 
from run_logger import init_run, log_agent, RUNS_DIR


# ───────── helpers ─────────────────
RUNS_DIR = pathlib.Path("runs")
RUNS_DIR.mkdir(exist_ok=True)

def new_run_id() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")

def log_event(run_id: str, node: str, state: CycleState) -> None:
    print(f"[{run_id}] {node:16}  "
          f"step={state.step_idx:<3}  "
          f"status={state.status:<14}  "
          f"retries={state.retries}")
    

def persist_state(run_id: str, state: CycleState) -> None:
    fp = RUNS_DIR / f"{run_id}.json"
    with fp.open("w") as f:
        json.dump(state.model_dump(mode="json"), f, indent=2)

# ───────── main routine ────────────
def run(app_name:str, task: str, url: str) -> Tuple[str, CycleState]:
    run_id = new_run_id()
    init_run(run_id)
    try:
        browser.open(url)
        print(f"[main] BrowserAgent id: {id(browser)} | driver is None? {browser.driver is None}")

        print(f"[{run_id}] Navigated to {url}")

        # 2) seed initial state
        init_state = CycleState(run_id=run_id, app_name=app_name, task=task)

        # 3) execute the graph, streaming every node
        
        for state_dict in workflow.stream(init_state, stream_mode="values"):
            state = CycleState(**state_dict)      # re-create the Pydantic model
            node  = state_dict.get("_last_node") or "?"   # if you stored it yourself
            log_event(run_id, node, state)
            #log_agent(node, state.step_idx, state.action)

            if node == "update_state":            # snapshot once per cycle
                persist_state(run_id, state)
                
                #log_action(run_id, state) 

    # 4) final persist and summary
    except Exception as exc:
        import logging
        logging.exception("Run %s crashed: %s", run_id, exc)
        # make sure we still have the last good `state`
    finally:
        if "state" in locals():
            persist_state(run_id, state)
            #log_action(run_id, state)
    print(f"\n[{run_id}] Finished with status = {state.status}")
    print(f"Saved trace to {RUNS_DIR / (run_id + '.json')}")
    return run_id, state

__all__ = ["log_agent"]
# ───────── CLI entry-point ─────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the Amazon book-search multi-agent workflow")
    
    parser.add_argument(
        "--app_name",
        default="Youtube",
        help="Name of the app"
    )
    parser.add_argument(
        "--task",
        default="Youtube app already open, find trending show, and play the first video that pop up",
        help="Natural-language task for the PlannerAgent")
    parser.add_argument(
        "--url",
        default="https://www.youtube.com/",
        help="Start URL for the BrowserAgent")

    args = parser.parse_args()

    run_id, _ = run(args.app_name, args.task, args.url)
