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
RUNS_DIR.mkdir(exist_ok=True)  # Ensure the runs directory exists

def new_run_id() -> str:
    """
    Generates a unique run identifier based on current date and time.

    Returns:
        str: A timestamp-based run ID (e.g., '20250703-101530').
    """
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")


def log_event(run_id: str, node: str, state: CycleState) -> None:
    """
    Logs a simple event to stdout for debugging purposes.

    Parameters:
        run_id (str): The unique run identifier.
        node (str): The name of the current node in the workflow graph.
        state (CycleState): The current state of the workflow cycle.
    """
    print(f"[{run_id}] {node:16}  "
          f"step={state.step_idx:<3}  "
          f"status={state.status:<14}  "
          f"retries={state.retries}")


def persist_state(run_id: str, state: CycleState) -> None:
    """
    Saves the current state of the workflow to disk as a JSON file.

    Parameters:
        run_id (str): The run identifier used to name the file.
        state (CycleState): The current state to persist.
    """
    fp = RUNS_DIR / f"{run_id}.json"
    with fp.open("w") as f:
        json.dump(state.model_dump(mode="json"), f, indent=2)


# ───────── main routine ────────────

def run(app_name: str, task: str, url: str) -> Tuple[str, CycleState]:
    """
    Runs the multi-agent workflow starting from the provided URL and task.

    Parameters:
        app_name (str): Name of the application being tested or controlled.
        task (str): Natural language task to guide the workflow.
        url (str): URL to initialize the BrowserAgent.

    Returns:
        Tuple[str, CycleState]: The run ID and final state after execution.
    """
    run_id = new_run_id()     # Create a unique identifier for this run
    init_run(run_id)          # Initialize the global RUN_ID

    try:
        browser.open(url)     # Launch browser to the target URL
        print(f"[main] BrowserAgent id: {id(browser)} | driver is None? {browser.driver is None}")
        print(f"[{run_id}] Navigated to {url}")

        # Initialize the cycle state
        init_state = CycleState(run_id=run_id, app_name=app_name, task=task)

        # Stream through the workflow graph
        for state_dict in workflow.stream(init_state, stream_mode="values"):
            state = CycleState(**state_dict)                    # Reconstruct the Pydantic model from dict
            node  = state_dict.get("_last_node") or "?"         # Fallback if node name is not found
            log_event(run_id, node, state)                      # Log progress to stdout

            # Persist state snapshot at the update_state node
            if node == "update_state":
                persist_state(run_id, state)
                # log_agent(node, state.step_idx, state.action)  # Optional logging to file

    except Exception as exc:
        # Log unexpected crash
        import logging
        logging.exception("Run %s crashed: %s", run_id, exc)

    finally:
        # Persist last known state even if an error occurred
        if "state" in locals():
            persist_state(run_id, state)
            # log_agent(run_id, state.step_idx, state.action)

    print(f"\n[{run_id}] Finished with status = {state.status}")
    print(f"Saved trace to {RUNS_DIR / (run_id + '.json')}")
    return run_id, state


__all__ = ["log_agent"]

# ───────── CLI entry-point ─────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the Amazon book-search multi-agent workflow"
    )

    parser.add_argument(
        "--app_name",
        default="Youtube",
        help="Name of the app"
    )
    parser.add_argument(
        "--task",
        default="Youtube app already open, find trending show, and play the first video that pop up",
        help="Natural-language task for the PlannerAgent"
    )
    parser.add_argument(
        "--url",
        default="https://www.youtube.com/",
        help="Start URL for the BrowserAgent"
    )

    args = parser.parse_args()

    # Run the full pipeline with provided arguments
    run_id, _ = run(args.app_name, args.task, args.url)