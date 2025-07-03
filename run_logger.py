from pathlib import Path
import json, datetime, os
from typing import List, Dict, Optional, Any

# Directory where all run logs will be saved
RUNS_DIR = Path("runs")
RUNS_DIR.mkdir(exist_ok=True)  # Create directory if it doesn't already exist

# Global variable to track the current run session
RUN_ID: str | None = None

def init_run(run_id: str) -> None:
    """
    Initializes the global run ID used to label log files for a session.

    Parameters:
        run_id (str): A unique identifier for the current run.
    """
    global RUN_ID
    RUN_ID = run_id  # Set the global RUN_ID used for logging

def log_agent(
    agent: str,
    step_idx: int,
    payload: dict | list,
    img_before: str = None,   
    img_after:  str = None,   
) -> None:
    """
    Logs information about an agent's action at a specific step in a process.

    Parameters:
        agent (str): Identifier or name of the agent performing the action.
        step_idx (int): Step index indicating the sequence or order of the action.
        payload (dict | list): Data relevant to the agent's action at this step.
        img_before (str, optional): Path or reference to an image before the action (if applicable). Defaults to None.
        img_after (str, optional): Path or reference to an image after the action (if applicable). Defaults to None.

    Raises:
        RuntimeError: If RUN_ID is not initialized before logging.
    """
    if RUN_ID is None:
        raise RuntimeError("RUN_ID not initialised")  # Ensure logging only occurs after initialization

    # Prepare a record with metadata and payload
    record = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),  # Current time in ISO format
        "agent":     agent,
        "step_idx":  step_idx,
        "payload":   payload,
        "img_before": img_before,  # optional image before action
        "img_after":  img_after,   # optional image after action
    }

    # Open log file for the current run in append mode
    fp = RUNS_DIR / f"{RUN_ID}_agentlog.jsonl"
    with fp.open("a") as f:
        for key, value in record.items():
            # Write each key-value pair as JSON-encoded value
            f.write(f"{key}: {json.dumps(value, ensure_ascii=False)}\n")
        f.write("\n")  # Separate records by a blank line
        f.flush()      # Ensure data is immediately written to disk (crash-safe)