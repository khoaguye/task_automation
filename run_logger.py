from pathlib import Path
import json, datetime, os
from typing import List, Dict, Optional, Any

RUNS_DIR = Path("runs")
RUNS_DIR.mkdir(exist_ok=True)

# will be set once at the start of each run
RUN_ID: str | None = None

def init_run(run_id: str) -> None:
    global RUN_ID
    RUN_ID = run_id

def log_agent(
    agent: str,
    step_idx: int,
    payload: dict | list,
    img_before: str = None,   
    img_after:  str = None,   
) -> None:
    if RUN_ID is None:
        raise RuntimeError("RUN_ID not initialised")
    record = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "agent":     agent,
        "step_idx":  step_idx,
        "payload":   payload,
        "img_before": img_before,  # optional
        "img_after":  img_after,   # optional
    }
    fp = RUNS_DIR / f"{RUN_ID}_agentlog.jsonl"
    with fp.open("a") as f:
        for key, value in record.items():
                    f.write(f"{key}: {json.dumps(value, ensure_ascii=False)}\n")
        f.write("\n")
        f.flush()          # crash-safe
