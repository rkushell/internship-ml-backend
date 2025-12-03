import os
import json
from fastapi import APIRouter, HTTPException

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
JSON_DIR = os.path.join(ROOT, "json_outputs")
RESULT_PATH = os.path.join(JSON_DIR, "last_results.json")

router = APIRouter()


@router.get("/")
def get_dashboard_data():
    """
    Returns the latest allocation + fairness + boost + round logs as JSON.

    Requires that /admin/allocate has been run at least once
    (so that last_results.json exists).
    """
    if not os.path.exists(RESULT_PATH):
        raise HTTPException(
            status_code=400,
            detail="Run /admin/allocate at least once before fetching dashboard data.",
        )

    with open(RESULT_PATH, "r") as f:
        return json.load(f)
