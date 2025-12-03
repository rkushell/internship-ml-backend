"""
utils.py
Utility helpers for:
 - directory creation
 - JSON saving
"""

import os
import json


def ensure_dirs(*paths):
    """
    Create all directories passed to this function.
    If directory exists, ignore silently.
    """
    for p in paths:
        if p is None:
            continue
        os.makedirs(p, exist_ok=True)


def save_json(path, data):
    """
    Saves a Python dict to a JSON file with indentation.

    Args:
        path: full file path
        data: dict to be written
    """
    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    with open(path, "w") as f:
        json.dump(data, f, indent=4)
