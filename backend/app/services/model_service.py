import os
import pandas as pd
from typing import Dict
from src.models import load_models_and_vectorizer, score_all_pairs

# Lazy-load models once
_MODEL_CACHE = None

def _load():
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        _MODEL_CACHE = load_models_and_vectorizer()
    return _MODEL_CACHE

def predict_score(payload) -> Dict:
    """
    payload: pydantic object with .student and .internship
    Returns a dict with match_score, accept_score, final_score
    """
    model_match, model_accept, vectorizer = _load()

    s = payload.student
    j = payload.internship

    df = pd.DataFrame([{
        "student_id": "tmp",
        "internship_id": "tmp",
        "skills": s.skills,
        "req_skills_job": j.req_skills,
        "gpa": s.gpa,
        "stipend_internship": j.stipend,
        "reservation": s.reservation,
        "gender": s.gender,
        "rural": int(s.rural),
        "pref_rank": 1
    }])

    X = score_all_pairs.__globals__  # not used; we'll call score_all_pairs directly below
    # Use score_all_pairs function properly:
    scored = score_all_pairs(df, model_match, model_accept, vectorizer)

    match = float(scored.iloc[0]["match_score"])
    accept = float(scored.iloc[0]["accept_score"])
    final = match * accept

    return {"match_score": match, "accept_score": accept, "final_score": final}
