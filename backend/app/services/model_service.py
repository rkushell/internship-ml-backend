import pandas as pd
from src.models import load_models_and_vectorizer, score_all_pairs

model_match, model_accept, vectorizer = load_models_and_vectorizer()

def predict_score(payload):
    s = payload.student
    j = payload.internship

    df = pd.DataFrame([{
        "skills": s.skills,
        "req_skills_job": j.req_skills,
        "gpa": s.gpa,
        "stipend_internship": j.stipend,
        "reservation": s.reservation,
        "gender": s.gender,
        "rural": s.rural,
        "pref_rank": 1
    }])

    scored = score_all_pairs(df, model_match, model_accept, vectorizer)

    match = float(scored.iloc[0]["match_score"])
    accept = float(scored.iloc[0]["accept_score"])
    final = match * accept

    return {
        "match_score": match,
        "accept_score": accept,
        "final_score": final
    }
