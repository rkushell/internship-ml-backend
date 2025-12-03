import pandas as pd
from backend.app.models import PredictRequest, PredictResponse
from src.models import load_models_and_vectorizer, score_all_pairs

def predict_single_pair(req: PredictRequest) -> PredictResponse:

    model_match, model_accept, vectorizer = load_models_and_vectorizer()

    df = pd.DataFrame([{
        "student_id": "temp_student",
        "internship_id": "temp_intern",
        "skills": req.student.skills,
        "req_skills_job": req.internship.req_skills,
        "gpa": req.student.gpa,
        "stipend_internship": req.internship.stipend,
        "reservation": req.student.reservation,
        "gender": req.student.gender,
        "rural": req.student.rural,
        "pref_rank": 7
    }])

    scored = score_all_pairs(df, model_match, model_accept, vectorizer).iloc[0]

    match_score = float(scored["match_score"])
    accept_score = float(scored["accept_score"])
    final_score = match_score * accept_score * 0.20

    return PredictResponse(
        match_score=match_score,
        accept_score=accept_score,
        final_score=final_score
    )
