from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict

from backend.app.services.model_service import predict_score

router = APIRouter()


class StudentInput(BaseModel):
    skills: str
    gpa: float
    reservation: str
    gender: str
    rural: int


class InternshipInput(BaseModel):
    req_skills: str
    stipend: float


class PredictRequest(BaseModel):
    student: StudentInput
    internship: InternshipInput


@router.post("/predict", response_model=Dict)
def predict(payload: PredictRequest):
    """
    Predict match_score, accept_score, and final_score for a single student-internship pair.
    """
    try:
        return predict_score(payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
