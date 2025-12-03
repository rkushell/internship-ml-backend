from fastapi import APIRouter
from pydantic import BaseModel
from backend.app.services.model_service import predict_score

router = APIRouter(tags=["Student"])

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

@router.post("/predict")
def predict(data: PredictRequest):
    return predict_score(data)
