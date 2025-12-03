from pydantic import BaseModel
from typing import Optional

class StudentInput(BaseModel):
    skills: str
    gpa: float
    reservation: str = "GEN"
    gender: str = "M"
    rural: int = 0

class InternshipInput(BaseModel):
    req_skills: str
    stipend: float = 0.0
    tier: Optional[str] = None

class PredictRequest(BaseModel):
    student: StudentInput
    internship: InternshipInput

class PredictResponse(BaseModel):
    match_score: float
    accept_score: float
    final_score: float
