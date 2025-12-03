from fastapi import APIRouter, UploadFile, File
from backend.app.services.data_service import upload_students_csv, upload_internships_csv
from backend.app.services.train_service import train_all
from backend.app.services.allocate_service import allocate_all, get_dashboard_data

router = APIRouter(tags=["Admin"])

@router.post("/upload/students")
def upload_students(file: UploadFile = File(...)):
    upload_students_csv(file)
    return {"message": "students.csv uploaded"}

@router.post("/upload/internships")
def upload_internships(file: UploadFile = File(...)):
    upload_internships_csv(file)
    return {"message": "internships.csv uploaded"}

@router.post("/train")
def train():
    return train_all()

@router.post("/allocate")
def allocate():
    return allocate_all()

@router.get("/dashboard")
def dashboard():
    return get_dashboard_data()
