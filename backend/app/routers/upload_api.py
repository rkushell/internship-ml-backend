from fastapi import APIRouter, UploadFile, File, HTTPException
import os
import shutil

router = APIRouter()

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA_DIR = os.path.join(ROOT, "data")

def save_uploaded_file(file: UploadFile, filename: str):
    os.makedirs(DATA_DIR, exist_ok=True)
    file_path = os.path.join(DATA_DIR, filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return file_path

@router.post("/students")
async def upload_students_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV allowed")
    path = save_uploaded_file(file, "students.csv")
    return {"message": "students.csv uploaded", "path": path}

@router.post("/internships")
async def upload_internships_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV allowed")
    path = save_uploaded_file(file, "internships.csv")
    return {"message": "internships.csv uploaded", "path": path}
