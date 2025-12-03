from fastapi import APIRouter, UploadFile, File, HTTPException
from backend.app.services.data_service import upload_students_csv, upload_internships_csv
from backend.app.services.train_service import train_all
from backend.app.services.allocate_service import allocate_all, get_dashboard_data, download_outputs

router = APIRouter()


@router.post("/upload/students")
def upload_students(file: UploadFile = File(...)):
    try:
        upload_students_csv(file)
        return {"message": "students.csv uploaded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload/internships")
def upload_internships(file: UploadFile = File(...)):
    try:
        upload_internships_csv(file)
        return {"message": "internships.csv uploaded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/train")
def train():
    """
    Heavy: trains models (run locally ideally).
    """
    try:
        return train_all()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/allocate")
def allocate():
    """
    Fast: run allocation using already-trained models.
    """
    try:
        return allocate_all()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/dashboard")
def dashboard():
    """
    Returns cached last_results.json contents (dashboard data).
    """
    try:
        return get_dashboard_data()
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail="Run /admin/allocate at least once before fetching dashboard data.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/download/{fname}")
def download_file(fname: str):
    """
    Download outputs by filename (final_allocations.csv, final_fairness_report.json, student_boost_impact.json, etc.)
    Ensure the file exists in output/ or json_outputs/
    """
    try:
        return download_outputs(fname)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
