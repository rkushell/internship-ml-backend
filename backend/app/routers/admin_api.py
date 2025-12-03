from fastapi import APIRouter, HTTPException
from backend.app.services.allocate_service import (
    train_models_service,
    run_allocation_pipeline,
)

router = APIRouter()


@router.post("/train")
def admin_train():
    """
    Heavy step: trains the ML models using current students.csv + internships.csv.
    You do NOT need to call this often – only when data distribution changes a lot.
    """
    try:
        return train_models_service()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/allocate")
def admin_allocate():
    """
    Fast step: uses already-trained models to run allocation on current CSVs.
    Produces final allocations + fairness + boost reports + dashboard JSON.
    """
    try:
        return run_allocation_pipeline()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
