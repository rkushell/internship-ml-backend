from fastapi import APIRouter, HTTPException
from backend.app.models import PredictRequest, PredictResponse
from backend.app.services.predict_service import predict_single_pair

router = APIRouter()

@router.post("/", response_model=PredictResponse)
def predict(req: PredictRequest):
    try:
        return predict_single_pair(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
