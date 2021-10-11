import logging
from typing import Optional

from fastapi import APIRouter, Query
from starlette.requests import Request

from app.core.utils import process_predict_payload
from app.schemas import InputPayload, PredictionResult

router = APIRouter()


@router.post("/predict", response_model=PredictionResult, name="predict")
async def predict(
        request: Request,
        payload: InputPayload,
        proba_threshold: Optional[float] = Query(.5, gt=0., lt=1.),
        return_proba: Optional[bool] = False
) -> PredictionResult:
    logging.info("/predict endpoint call")

    model = request.app.state.model

    X = process_predict_payload(payload)
    logging.info(f"Inference on {X.shape[0]} samples")

    if return_proba:
        predictions = model.predict_proba(X)[:, 1]
    else:
        predictions = model.predict(X, proba_threshold)

    result = PredictionResult(
        predictions=predictions.tolist()
    )
    return result
