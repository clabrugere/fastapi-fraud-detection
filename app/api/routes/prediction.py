import logging

from fastapi import APIRouter
from starlette.requests import Request

from app.core import utils
from app.schemas import InputPayload, PredictionResult

router = APIRouter()


@router.post("/predict", response_model=PredictionResult, name="predict")
async def predict(request: Request, payload: InputPayload):
    logging.info("/predict endpoint call")

    model = request.app.state.model

    X = utils.process_predict_payload(payload)
    predictions = model.predict(X)

    return predictions
