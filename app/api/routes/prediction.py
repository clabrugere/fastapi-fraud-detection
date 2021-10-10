import logging

from app.schemas import InputPayload, PredictionResult
from app.services import utils
from fastapi import APIRouter
from starlette.requests import Request

routeur = APIRouter()

@routeur.post("/predict", response_model=PredictionResult, name="predict")
async def predict(request: Request, payload: InputPayload):
    logging.info("/predict endpoint call")
    
    model = request.app.state.model
    utils.check_model(model)
    
    X = utils.process_predict_payload(payload)
    predictions = model.predict(X)
    result = utils.process_prediction_result(predictions)
    
    return result
