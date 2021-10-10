import logging
from starlette.requests import Request

from fastapi import APIRouter
from ...schemas import InputPayload, PredictionResult
from ...services import utils


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