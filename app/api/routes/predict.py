import logging
from starlette.requests import Request

from fastapi import APIRouter
from ...schemas import InputPayload, Predictions
from ...services import utils


routeur = APIRouter()

@routeur.post("/predict", response_model=Predictions, name="predict")
async def predict(request: Request, payload: InputPayload):
    logging.info("/predict endpoint call")
    
    model = request.app.state.model
    utils.check_model(model)
    
    X = utils.preprocess_predict_payload(payload)
    predictions = model.predict(X)
    result = utils.postprocess_prediction(predictions)
    
    return result