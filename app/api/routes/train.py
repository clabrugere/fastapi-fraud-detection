import logging
from starlette.requests import Request

from fastapi import APIRouter
from ...schemas import InputPayload, TargetPayload, Validation
from ...services import utils


routeur = APIRouter()

@routeur.post("/train", response_model=Validation, name="model_training")
async def train(request: Request, input: InputPayload, target: TargetPayload):
    logging.info("/train endpoint call")
    
    model = request.app.state.model
    utils.check_model(model)
    
    X, y = utils.preprocess_validation_payload(input, target)
    scores, n_splits, metric, model_name = model.validat(X, y)
    result = utils.postprocess_validation(scores, n_splits, metric, model_name)
    
    return result
