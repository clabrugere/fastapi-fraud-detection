import logging
from starlette.requests import Request

from fastapi import APIRouter
from ...schemas import InputPayload, TargetPayload, ValidationResult
from ...services import utils


routeur = APIRouter()

@routeur.post("/validation", response_model=ValidationResult, name="model_validation")
async def evaluate(request: Request, input: InputPayload, target: TargetPayload):
    logging.info("/validation endpoint call")
    
    model = request.app.state.model
    utils.check_model(model)
    
    X, y = utils.process_validation_payload(input, target)
    scores, n_splits, metric, model_name = model.validat(X, y)
    result = utils.process_validation_result(scores, n_splits, metric, model_name)
    
    return result
