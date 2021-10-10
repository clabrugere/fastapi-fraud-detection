import logging
from starlette.requests import Request

from fastapi import APIRouter
from app.schemas import ValidationResult
from app.services import utils


routeur = APIRouter()

@routeur.get("/validation", response_model=ValidationResult, name="model_validation")
async def evaluate(request: Request):
    logging.info("/validation endpoint call")
    
    model = request.app.state.model
    utils.check_model(model)
    
    X, y = utils.fetch_validation_data()
    scores, n_splits, metric, model_name = model.validat(X, y)
    result = utils.process_validation_result(scores, n_splits, metric, model_name)
    
    return result
