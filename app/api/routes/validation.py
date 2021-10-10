import logging

import numpy as np
from fastapi import APIRouter
from starlette.requests import Request

from app.core import utils
from app.schemas import ValidationResult

router = APIRouter()


@router.get("/validation", response_model=ValidationResult, name="model_validation")
async def evaluate(request: Request):
    logging.info("/validation endpoint call")

    model = request.app.state.model

    logging.info("fetch data...")

    X, y = utils.fetch_validation_data()
    scores, n_splits, metrics, model_name = model.validate(X, y)
    result = {
        "scores_mean": np.mean(scores, axis=0).tolist(),
        "scores_std": np.std(scores, axis=0).tolist(),
        "folds": n_splits,
        "metrics": metrics,
        "model": model_name
    }
    return result
