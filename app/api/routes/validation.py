import logging
from typing import Optional

import numpy as np
from fastapi import APIRouter, Query
from starlette.requests import Request

from app.schemas import ValidationResult
from app.services.data import fetch_validation_data
from app.services.validation import validate

router = APIRouter()


@router.get("/validation", response_model=ValidationResult, name="model_validation")
async def evaluate(
        request: Request,
        n_splits: Optional[int] = Query(5, ge=1),
        proba_threshold: Optional[float] = Query(.5, gt=0., lt=1.)
) -> ValidationResult:
    logging.info("/validation endpoint call")

    model = request.app.state.model

    logging.info("fetch data...")

    X, y = fetch_validation_data()
    scores, n_splits, metrics, model_name = validate(model, X, y, n_splits=n_splits, proba_threshold=proba_threshold)

    result = ValidationResult(
        scores_mean=np.mean(scores, axis=0).tolist(),
        scores_std=np.std(scores, axis=0).tolist(),
        folds=n_splits,
        metrics=metrics,
        model=model_name
    )

    return result
