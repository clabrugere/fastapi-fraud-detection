import logging
from typing import Optional

from fastapi import APIRouter, Query
from starlette.requests import Request

from app.core.config import DATA_MODEL_PATH
from app.schemas import TrainingResult
from app.services.data import fetch_training_data

router = APIRouter()


@router.get("/train", response_model=TrainingResult, name="model_training")
async def train(
        request: Request,
        n_splits: Optional[int] = Query(3, ge=1),
        n_iter: Optional[int] = Query(10, ge=1)
) -> TrainingResult:
    logging.info("/train endpoint call")

    logging.info("fetch data...")
    X, y = fetch_training_data()

    model = request.app.state.model

    logging.info("training model...")
    model.fit(X, y, n_splits=n_splits, n_iter=n_iter)

    logging.info(f"best_score: {model.best_score}")
    logging.info(f"save trained model {DATA_MODEL_PATH}")

    model.save(DATA_MODEL_PATH)
    request.app.state.model = model

    result = TrainingResult(
        best_score=model.best_score
    )
    return result
