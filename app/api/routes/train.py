import logging

from fastapi import APIRouter
from starlette.requests import Request

from app.core import utils
from app.core.config import DATA_MODEL_PATH
from app.schemas import TrainingResult

router = APIRouter()


@router.get("/train", response_model=TrainingResult, name="model_training")
async def train(request: Request):
    logging.info("/train endpoint call")

    logging.info("fetch data...")
    X, y = utils.fetch_training_data()

    model = request.app.state.model

    logging.info("training model...")
    model.fit(X, y)

    logging.info(f"best_score: {model.best_score}")
    logging.info(f"save trained model {DATA_MODEL_PATH}")

    model.save(DATA_MODEL_PATH)
    request.app.state.model = model

    return {"best_score": f"{model.best_score}"}
