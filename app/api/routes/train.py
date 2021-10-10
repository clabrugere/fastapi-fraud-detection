import logging
from starlette.requests import Request
from fastapi import APIRouter
from app.schemas import TrainingResult
from app.core.config import DATA_MODEL_PATH, DATA_X_TRAIN_PATH, DATA_y_TRAIN_PATH
from app.services import utils


routeur = APIRouter()

@routeur.get("/train", response_model=TrainingResult, name="model_training")
async def train(request: Request):
    logging.info("/train endpoint call")
    
    X, y = utils.fetch_training_data()
    model = request.app.state.model
    model.fit(X, y)
    model.save(DATA_MODEL_PATH)
    request.app.state.model = model
    
    result = utils.process_training_result("model trained successfully")
    
    return result
