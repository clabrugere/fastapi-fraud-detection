from fastapi import HTTPException, status
import numpy as np
import pandas as pd

from app.core.config import DATA_X_TEST_PATH, DATA_X_TRAIN_PATH, DATA_y_TEST_PATH, DATA_y_TRAIN_PATH
from app.schemas import InputPayload, TrainingResult, ValidationResult, PredictionResult, InputFile, InputTarget


def check_model(model):
    if model is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No model loaded on startup. Call /train before /predict.")


def process_predict_payload(input: InputPayload):
    
    if input is None:
        raise HTTPException(status_code=status.HTTP_406_NOT_ACCEPTABLE, detail=f"{input} is not a valid input")
    
    X = np.array(input)
    
    return X


def process_training_result(result):
    return TrainingResult(result)


def process_prediction_result(predictions):
    return PredictionResult(predictions)


def process_validation_result(scores, n_splits, metrics, model_name):
    return ValidationResult(
        score_mean=np.mean(scores, axis=1).tolist(),
        score_std=np.mean(scores, axis=1).tolist(),
        folds=n_splits,
        metrics=metrics,
        model=model_name
    )
    

def validate_input_files(X, y):
    try:
        InputFile.validate(X)
    except HTTPException(status_code=status.HTTP_406_NOT_ACCEPTABLE, detail="not a valid input") as err:
        raise err
    
    try:
        InputTarget.validate(y)
    except HTTPException(status_code=status.HTTP_406_NOT_ACCEPTABLE, detail="not a valid input") as err:
        raise err
    

def fetch_data(X_path, y_path):
    X, y = pd.read_csv(X_path), pd.read_csv(y_path)
    validate_input_files(X, y)
    
    return X, y


def fetch_training_data():
    return fetch_data(DATA_X_TRAIN_PATH, DATA_y_TRAIN_PATH)
    

def fetch_validation_data():
    return fetch_data(DATA_X_TEST_PATH, DATA_y_TEST_PATH)
    