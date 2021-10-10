import logging
from fastapi import HTTPException, status
import numpy as np
from ..schemas import InputPayload, TargetPayload, Prediction, Validation


def check_model(model):
    if model is None:
        logging.error("No model loaded on startup. Call /train before /predict.")
        raise HTTPException(status_code=404, detail="No model loaded on startup. Call /train before /predict.")


def preprocess_predict_payload(input: InputPayload):
    
    if input is None:
            raise ValueError(f"{input} is not a valid input")
    
    X = np.array(input)
    
    return X


def preprocess_validation_payload(input: InputPayload, target: TargetPayload = None):
    
    if input is None:
            raise ValueError(f"{input} is not a valid input")

    if target is None:
            raise ValueError(f"{target} is not a valid target")
    
    X, y = np.array(input), np.array(target)
    
    return X, y


def postprocess_prediction(predictions):
    return Prediction(predictions)


def postprocess_validation(scores, n_splits, metric, model_name):
    return Validation(
        score_mean=np.mean(scores),
        score_std=np.mean(scores),
        folds=n_splits,
        metric=metric,
        model=model_name
    )