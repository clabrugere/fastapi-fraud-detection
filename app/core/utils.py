import numpy as np
import pandas as pd
from fastapi import HTTPException, status

from app.core.config import (DATA_X_TEST_PATH, DATA_X_TRAIN_PATH,
                             DATA_y_TEST_PATH, DATA_y_TRAIN_PATH)
from app.schemas import (InputFile, InputPayload, InputTarget)


def process_predict_payload(input: InputPayload):
    
    if input is None:
        raise HTTPException(status_code=status.HTTP_406_NOT_ACCEPTABLE, detail=f"{input} is not a valid input")
    
    X = np.array(input)
    
    return X


def validate_input_files(X, y):
    try:
        InputFile.validate(X)
    except ValueError as err:
        raise err
    
    try:
        InputTarget.validate(y)
    except ValueError as err:
        raise err


def fetch_data(X_path, y_path):
    X, y = pd.read_csv(X_path), pd.read_csv(y_path)
    validate_input_files(X, y)
    
    return X.values, y.values.ravel()


def fetch_training_data():
    return fetch_data(DATA_X_TRAIN_PATH, DATA_y_TRAIN_PATH)
    

def fetch_validation_data():
    return fetch_data(DATA_X_TEST_PATH, DATA_y_TEST_PATH)
    