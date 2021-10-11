import numpy as np
from fastapi import HTTPException, status
from fastapi.encoders import jsonable_encoder

from app.schemas import (InputFile, InputPayload, InputTarget)


def process_predict_payload(input: InputPayload) -> np.ndarray:
    if input is None:
        raise HTTPException(status_code=status.HTTP_406_NOT_ACCEPTABLE, detail=f"{input} is not a valid input")

    input_encoded = jsonable_encoder(input)['samples']
    X = np.asarray([list(s.values()) for s in input_encoded])

    return X


def validate_input_files(X: np.ndarray, y: np.ndarray) -> None:
    try:
        InputFile.validate(X)
    except ValueError as err:
        raise err

    try:
        InputTarget.validate(y)
    except ValueError as err:
        raise err
