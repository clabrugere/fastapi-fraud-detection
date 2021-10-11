from typing import Tuple

import numpy as np
import pandas as pd

from app.core.config import (DATA_X_VAL_PATH, DATA_X_TRAIN_PATH,
                             DATA_y_VAL_PATH, DATA_y_TRAIN_PATH)
from app.core.utils import validate_input_files


def fetch_data(X_path, y_path) -> Tuple[np.ndarray, np.ndarray]:
    X, y = pd.read_csv(X_path), pd.read_csv(y_path)
    validate_input_files(X, y)

    return X.values, y.values.ravel()


def fetch_training_data() -> Tuple[np.ndarray, np.ndarray]:
    return fetch_data(DATA_X_TRAIN_PATH, DATA_y_TRAIN_PATH)


def fetch_validation_data() -> Tuple[np.ndarray, np.ndarray]:
    return fetch_data(DATA_X_VAL_PATH, DATA_y_VAL_PATH)
