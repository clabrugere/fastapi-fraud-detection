from typing import List
from pydantic import BaseModel


class Prediction(BaseModel):
    predictions: List[float]