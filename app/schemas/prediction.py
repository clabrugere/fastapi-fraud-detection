from typing import List
from pydantic import BaseModel


class PredictionResult(BaseModel):
    predictions: List[float]