from typing import List

from pydantic import BaseModel


class TrainingResult(BaseModel):
    message: str


class ValidationResult(BaseModel):
    score_mean: List[float]
    score_std: List[float]
    folds: int
    metrics: List[str]
    model: str


class PredictionResult(BaseModel):
    predictions: List[float]
