from typing import List

from pydantic import BaseModel


class TrainingResult(BaseModel):
    best_score: float


class ValidationResult(BaseModel):
    scores_mean: List[float]
    scores_std: List[float]
    folds: int
    metrics: List[str]
    model: str


class PredictionResult(BaseModel):
    predictions: List[float]
