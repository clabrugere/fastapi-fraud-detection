from pydantic import BaseModel


class ValidationResult(BaseModel):
    score_mean: float
    score_std: float
    folds: int
    metric: str
    model: str