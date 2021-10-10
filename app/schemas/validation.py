from pydantic import BaseModel


class Validation(BaseModel):
    score_mean: float
    score_std: float
    folds: int
    metric: str
    model: str