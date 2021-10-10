from typing import List
from pydantic import BaseModel


class Sample(BaseModel):
    feature_1: float
    feature_2: str
    

class InputPayload(BaseModel):
    samples: List[Sample]

    
class TargetPayload(BaseModel):
    target: List[float]
