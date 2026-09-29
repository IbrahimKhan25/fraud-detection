"""Request/response shapes for the scoring endpoint.

The core fields are the ones the feature store needs to build the entity and
its rolling features. Everything else the model was trained on (the other ~420
raw IEEE-CIS columns: C1-C14, D1-D15, V1-V339, M1-M9, id_*, ...) goes in
`features`, keyed by the original column name. Missing keys are scored as
missing (NaN), which is what LightGBM saw for missing values in training.
"""
from typing import Dict, Optional, Union

from pydantic import BaseModel, Field


class Transaction(BaseModel):
    TransactionID: int
    TransactionDT: float = Field(description="Seconds offset, same units as IEEE-CIS TransactionDT")
    TransactionAmt: float
    # card1 is int64 in the raw data and the rest are float64. That matters:
    # entity IDs are built by string-concatenating these, so 13926 and
    # 13926.0 would produce different entities. See entity_id_for() in app.py.
    card1: Optional[int] = None
    card2: Optional[float] = None
    card3: Optional[float] = None
    card5: Optional[float] = None
    addr1: Optional[float] = None
    addr2: Optional[float] = None
    DeviceInfo: Optional[str] = None
    features: Dict[str, Optional[Union[float, str]]] = Field(default_factory=dict)


class ScoreResponse(BaseModel):
    transaction_id: int
    fraud_score: float
    entity_id: str
    engineered_features: Dict[str, Optional[float]]
    latency_ms: float
