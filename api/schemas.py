from datetime import datetime
from uuid import uuid4

from pydantic import BaseModel, Field


class TransactionInput(BaseModel):
    amount: float
    user_id: str
    merchant_id: str
    timestamp: str


class FeatureImpact(BaseModel):
    feature: str
    impact: float


class PredictionOutput(BaseModel):
    transaction_id: str = Field(default_factory=lambda: str(uuid4()))
    fraud_probability: float
    is_fraud: bool
    risk_level: str
    explanation: dict[str, list[FeatureImpact]]
    threshold_used: float
    processed_at: datetime
