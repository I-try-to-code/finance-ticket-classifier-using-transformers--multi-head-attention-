"""Pydantic schemas for request validation and response serialization."""

from typing import List
from pydantic import BaseModel, ConfigDict, Field, field_validator


class PredictRequest(BaseModel):
    """Input payload for customer query intent classification."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "My transfer is still pending"
            }
        }
    )

    text: str = Field(
        ...,
        description="The customer query text to classify.",
        min_length=1,
        max_length=1000,
    )

    @field_validator("text")
    @classmethod
    def validate_non_whitespace(cls, v: str) -> str:
        trimmed = v.strip()
        if not trimmed:
            raise ValueError("Query text cannot be empty or contain only whitespace.")
        return trimmed


class TopPrediction(BaseModel):
    """Candidate intent and its predicted probability."""

    intent: str = Field(..., description="The Banking77 intent category name.")
    probability: float = Field(..., description="The predicted probability score (0.0 to 1.0).", ge=0.0, le=1.0)


class PredictResponse(BaseModel):
    """Output payload for intent classification prediction."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "My transfer is still pending",
                "predicted_intent": "pending_transfer",
                "confidence": 0.9175,
                "top_predictions": [
                    {"intent": "pending_transfer", "probability": 0.9175},
                    {"intent": "balance_not_updated_after_bank_transfer", "probability": 0.0251},
                    {"intent": "transfer_timing", "probability": 0.0117}
                ],
                "latency_ms": 19.65
            }
        }
    )

    text: str = Field(..., description="The input customer query text.")
    predicted_intent: str = Field(..., description="The top predicted intent category.")
    confidence: float = Field(..., description="The confidence score of the top predicted intent.")
    top_predictions: List[TopPrediction] = Field(..., description="Top candidate predictions with probabilities.")
    latency_ms: float = Field(..., description="Inference latency in milliseconds.")


class HealthResponse(BaseModel):
    """Service health and hardware status payload."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "status": "healthy",
                "model_name": "distilbert-base-uncased (Banking77 V2)",
                "device": "cuda",
                "model_loaded": True
            }
        }
    )

    status: str = Field(..., description="Current status of the service.")
    model_name: str = Field(..., description="Name of the underlying Transformer model checkpoint.")
    device: str = Field(..., description="Hardware device hosting the model (e.g. cuda or cpu).")
    model_loaded: bool = Field(..., description="Whether the predictor model is successfully loaded in memory.")
