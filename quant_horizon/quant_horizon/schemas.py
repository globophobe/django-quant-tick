"""Pydantic models for request/response validation."""

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Prediction request schema."""

    features: dict = Field(
        ...,
        description="Feature dictionary (e.g., {'close': 100.0, 'volume': 1000, ...})",
    )
    touch_tolerance: float = Field(
        ..., description="Max acceptable P(touch) for valid config", ge=0.0, le=1.0
    )
    widths: list[float] | None = Field(
        None, description="Width grid to search (defaults to DEFAULT_WIDTHS)"
    )
    asymmetries: list[float] | None = Field(
        None, description="Asymmetry grid to search (defaults to DEFAULT_ASYMMETRIES)"
    )


class PredictResponse(BaseModel):
    """Prediction response schema."""

    lower_bound: float | None = None
    upper_bound: float | None = None
    borrow_ratio: float | None = None
    p_touch_lower: float | None = None
    p_touch_upper: float | None = None
    width: float | None = None
    asymmetry: float | None = None
    error: str | None = None
    message: str | None = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    model_path: str | None = None
