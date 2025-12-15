"""FastAPI prediction service."""

import logging
import os

import pandas as pd
from fastapi import FastAPI, HTTPException

from quant_core.inference import run_inference

from .config import get_model_path
from .schemas import HealthResponse, PredictRequest, PredictResponse
from .storage import load_model_from_gcs

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="quant_horizon",
    description="ML prediction service for liquidity provision range selection",
    version="0.1.0",
)

# Global state for loaded model
MODEL_BUNDLE = None
MODEL_PATH = None


@app.on_event("startup")
async def startup():
    """Load model on startup."""
    global MODEL_BUNDLE, MODEL_PATH

    try:
        MODEL_PATH = get_model_path()
        logger.info(f"Loading model from: {MODEL_PATH}")
        MODEL_BUNDLE = load_model_from_gcs(MODEL_PATH)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model on startup: {e}")
        # Don't raise - let service start but fail health checks
        # This allows debugging via logs


@app.get("/", response_model=dict)
def root():
    """Root endpoint."""
    return {
        "service": "quant_horizon",
        "version": "0.1.0",
        "description": "ML prediction service",
    }


@app.get("/health", response_model=HealthResponse)
def health():
    """Health check endpoint."""
    return HealthResponse(
        status="ok" if MODEL_BUNDLE is not None else "error",
        model_loaded=MODEL_BUNDLE is not None,
        model_path=MODEL_PATH,
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """Run ML prediction.

    Args:
        request: Prediction request with features and config

    Returns:
        Optimal configuration or error message
    """
    if MODEL_BUNDLE is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Check logs and MODEL_PATH environment variable.",
        )

    try:
        # Convert features dict to DataFrame
        features_df = pd.DataFrame([request.features])

        # Call quant_core inference
        result = run_inference(
            features=features_df,
            models=MODEL_BUNDLE["models"],
            metadata=MODEL_BUNDLE["metadata"],
            touch_tolerance=request.touch_tolerance,
            widths=request.widths,
            asymmetries=request.asymmetries,
        )

        # Return result (either success or error dict)
        return PredictResponse(**result)

    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
