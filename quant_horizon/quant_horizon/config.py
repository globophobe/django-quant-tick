"""Configuration settings."""

import os


def get_model_path() -> str:
    """Get model path from environment variable.

    Returns:
        GCS path to model bundle

    Raises:
        ValueError: If MODEL_PATH not set
    """
    model_path = os.getenv("MODEL_PATH")
    if not model_path:
        raise ValueError(
            "MODEL_PATH environment variable not set. "
            "Example: gs://bucket/models/model.joblib"
        )
    return model_path


def get_port() -> int:
    """Get HTTP port from environment variable."""
    return int(os.getenv("PORT", "8080"))
