"""GCS model loading utilities."""

import logging
from io import BytesIO

import joblib
from google.cloud import storage

logger = logging.getLogger(__name__)


def load_model_from_gcs(gcs_path: str) -> dict:
    """Load model bundle from GCS.

    Args:
        gcs_path: GCS path (gs://bucket/path/model.joblib)

    Returns:
        Model bundle dict with keys:
        {
            "models": {"lower": model, "upper": model},
            "metadata": {
                "feature_cols": [...],
                "max_horizon": 180,
                "decision_horizons": [60, 120, 180],
                ...
            }
        }

    Raises:
        ValueError: If path format is invalid
        Exception: If download or deserialization fails
    """
    if not gcs_path.startswith("gs://"):
        raise ValueError(f"Invalid GCS path: {gcs_path}. Must start with gs://")

    # Parse gs://bucket/path/to/file.joblib
    path_parts = gcs_path[5:].split("/", 1)  # Remove 'gs://' and split
    if len(path_parts) != 2:
        raise ValueError(f"Invalid GCS path format: {gcs_path}")

    bucket_name, blob_path = path_parts

    logger.info(f"Loading model from GCS: bucket={bucket_name}, path={blob_path}")

    try:
        # Download from GCS
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_path)

        # Download to BytesIO
        buffer = BytesIO()
        blob.download_to_file(buffer)
        buffer.seek(0)

        # Deserialize
        model_bundle = joblib.load(buffer)

        # Validate structure
        if "models" not in model_bundle or "metadata" not in model_bundle:
            raise ValueError(
                "Invalid model bundle format. Expected keys: 'models', 'metadata'"
            )

        if "lower" not in model_bundle["models"] or "upper" not in model_bundle["models"]:
            raise ValueError(
                "Invalid models dict. Expected keys: 'lower', 'upper'"
            )

        logger.info(
            f"Model loaded successfully. Feature count: {len(model_bundle['metadata'].get('feature_cols', []))}"
        )

        return model_bundle

    except Exception as e:
        logger.error(f"Failed to load model from GCS: {e}")
        raise
