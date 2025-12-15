# quant_horizon

FastAPI prediction service for ML-based liquidity provision range selection.

## Overview

This service provides a lightweight HTTP API for running ML predictions. It:
- Loads trained models from Google Cloud Storage on startup
- Exposes a POST /predict endpoint for inference
- Uses the shared `quant_core` package for all ML logic

## Architecture

```
quant_horizon (FastAPI service)
    ↓ imports
quant_core (pure ML logic)
```

## Deployment

Deployed to Google Cloud Run as a containerized service.

## API

### POST /predict

Request:
```json
{
  "features": {"close": 100.0, "volume": 1000, ...},
  "touch_tolerance": 0.15,
  "widths": [0.02, 0.03, 0.04],
  "asymmetries": [-0.2, 0.0, 0.2]
}
```

Response:
```json
{
  "lower_bound": -0.03,
  "upper_bound": 0.05,
  "borrow_ratio": 0.5,
  "p_touch_lower": 0.12,
  "p_touch_upper": 0.14,
  "width": 0.04,
  "asymmetry": 0.0
}
```

### GET /health

Returns service health status.

## Environment Variables

- `MODEL_PATH` - GCS path to model bundle (gs://bucket/path/model.joblib)
- `PORT` - HTTP port (default: 8080)
