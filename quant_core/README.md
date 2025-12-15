# quant_core

Pure Python ML prediction logic shared between quant_tick and quant_horizon.

## Overview

This package contains the core ML functionality extracted from quant_tick:
- Hazard-based survival analysis prediction
- Feature engineering
- Optimal configuration selection
- Calibration and probability adjustment

## Dependencies

- lightgbm - Gradient boosting models
- scikit-learn - Calibration (isotonic, Platt)
- pandas - Data structures
- numpy - Numerical operations

**No Django or FastAPI dependencies** - pure ML code only.

## Usage

```python
from quant_core.inference import run_inference

# Load models and metadata
models = {"lower": lower_model, "upper": upper_model}
metadata = {"feature_cols": [...], "max_horizon": 180, ...}

# Run inference
result = run_inference(
    features=features_df,
    models=models,
    metadata=metadata,
    touch_tolerance=0.15
)
```

## Architecture

- `prediction.py` - Core prediction functions (hazard → probabilities)
- `features.py` - Feature engineering (volatility, momentum, cross-exchange)
- `constants.py` - Default parameters (widths, asymmetries)
- `inference.py` - High-level API
