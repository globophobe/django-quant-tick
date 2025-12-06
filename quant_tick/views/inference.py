import logging
import pickle
from io import BytesIO
from typing import Any

import joblib
import pandas as pd
from django.utils import timezone
from rest_framework.generics import ListAPIView
from rest_framework.request import Request
from rest_framework.response import Response

from quant_tick.lib.labels import _compute_features
from quant_tick.lib.ml import (
    DEFAULT_ASYMMETRIES,
    DEFAULT_WIDTHS,
    compute_bound_features,
    find_optimal_config,
    hazard_to_per_horizon_probs,
    prepare_features,
)
from quant_tick.lib.schema import MLSchema
from quant_tick.models import CandleData, MLArtifact, MLConfig, MLSignal

logger = logging.getLogger(__name__)


class InferenceView(ListAPIView):
    """Inference API."""

    queryset = MLConfig.objects.filter(is_active=True)

    def get(self, request: Request, *args: Any, **kwargs: Any) -> Response:
        """Run inference for each active MLConfig."""
        results = []

        for cfg in self.get_queryset():
            result = self._run_inference(cfg)
            results.append(result)

        return Response({"ok": True, "results": results})

    def _run_inference(self, cfg: MLConfig) -> dict:
        """Run inference for a single MLConfig."""
        # Get decision horizons from config
        decision_horizons = cfg.json_data.get("decision_horizons", [60, 120, 180])

        # Load survival models
        try:
            lower_artifact = cfg.ml_artifacts.get(model_type="hazard_lower")
            upper_artifact = cfg.ml_artifacts.get(model_type="hazard_upper")
        except MLArtifact.DoesNotExist:
            logger.error(f"{cfg}: No survival models found. Train models first.")
            return {"config": cfg.code_name, "error": "missing survival models"}

        # Load models from artifacts
        lower_model = self._load_model(lower_artifact)
        upper_model = self._load_model(upper_artifact)

        if lower_model is None or upper_model is None:
            return {"config": cfg.code_name, "error": "failed to load models"}

        # Get feature columns and max_horizon
        feature_cols = lower_artifact.feature_columns
        max_horizon = cfg.json_data.get("max_horizon", cfg.horizon_bars)

        logger.info(
            f"{cfg}: Loaded survival models: max_horizon={max_horizon}, "
            f"decision_horizons={decision_horizons}"
        )

        # Validate feature hash for drift detection
        from quant_tick.lib.train import compute_feature_hash
        current_hash = compute_feature_hash(feature_cols)
        stored_hash = cfg.json_data.get("feature_hash")

        if stored_hash and current_hash != stored_hash:
            logger.warning(
                f"{cfg}: Feature schema drift detected! "
                f"Current hash={current_hash}, stored hash={stored_hash}. "
                f"Models may produce incorrect predictions."
            )

        # Get recent candle data
        features_df = self._get_latest_features(cfg)
        if features_df is None or len(features_df) == 0:
            logger.warning(f"{cfg}: no feature data available")
            return {"config": cfg.code_name, "error": "no data"}

        # Check for missing features
        # Get decision horizons for schema filtering
        decision_horizons = cfg.json_data.get("decision_horizons", [60, 120, 180])

        # Use centralized schema to get data features (excludes config cols added during prediction)
        data_feature_cols = MLSchema.get_data_features(feature_cols, decision_horizons)
        available_cols = set(features_df.columns)
        missing_cols = [c for c in data_feature_cols if c not in available_cols]
        if missing_cols:
            raise ValueError(
                f"{cfg}: Cannot run inference - {len(missing_cols)} required feature(s) missing: "
                f"{', '.join(missing_cols[:10])}{'...' if len(missing_cols) > 10 else ''}. "
                f"Model expects features: {data_feature_cols}. "
                f"Ensure all features are available in candle data."
            )

        # Get the latest bar's features
        latest = features_df.iloc[[-1]].copy()

        # Prediction workflow for range touch probabilities:
        #
        # 1. Load trained models (one per horizon+side, e.g., lower_h60, upper_h180)
        # 2. For each horizon H:
        #    - Get raw model prediction: P_raw(hit_by_H)
        #    - Apply calibrator: P_cal = calibrator.transform(P_raw)
        #    - Store in horizon_probs dict
        # 3. Enforce monotonicity: P_60 ≤ P_120 ≤ P_180 (cumulative max)
        # 4. Return monotone probabilities
        #
        # These probabilities represent: "What % chance does this bound get touched
        # within the next H bars, given current features?"
        #
        # NOT predicted: direction, magnitude, timing within horizon, fee earnings

        # Create prediction functions using survival models
        def predict_lower(features: pd.DataFrame, lower_pct: float, upper_pct: float) -> float:
            """Predict max P(hit_lower) across decision horizons using survival model."""
            # Add bound features
            feat_with_bounds = compute_bound_features(features, lower_pct, upper_pct)

            # Exclude k from prepare_features (not in live candle data)
            base_feature_cols = [c for c in feature_cols if c != "k"]
            X_array, expanded_cols = prepare_features(feat_with_bounds, base_feature_cols)

            # Rebuild DataFrame (no k yet)
            X_df = pd.DataFrame(X_array, columns=expanded_cols)
            X_base = X_df.iloc[[0]]

            # Reconstruct per-horizon probabilities from hazard model
            # Pass full feature_cols (includes k) - hazard_to_per_horizon_probs will add it
            horizon_probs = hazard_to_per_horizon_probs(
                lower_model,
                X_base,
                feature_cols,  # Full training order WITH k
                decision_horizons,
                max_horizon,
            )

            # Return max risk (conservative)
            return float(max(horizon_probs.values()))

        def predict_upper(features: pd.DataFrame, lower_pct: float, upper_pct: float) -> float:
            """Predict max P(hit_upper) across decision horizons using survival model."""
            feat_with_bounds = compute_bound_features(features, lower_pct, upper_pct)
            base_feature_cols = [c for c in feature_cols if c != "k"]
            X_array, expanded_cols = prepare_features(feat_with_bounds, base_feature_cols)
            X_df = pd.DataFrame(X_array, columns=expanded_cols)
            X_base = X_df.iloc[[0]]

            horizon_probs = hazard_to_per_horizon_probs(
                upper_model,
                X_base,
                feature_cols,  # Full training order WITH k
                decision_horizons,
                max_horizon,
            )

            return float(max(horizon_probs.values()))

        # Find optimal config
        widths = cfg.json_data.get("widths", DEFAULT_WIDTHS)
        asymmetries = cfg.json_data.get("asymmetries", DEFAULT_ASYMMETRIES)

        optimal = find_optimal_config(
            predict_lower,
            predict_upper,
            latest,
            touch_tolerance=cfg.touch_tolerance,
            widths=widths,
            asymmetries=asymmetries,
        )

        if optimal is None:
            logger.info(f"{cfg}: no valid config found")
            return {"config": cfg.code_name, "error": "no valid config"}

        timestamp = latest["timestamp"].iloc[0] if "timestamp" in latest.columns else timezone.now()

        # Skip if already processed.
        if cfg.last_processed_timestamp and timestamp <= cfg.last_processed_timestamp:
            logger.info(f"{cfg}: skipping already processed bar at {timestamp}")
            return {"config": cfg.code_name, "skipped": True, "timestamp": str(timestamp)}

        # Create MLSignal
        signal = MLSignal.objects.create(
            ml_config=cfg,
            timestamp=timestamp,
            lower_bound=optimal.lower_pct,
            upper_bound=optimal.upper_pct,
            borrow_ratio=optimal.borrow_ratio,
            p_touch_lower=optimal.p_touch_lower,
            p_touch_upper=optimal.p_touch_upper,
            json_data={
                "width": optimal.width,
                "asymmetry": optimal.asymmetry,
            },
        )

        cfg.last_processed_timestamp = timestamp
        cfg.save(update_fields=["last_processed_timestamp"])

        logger.info(
            f"{cfg}: signal created - bounds=[{optimal.lower_pct:.3f}, {optimal.upper_pct:.3f}], "
            f"borrow_ratio={optimal.borrow_ratio:.2f}"
        )

        return {
            "config": cfg.code_name,
            "signal_id": signal.id,
            "lower_bound": optimal.lower_pct,
            "upper_bound": optimal.upper_pct,
            "borrow_ratio": optimal.borrow_ratio,
            "p_touch_lower": optimal.p_touch_lower,
            "p_touch_upper": optimal.p_touch_upper,
        }

    def _load_model(self, artifact: MLArtifact) -> Any:
        """Load model."""
        try:
            artifact.artifact.seek(0)
            buf = BytesIO(artifact.artifact.read())
            return joblib.load(buf)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None

    def _get_latest_features(self, config: MLConfig) -> pd.DataFrame | None:
        """Get latest features."""
        candle_data = (
            CandleData.objects.filter(candle=config.candle)
            .order_by("-timestamp")[:config.inference_lookback]
        )

        if not candle_data:
            return None

        df = pd.DataFrame([
            {"timestamp": cd.timestamp, **cd.json_data}
            for cd in reversed(candle_data)
        ])

        if df.empty:
            return None

        # Use config.symbol.exchange as canonical for multi-exchange candles
        df = _compute_features(df, canonical_exchange=config.symbol.exchange)
        return df.dropna(subset=["close"]) if "close" in df.columns else df
