import logging
import os
from typing import Any

import httpx
import pandas as pd
from django.utils import timezone
from quant_core.constants import DEFAULT_ASYMMETRIES, DEFAULT_WIDTHS
from quant_core.features import _compute_features, compute_max_warmup_bars
from rest_framework.generics import ListAPIView
from rest_framework.request import Request
from rest_framework.response import Response

from quant_tick.models import CandleData, MLConfig, MLSignal

logger = logging.getLogger(__name__)

QUANT_HORIZON_URL = os.getenv("QUANT_HORIZON_URL", "http://localhost:8000")


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
        """Run inference for a single MLConfig via quant_horizon service."""
        # Get recent candle data and compute features
        features_df = self._get_latest_features(cfg)
        if features_df is None or len(features_df) == 0:
            logger.warning(f"{cfg}: no feature data available")
            return {"config": cfg.code_name, "error": "no data"}

        # Get the latest bar's features
        latest = features_df.iloc[-1]

        # Call quant_horizon service
        request_data = {
            "features": latest.to_dict(),
            "touch_tolerance": cfg.touch_tolerance,
            "widths": cfg.json_data.get("widths", DEFAULT_WIDTHS),
            "asymmetries": cfg.json_data.get("asymmetries", DEFAULT_ASYMMETRIES),
        }

        try:
            response = httpx.post(
                f"{QUANT_HORIZON_URL}/predict",
                json=request_data,
                timeout=30.0,
            )
            response.raise_for_status()
            result = response.json()
        except httpx.HTTPError as e:
            logger.error(f"{cfg}: quant_horizon request failed: {e}")
            return {"config": cfg.code_name, "error": "service_unavailable"}

        if "error" in result:
            logger.info(f"{cfg}: {result.get('error')}: {result.get('message', '')}")
            return {"config": cfg.code_name, "error": result["error"]}

        # Extract timestamp
        timestamp = latest.get("timestamp", timezone.now())
        if isinstance(timestamp, pd.Timestamp):
            timestamp = timestamp.to_pydatetime()

        # Skip if already processed
        if cfg.last_processed_timestamp and timestamp <= cfg.last_processed_timestamp:
            logger.info(f"{cfg}: skipping already processed bar at {timestamp}")
            return {
                "config": cfg.code_name,
                "skipped": True,
                "timestamp": str(timestamp),
            }

        # Extract features for audit trail
        metadata_cols = {"timestamp", "close", "open", "high", "low", "volume"}
        feature_dict = {
            k: float(v) if isinstance(v, (int, float)) else v
            for k, v in latest.to_dict().items()
            if k not in metadata_cols and pd.notna(v)
        }

        # Create MLSignal with features for audit trail
        signal = MLSignal.objects.create(
            ml_config=cfg,
            timestamp=timestamp,
            lower_bound=result["lower_bound"],
            upper_bound=result["upper_bound"],
            borrow_ratio=result["borrow_ratio"],
            p_touch_lower=result["p_touch_lower"],
            p_touch_upper=result["p_touch_upper"],
            json_data={
                "width": result["width"],
                "asymmetry": result["asymmetry"],
                "features": feature_dict,
            },
        )

        cfg.last_processed_timestamp = timestamp
        cfg.save(update_fields=["last_processed_timestamp"])

        logger.info(
            f"{cfg}: signal created - bounds=[{result['lower_bound']:.3f}, {result['upper_bound']:.3f}], "
            f"borrow_ratio={result['borrow_ratio']:.2f}"
        )

        return {
            "config": cfg.code_name,
            "signal_id": signal.id,
            "lower_bound": result["lower_bound"],
            "upper_bound": result["upper_bound"],
            "borrow_ratio": result["borrow_ratio"],
            "p_touch_lower": result["p_touch_lower"],
            "p_touch_upper": result["p_touch_upper"],
        }

    def _get_latest_features(self, cfg: MLConfig) -> pd.DataFrame | None:
        """Get latest features."""
        # Validate sufficient lookback
        max_warmup = compute_max_warmup_bars()
        if cfg.inference_lookback < max_warmup:
            logger.error(
                f"{cfg}: inference_lookback ({cfg.inference_lookback}) < max_warmup_bars ({max_warmup})"
            )
            return None

        candle_data = CandleData.objects.filter(candle=cfg.candle).order_by(
            "-timestamp"
        )[: cfg.inference_lookback]

        if not candle_data.exists():
            return None

        df = pd.DataFrame(
            [
                {"timestamp": cd.timestamp, **cd.json_data}
                for cd in reversed(candle_data)
            ]
        )

        df = _compute_features(df, canonical_exchange=cfg.symbol.exchange)
        return df.dropna(subset=["close"]) if "close" in df.columns else df
