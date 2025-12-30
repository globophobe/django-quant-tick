import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from django.conf import settings
from django.http import Http404
from django.utils import timezone
from rest_framework.generics import ListAPIView
from rest_framework.request import Request
from rest_framework.response import Response

from quant_tick.lib.meta import build_event_dataset
from quant_tick.models import MetaArtifact, MetaModel, MetaSignal

logger = logging.getLogger(__name__)


def _load_bundle(artifact: MetaArtifact) -> dict:
    """Load joblib bundle from FileField."""
    file_path = artifact.file.path
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Bundle not found: {file_path}")
    return joblib.load(file_path)


class MetaInferenceView(ListAPIView):
    """Local meta-model inference (no quant_horizon)."""

    queryset = MetaModel.objects.filter(is_active=True)

    def get(self, request: Request, *args: Any, **kwargs: Any) -> Response:
        code_name = request.query_params.get("model")
        if not code_name:
            return Response(
                {
                    "ok": False,
                    "error": "missing_params",
                    "message": "Provide ?model=<meta_model_code>",
                },
                status=400,
            )

        try:
            meta_model = self.queryset.get(code_name=code_name)
        except MetaModel.DoesNotExist:
            raise Http404(f"MetaModel '{code_name}' not found")

        candle = meta_model.candle
        if candle is None:
            return Response(
                {"ok": False, "error": "no_candle", "message": "MetaModel has no candle"},
                status=400,
            )

        artifact = (
            MetaArtifact.objects.filter(meta_model=meta_model, is_active=True)
            .order_by("-created_at")
            .first()
        )
        if artifact is None:
            return Response(
                {"ok": False, "error": "no_artifact", "message": "No active artifact"},
                status=404,
            )

        bundle = _load_bundle(artifact)
        model = bundle["model"]
        metadata = bundle.get("metadata", {})
        feature_cols = metadata.get("feature_cols", [])
        threshold = metadata.get("threshold", 0.5)

        now = timezone.now()
        lookback_hours = int(request.query_params.get("lookback_hours", 24))
        ts_from = now - timedelta(hours=lookback_hours)

        events = build_event_dataset(
            candle,
            timestamp_from=ts_from,
            timestamp_to=now,
            meta_model=meta_model,
            include_incomplete=False,
        )
        if events.empty:
            return Response(
                {"ok": False, "error": "no_events", "message": "No events in window"},
                status=404,
            )

        latest = events.iloc[-1]
        feature_dict = {
            col: latest.get(col) for col in feature_cols if col in latest.index
        }
        df_feat = pd.DataFrame([feature_dict])
        proba = float(model.predict_proba(df_feat)[0, 1])  # positive class
        decision = "take" if proba >= threshold else "skip"

        signal = MetaSignal.objects.create(
            meta_model=meta_model,
            timestamp=latest["timestamp_entry"],
            probability=proba,
            decision=decision,
            json_data={
                "threshold": threshold,
                "feature_cols": feature_cols,
                "features": feature_dict,
                "bundle_metadata": metadata,
            },
        )

        return Response(
            {
                "ok": True,
                "meta_model": meta_model.code_name,
                "candle": candle.code_name,
                "signal_id": signal.id,
                "probability": proba,
                "decision": decision,
                "threshold": threshold,
            }
        )
