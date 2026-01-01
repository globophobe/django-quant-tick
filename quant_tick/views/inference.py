from typing import Any

from django.db.models import Prefetch
from django.http import Http404
from rest_framework.generics import ListAPIView
from rest_framework.request import Request
from rest_framework.response import Response

from quant_tick.models import CandleData, MLArtifact, Strategy


class InferenceView(ListAPIView):
    """Strategy-centric inference view."""

    queryset = Strategy.objects.filter(is_active=True)

    def get(self, request: Request, *args: Any, **kwargs: Any) -> Response:
        code_name = request.query_params.get("strategy")
        queryset = self.queryset.select_related("candle").prefetch_related(
            Prefetch(
                "artifacts",
                queryset=MLArtifact.objects.filter(is_active=True).order_by(
                    "-created_at"
                ),
            )
        )

        if code_name:
            queryset = queryset.filter(code_name=code_name)
            if not queryset.exists():
                raise Http404(f"Strategy '{code_name}' not found")

        results: list[dict] = []
        for strategy in queryset:
            candle_data = (
                CandleData.objects.filter(candle=strategy.candle)
                .order_by("-timestamp")
                .first()
            )
            if candle_data is None:
                continue
            signal = strategy.inference(candle_data)
            if signal is None:
                continue
            results.append(
                {
                    "strategy": strategy.code_name,
                    "signal_id": signal.id,
                    "timestamp": signal.timestamp,
                    "decision": signal.decision,
                    "probability": signal.probability,
                }
            )

        return Response({"ok": True, "count": len(results), "results": results})
