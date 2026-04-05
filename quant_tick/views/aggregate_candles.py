import logging

from django.db.models import QuerySet
from django.http import HttpRequest, JsonResponse
from django.views import View

from quant_tick.models import Candle, TaskState
from quant_tick.storage import convert_candle_cache_to_daily
from quant_tick.views.aggregate_trades import get_request_params

logger = logging.getLogger(__name__)


class AggregateCandleView(View):
    """Aggregate candle view."""

    queryset = Candle.objects.filter(is_active=True).select_related("symbol")

    def get_task_state(self) -> TaskState:
        """Get the global task state for candle aggregation."""
        task_state, _ = TaskState.objects.get_or_create(
            name="aggregate_candles",
            exchange="",
        )
        return task_state

    def get_queryset(self) -> QuerySet:
        """Get queryset."""
        queryset = self.queryset
        code_name = self.request.GET.get("code_name")
        if code_name:
            queryset = queryset.filter(code_name=code_name)
        return queryset

    def get_params(self, request: HttpRequest) -> list[tuple]:
        """Get params."""
        timestamp_from, timestamp_to, retry = get_request_params(request)
        queryset = self.get_queryset()
        return [
            (
                candle,
                timestamp_from,
                timestamp_to,
                retry,
            )
            for candle in queryset
        ]

    def get(self, request: HttpRequest, *args, **kwargs) -> JsonResponse:
        """Get data for each symbol."""
        try:
            params = self.get_params(request)
        except ValueError as exc:
            return JsonResponse({"error": str(exc)}, status=400)
        task_state = self.get_task_state()
        if not task_state.can_run():
            return JsonResponse({"ok": True, "skipped": "backoff"})
        if not task_state.acquire():
            return JsonResponse({"ok": True, "skipped": "locked"})
        try:
            for candle, timestamp_from, timestamp_to, retry in params:
                logger.info("{candle}: starting...".format(**{"candle": str(candle)}))
                candle.candles(timestamp_from, timestamp_to, retry)
            for candle, _timestamp_from, _timestamp_to, _retry in params:
                convert_candle_cache_to_daily(candle)
        except Exception:
            task_state.mark_recent_error()
            raise
        else:
            task_state.clear_recent_error()
        finally:
            task_state.release()
        return JsonResponse({"ok": True})
