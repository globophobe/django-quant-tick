import logging

from django_filters.rest_framework import DjangoFilterBackend
from rest_framework.generics import ListAPIView
from rest_framework.request import Request
from rest_framework.response import Response

from quant_tick.filters import CandleFilter
from quant_tick.models import Candle, TaskState
from quant_tick.serializers import TimeAgoWithRetrySerializer
from quant_tick.storage import convert_candle_cache_to_daily

logger = logging.getLogger(__name__)


class AggregateCandleView(ListAPIView):
    """Aggregate candle view."""

    queryset = Candle.objects.filter(is_active=True).select_related("symbol")
    filter_backends = (DjangoFilterBackend,)
    filterset_class = CandleFilter

    def get_task_state(self) -> TaskState:
        """Get the global task state for candle aggregation."""
        task_state, _ = TaskState.objects.get_or_create(
            name="aggregate_candles",
            exchange="",
        )
        return task_state

    def get_params(self, request: Request) -> list[tuple]:
        """Get params."""
        serializer = TimeAgoWithRetrySerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data
        queryset = self.get_queryset()
        return [
            (
                candle,
                data["timestamp_from"],
                data["timestamp_to"],
                data["retry"],
            )
            for candle in self.filter_queryset(queryset)
        ]

    def get(self, request: Request, *args, **kwargs) -> Response:
        """Get data for each symbol."""
        task_state = self.get_task_state()
        if not task_state.can_run():
            return Response({"ok": True, "skipped": "backoff"})
        if not task_state.acquire():
            return Response({"ok": True, "skipped": "locked"})
        params = self.get_params(request)
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
        return Response({"ok": True})
