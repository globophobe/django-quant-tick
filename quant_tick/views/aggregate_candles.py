import logging

from django.db.models import QuerySet
from django.http import HttpRequest, JsonResponse
from django.views import View

from quant_tick.constants import Exchange, TaskType
from quant_tick.models import Candle, TaskState
from quant_tick.views.aggregate_trades import get_request_params

logger = logging.getLogger(__name__)


class AggregateCandleView(View):

    def get_exchange(self) -> str:
        exchange = self.request.GET.get("exchange", "")
        if exchange and exchange not in Exchange.values:
            raise ValueError("Invalid exchange.")
        return exchange

    def get_task_state(self, exchange: str) -> TaskState:
        task_state, _ = TaskState.objects.get_or_create(
            task_type=TaskType.AGGREGATE_CANDLES,
            exchange=exchange,
        )
        return task_state

    def get_queryset(self) -> QuerySet:
        queryset = Candle.objects.filter(is_active=True).select_related("symbol")
        exchange = self.get_exchange()
        if exchange:
            queryset = queryset.filter(symbol__exchange=exchange)
        return queryset

    def get_params(self, request: HttpRequest) -> list[tuple]:
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
        try:
            exchange = self.get_exchange()
            params = self.get_params(request)
        except ValueError as exc:
            return JsonResponse({"error": str(exc)}, status=400)
        task_state = self.get_task_state(exchange)
        if not task_state.can_run():
            return JsonResponse({"ok": True, "skipped": "backoff"})
        if not task_state.acquire():
            return JsonResponse({"ok": True, "skipped": "locked"})
        try:
            for candle, timestamp_from, timestamp_to, retry in params:
                logger.info("{candle}: starting...".format(**{"candle": str(candle)}))
                candle.candles(timestamp_from, timestamp_to, retry)
        except Exception:
            task_state.mark_recent_error()
            raise
        else:
            task_state.clear_recent_error()
        finally:
            task_state.release()
        return JsonResponse({"ok": True})
