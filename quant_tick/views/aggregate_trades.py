import logging

import pandas as pd
from django.db.models import QuerySet
from django.http import HttpRequest, JsonResponse
from django.views import View

from quant_tick.constants import Exchange
from quant_tick.exchanges import api
from quant_tick.lib import get_current_time, get_min_time
from quant_tick.models import Symbol, TaskState
from quant_tick.storage import convert_trade_data_to_daily

logger = logging.getLogger(__name__)


def get_request_params(request: HttpRequest) -> tuple[pd.Timestamp, pd.Timestamp, bool]:
    """Parse aggregate request params."""
    time_ago = request.GET.get("time_ago", "1d")
    retry = request.GET.get("retry", "").lower() in {"1", "true", "yes", "on"}
    try:
        delta = pd.Timedelta(time_ago)
    except ValueError as exc:
        raise ValueError(f"Cannot parse {time_ago}.") from exc
    timestamp_to = get_min_time(get_current_time(), "1min")
    timestamp_from = timestamp_to - delta
    return timestamp_from, timestamp_to, retry


class AggregateTradeDataView(View):
    """Aggregate trade data view."""

    queryset = Symbol.objects.filter(is_active=True)

    def get_queryset(self) -> QuerySet:
        """Get queryset."""
        queryset = self.queryset.filter(exchange=self.get_exchange())
        api_symbol = self.request.GET.get("api_symbol")
        if api_symbol:
            queryset = queryset.filter(api_symbol=api_symbol)
        return queryset

    def get_exchange(self) -> str:
        """Get the required path exchange."""
        exchange = self.kwargs.get("exchange")
        if exchange not in Exchange.values:
            raise ValueError("Invalid exchange.")
        return exchange

    def get_task_state(self, exchange: str) -> TaskState:
        """Get the exchange-scoped task state."""
        task_state, _ = TaskState.objects.get_or_create(
            name="aggregate_trades",
            exchange=exchange,
        )
        return task_state

    def get_params(self, request: HttpRequest) -> list[tuple]:
        """Get params."""
        timestamp_from, timestamp_to, retry = get_request_params(request)
        queryset = self.get_queryset()
        return [
            (
                symbol,
                timestamp_from,
                timestamp_to,
                retry,
            )
            for symbol in queryset
        ]

    def get(self, request: HttpRequest, *args, **kwargs) -> JsonResponse:
        """Get data for each symbol."""
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
            for symbol, timestamp_from, timestamp_to, retry in params:
                logger.info("{symbol}: starting...".format(**{"symbol": str(symbol)}))
                api(symbol, timestamp_from, timestamp_to, retry)
            for symbol, timestamp_from, timestamp_to, _retry in params:
                convert_trade_data_to_daily(symbol, timestamp_from, timestamp_to)
        except Exception:
            task_state.mark_recent_error()
            raise
        else:
            task_state.clear_recent_error()
        finally:
            task_state.release()
        return JsonResponse({"ok": True})
