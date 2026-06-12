import logging

import pandas as pd
from django.http import HttpRequest, JsonResponse
from django.views import View

from quant_tick.lib import get_min_time
from quant_tick.constants import TaskType
from quant_tick.models import Candle, Symbol, TaskState
from quant_tick.models.task_state import TASK_STATE_EXCHANGE_ALL
from quant_tick.storage import (
    convert_candle_cache_to_daily,
    convert_trade_data_to_daily,
    get_compact_max_timestamp_to,
)
from quant_tick.views.aggregate_trades import get_request_params

logger = logging.getLogger(__name__)

COMPACT_TASK_EXCHANGE = TASK_STATE_EXCHANGE_ALL
COMPACT_TASK_API_SYMBOL = "all"

class CompactView(View):
    """Compact trade data and candle cache."""

    symbol_queryset = Symbol.objects.filter(is_active=True)
    candle_queryset = Candle.objects.filter(is_active=True).select_related("symbol")

    def get_task_state(self) -> TaskState:
        task_state, _ = TaskState.objects.get_or_create(
            task_type=TaskType.COMPACT,
            exchange=COMPACT_TASK_EXCHANGE,
            api_symbol=COMPACT_TASK_API_SYMBOL,
        )
        return task_state

    def get(self, request: HttpRequest, *args, **kwargs) -> JsonResponse:
        try:
            timestamp_from, timestamp_to = get_request_params(request)
            min_timestamp_from = get_min_time(timestamp_to - pd.Timedelta("7d"), "1d")
            timestamp_from = max(timestamp_from, min_timestamp_from)
            timestamp_to = get_compact_max_timestamp_to(timestamp_to)
        except ValueError as exc:
            return JsonResponse({"error": str(exc)}, status=400)

        task_state = self.get_task_state()
        if not task_state.can_run():
            return JsonResponse({"ok": True, "skipped": "backoff"})
        if not task_state.acquire():
            return JsonResponse({"ok": True, "skipped": "locked"})

        failed = 0
        try:
            for symbol in self.symbol_queryset:
                try:
                    convert_trade_data_to_daily(symbol, timestamp_from, timestamp_to)
                except Exception:
                    failed += 1
                    logger.exception("TradeData compaction failed for %s", symbol)

            for candle in self.candle_queryset:
                try:
                    convert_candle_cache_to_daily(candle)
                except Exception:
                    failed += 1
                    logger.exception("CandleCache compaction failed for %s", candle)

            if failed:
                task_state.mark_recent_error(backoff=False)
            else:
                task_state.clear_recent_error()
        finally:
            task_state.release()

        return JsonResponse({"ok": True})
