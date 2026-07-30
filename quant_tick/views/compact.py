import logging

import pandas as pd
from django.http import HttpRequest, JsonResponse
from django.views import View

from quant_tick.lib import get_min_time
from quant_tick.constants import TaskType
from quant_tick.models import Candle, Symbol, TaskState
from quant_tick.models.task_state import TASK_STATE_EXCHANGE_ALL
from quant_tick.services.task_lease import (
    TaskLeaseHeartbeat,
    TaskLeaseLost,
    clear_task_recent_error,
    mark_task_recent_error,
)
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
        lease_heartbeat = TaskLeaseHeartbeat(state=task_state)
        try:
            lease_heartbeat.start()
            for symbol in self.symbol_queryset.all():
                try:
                    convert_trade_data_to_daily(
                        symbol,
                        timestamp_from,
                        timestamp_to,
                        assert_lease_owned=lease_heartbeat.assert_owned,
                    )
                except Exception:
                    lease_heartbeat.assert_owned()
                    failed += 1
                    logger.exception("TradeData compaction failed for %s", symbol)
                else:
                    lease_heartbeat.assert_owned()

            for candle in self.candle_queryset.all():
                try:
                    convert_candle_cache_to_daily(
                        candle,
                        assert_lease_owned=lease_heartbeat.assert_owned,
                    )
                except Exception:
                    lease_heartbeat.assert_owned()
                    failed += 1
                    logger.exception("CandleCache compaction failed for %s", candle)
                else:
                    lease_heartbeat.assert_owned()

            if failed:
                mark_task_recent_error(state=task_state, backoff=False)
            else:
                clear_task_recent_error(state=task_state)
        except TaskLeaseLost:
            logger.exception("Compaction task lease ownership lost")
            raise
        finally:
            lease_heartbeat.stop()
            task_state.release()

        return JsonResponse({"ok": True})
