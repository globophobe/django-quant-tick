import logging

import pandas as pd
from django.http import HttpRequest, JsonResponse
from django.views import View

from quant_tick.lib import get_min_time
from quant_tick.models import Candle, Symbol
from quant_tick.storage import convert_candle_cache_to_daily, convert_trade_data_to_daily
from quant_tick.views.aggregate_trades import get_request_params

logger = logging.getLogger(__name__)


class CompactView(View):
    """Compact trade data and candle cache."""

    symbol_queryset = Symbol.objects.filter(is_active=True)
    candle_queryset = Candle.objects.filter(is_active=True).select_related("symbol")

    def get(self, request: HttpRequest, *args, **kwargs) -> JsonResponse:
        try:
            timestamp_from, timestamp_to, _retry = get_request_params(request)
            min_timestamp_from = get_min_time(timestamp_to - pd.Timedelta("7d"), "1d")
            timestamp_from = max(timestamp_from, min_timestamp_from)
        except ValueError as exc:
            return JsonResponse({"error": str(exc)}, status=400)

        for symbol in self.symbol_queryset:
            try:
                convert_trade_data_to_daily(symbol, timestamp_from, timestamp_to)
            except Exception:
                logger.exception("TradeData compaction failed for %s", symbol)

        for candle in self.candle_queryset:
            try:
                convert_candle_cache_to_daily(candle)
            except Exception:
                logger.exception("CandleCache compaction failed for %s", candle)

        return JsonResponse({"ok": True})
