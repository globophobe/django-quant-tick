import logging
from collections.abc import Iterable

from django.db.models import Q
from django.http import HttpRequest, JsonResponse
from django.views import View

from quant_tick.constants import Exchange, SymbolType, TaskType
from quant_tick.exchanges.api import (
    exchange_candles as fetch_symbol_exchange_candles,
    funding as fetch_symbol_funding,
)
from quant_tick.models import Symbol, TaskState
from quant_tick.views.aggregate_trades import (
    TRANSIENT_COLLECTION_ERRORS,
    get_request_params,
)

logger = logging.getLogger(__name__)

FUNDING_SUPPORTED_EXCHANGES = (
    Exchange.BINANCE_FUTURES,
    Exchange.BITMEX,
    Exchange.COINBASE_ADVANCED,
    Exchange.HYPERLIQUID,
)


def fetch_funding(
    symbols: Iterable[Symbol],
    timestamp_from,
    timestamp_to,
    retry: bool,
) -> int:
    count = 0
    for symbol in symbols:
        if symbol.symbol_type != SymbolType.PERPETUAL:
            continue
        logger.info("{symbol}: funding starting...".format(symbol=str(symbol)))
        fetch_symbol_funding(symbol, timestamp_from, timestamp_to, retry)
        count += 1
    return count


def fetch_exchange_candles(
    symbols: Iterable[Symbol],
    timestamp_from,
    timestamp_to,
    retry: bool,
) -> int:
    count = 0
    for symbol in symbols:
        if not symbol.exchange_candle_resolution:
            continue
        logger.info(
            "{symbol}: exchange candles starting...".format(symbol=str(symbol))
        )
        fetch_symbol_exchange_candles(
            symbol,
            timestamp_from,
            timestamp_to,
            resolution=symbol.exchange_candle_resolution,
            retry=retry,
        )
        count += 1
    return count


class FetchExchangeDataView(View):
    queryset = Symbol.objects.filter(is_active=True)

    def get_configured_exchanges(self) -> tuple[str, ...]:
        configured_exchange_candles = ~Q(exchange_candle_resolution="")
        exchanges = (
            self.queryset.filter(
                Q(exchange__in=FUNDING_SUPPORTED_EXCHANGES)
                | configured_exchange_candles
            )
            .order_by("exchange")
            .values_list("exchange", flat=True)
            .distinct()
        )
        return tuple(exchanges)

    def has_exchange_data(self, exchange: str) -> bool:
        queryset = self.queryset.filter(exchange=exchange)
        if exchange in FUNDING_SUPPORTED_EXCHANGES:
            return queryset.exists()
        return (
            queryset
            .exclude(exchange_candle_resolution="")
            .exists()
        )

    def get_exchanges(self, request: HttpRequest) -> tuple[str, ...]:
        exchange = request.GET.get("exchange")
        if exchange is None:
            return self.get_configured_exchanges()
        if exchange not in Exchange.values:
            raise ValueError("Invalid exchange.")
        if not self.has_exchange_data(exchange):
            raise ValueError("Exchange data is not configured for this exchange.")
        return (exchange,)

    def get_symbols(self, exchange: str) -> list[Symbol]:
        queryset = self.queryset.filter(exchange=exchange)
        api_symbol = self.request.GET.get("api_symbol")
        if api_symbol:
            queryset = queryset.filter(api_symbol=api_symbol)
        return list(queryset)

    def get_task_state(self) -> TaskState:
        task_state, _ = TaskState.objects.get_or_create(
            task_type=TaskType.FETCH_EXCHANGE_DATA,
        )
        return task_state

    def fetch_exchange_data(
        self,
        exchange: str,
        timestamp_from,
        timestamp_to,
        retry: bool,
    ) -> dict:
        symbols = self.get_symbols(exchange)
        counts = {"funding": 0, "exchange_candles": 0}
        if exchange in FUNDING_SUPPORTED_EXCHANGES:
            counts["funding"] = fetch_funding(
                symbols,
                timestamp_from,
                timestamp_to,
                retry,
            )
        counts["exchange_candles"] = fetch_exchange_candles(
            symbols,
            timestamp_from,
            timestamp_to,
            retry,
        )
        return counts

    def fetch_exchange_data_for_exchanges(
        self,
        exchanges: tuple[str, ...],
        timestamp_from,
        timestamp_to,
        retry: bool,
    ) -> dict:
        data = {}
        totals = {"funding": 0, "exchange_candles": 0}
        for exchange in exchanges:
            counts = self.fetch_exchange_data(
                exchange,
                timestamp_from,
                timestamp_to,
                retry,
            )
            data[exchange] = counts
            totals["funding"] += counts["funding"]
            totals["exchange_candles"] += counts["exchange_candles"]
        return {**totals, "exchanges": data}

    def get(self, request: HttpRequest, *args, **kwargs) -> JsonResponse:
        try:
            exchanges = self.get_exchanges(request)
            timestamp_from, timestamp_to, retry = get_request_params(request)
        except ValueError as exc:
            return JsonResponse({"error": str(exc)}, status=400)
        task_state = self.get_task_state()
        if not task_state.can_run():
            return JsonResponse({"ok": True, "skipped": "backoff"})
        if not task_state.acquire():
            return JsonResponse({"ok": True, "skipped": "locked"})
        try:
            counts = self.fetch_exchange_data_for_exchanges(
                exchanges,
                timestamp_from,
                timestamp_to,
                retry,
            )
        except TRANSIENT_COLLECTION_ERRORS:
            task_state.mark_recent_error(backoff=False)
            raise
        except Exception:
            task_state.mark_recent_error()
            raise
        else:
            task_state.clear_recent_error()
        finally:
            task_state.release()
        return JsonResponse({"ok": True, **counts})
