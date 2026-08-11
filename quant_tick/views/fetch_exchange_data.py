import logging
from collections.abc import Callable

from django.db.models import Q
from django.http import HttpRequest, JsonResponse
from django.views import View

from quant_tick.constants import Exchange, SymbolType, TaskType
from quant_tick.exchanges.api import (
    exchange_candles as fetch_symbol_exchange_candles,
)
from quant_tick.exchanges.api import (
    funding as fetch_symbol_funding,
)
from quant_tick.forms import (
    FetchExchangeDataRequestForm,
    format_form_errors,
)
from quant_tick.lib.task_errors import is_transient_task_error
from quant_tick.models import Symbol, TaskState
from quant_tick.services.task_lease import (
    TaskLeaseHeartbeat,
    TaskLeaseLost,
    clear_task_recent_error,
    mark_task_recent_error,
)
from quant_tick.views.aggregate_trades import (
    TRANSIENT_COLLECTION_ERRORS,
    get_timestamp_range,
    is_soft_collection_error,
)

logger = logging.getLogger(__name__)

FUNDING_SUPPORTED_EXCHANGES = (
    Exchange.BINANCE_FUTURES,
    Exchange.BITFINEX,
    Exchange.BITMEX,
    Exchange.BYBIT_LINEAR,
    Exchange.BYBIT_INVERSE,
    Exchange.DERIBIT,
    Exchange.HYPERLIQUID,
)


class FetchExchangeDataView(View):
    queryset = Symbol.objects.filter(is_active=True)

    def get_configured_exchanges(self) -> tuple[str, ...]:
        configured_exchange_candles = ~Q(exchange_candle_resolution="")
        configured_funding = Q(
            exchange__in=FUNDING_SUPPORTED_EXCHANGES,
            symbol_type=SymbolType.PERPETUAL,
        )
        exchanges = (
            self.queryset.filter(configured_funding | configured_exchange_candles)
            .order_by("exchange")
            .values_list("exchange", flat=True)
            .distinct()
        )
        return tuple(exchanges)

    def has_exchange_data(self, exchange: str) -> bool:
        queryset = self.queryset.filter(exchange=exchange)
        configured_exchange_candles = ~Q(exchange_candle_resolution="")
        if exchange in FUNDING_SUPPORTED_EXCHANGES:
            return queryset.filter(
                Q(symbol_type=SymbolType.PERPETUAL) | configured_exchange_candles
            ).exists()
        return queryset.exclude(exchange_candle_resolution="").exists()

    def get_query_form(self, request: HttpRequest) -> FetchExchangeDataRequestForm:
        form = FetchExchangeDataRequestForm(request.GET)
        if not form.is_valid():
            raise ValueError(format_form_errors(form))
        return form

    def get_exchanges(self, exchange: str) -> tuple[str, ...]:
        if not exchange:
            return self.get_configured_exchanges()
        if not self.has_exchange_data(exchange):
            raise ValueError("Exchange data is not configured for this exchange.")
        return (exchange,)

    def get_symbols(self, exchange: str, api_symbol: str) -> list[Symbol]:
        queryset = self.queryset.filter(exchange=exchange)
        if api_symbol:
            queryset = queryset.filter(api_symbol=api_symbol)
        configured = ~Q(exchange_candle_resolution="")
        if exchange in FUNDING_SUPPORTED_EXCHANGES:
            configured |= Q(symbol_type=SymbolType.PERPETUAL)
        return list(queryset.filter(configured))

    def get_task_state(self, symbol: Symbol) -> TaskState:
        task_state, _ = TaskState.objects.get_or_create(
            task_type=TaskType.FETCH_EXCHANGE_DATA,
            exchange=symbol.exchange,
            api_symbol=symbol.api_symbol,
        )
        return task_state

    def fetch_symbol_exchange_data(
        self,
        symbol: Symbol,
        timestamp_from,
        timestamp_to,
        retry: bool,
        assert_lease_owned: Callable[[], None] | None = None,
    ) -> dict:
        counts = {"funding": 0, "exchange_candles": 0}
        if (
            symbol.exchange in FUNDING_SUPPORTED_EXCHANGES
            and symbol.symbol_type == SymbolType.PERPETUAL
        ):
            logger.info(f"{symbol!s}: funding starting...")
            fetch_symbol_funding(
                symbol,
                timestamp_from,
                timestamp_to,
                retry,
                assert_lease_owned=assert_lease_owned,
            )
            counts["funding"] = 1
        if symbol.exchange_candle_resolution:
            logger.info(f"{symbol!s}: exchange candles starting...")
            fetch_symbol_exchange_candles(
                symbol,
                timestamp_from,
                timestamp_to,
                resolution=symbol.exchange_candle_resolution,
                retry=retry,
                assert_lease_owned=assert_lease_owned,
            )
            counts["exchange_candles"] = 1
        return counts

    def fetch_exchange_data(
        self,
        exchange: str,
        timestamp_from,
        timestamp_to,
        retry: bool,
        api_symbol: str,
    ) -> dict:
        counts = {"funding": 0, "exchange_candles": 0, "failed": 0, "skipped": 0}
        for symbol in self.get_symbols(exchange, api_symbol):
            task_state = self.get_task_state(symbol)
            if not task_state.can_run():
                counts["skipped"] += 1
                continue
            if not task_state.acquire():
                counts["skipped"] += 1
                continue
            lease_heartbeat = TaskLeaseHeartbeat(state=task_state)
            try:
                try:
                    lease_heartbeat.start()
                    symbol_counts = self.fetch_symbol_exchange_data(
                        symbol,
                        timestamp_from,
                        timestamp_to,
                        retry,
                        assert_lease_owned=lease_heartbeat.assert_owned,
                    )
                except TaskLeaseLost:
                    raise
                except TRANSIENT_COLLECTION_ERRORS:
                    mark_task_recent_error(state=task_state, backoff=False)
                    counts["failed"] += 1
                    logger.exception("%s: fetch exchange data failed", symbol)
                except Exception as exc:
                    if is_soft_collection_error(exc):
                        lease_heartbeat.assert_owned()
                        counts["failed"] += 1
                        logger.warning(
                            "%s: fetch exchange data skipped: %s", symbol, exc
                        )
                    elif is_transient_task_error(exc):
                        mark_task_recent_error(state=task_state, backoff=False)
                        counts["failed"] += 1
                        logger.exception("%s: fetch exchange data failed", symbol)
                    else:
                        mark_task_recent_error(state=task_state)
                        counts["failed"] += 1
                        logger.exception("%s: fetch exchange data failed", symbol)
                else:
                    clear_task_recent_error(state=task_state)
                    counts["funding"] += symbol_counts["funding"]
                    counts["exchange_candles"] += symbol_counts["exchange_candles"]
            except TaskLeaseLost:
                counts["failed"] += 1
                logger.exception("%s: task lease ownership lost", symbol)
            finally:
                lease_heartbeat.stop()
                task_state.release()
        return counts

    def fetch_exchange_data_for_exchanges(
        self,
        exchanges: tuple[str, ...],
        timestamp_from,
        timestamp_to,
        retry: bool,
        api_symbol: str,
    ) -> dict:
        data = {}
        totals = {"funding": 0, "exchange_candles": 0, "failed": 0, "skipped": 0}
        for exchange in exchanges:
            counts = self.fetch_exchange_data(
                exchange,
                timestamp_from,
                timestamp_to,
                retry,
                api_symbol,
            )
            data[exchange] = counts
            totals["funding"] += counts["funding"]
            totals["exchange_candles"] += counts["exchange_candles"]
            totals["failed"] += counts["failed"]
            totals["skipped"] += counts["skipped"]
        return {**totals, "exchanges": data}

    def get(self, request: HttpRequest, *args, **kwargs) -> JsonResponse:
        try:
            form = self.get_query_form(request)
            query = form.cleaned_data
            exchanges = self.get_exchanges(query["exchange"])
            timestamp_from, timestamp_to = get_timestamp_range(query["time_ago"])
        except ValueError as exc:
            return JsonResponse({"error": str(exc)}, status=400)
        counts = self.fetch_exchange_data_for_exchanges(
            exchanges,
            timestamp_from,
            timestamp_to,
            False,
            query["api_symbol"],
        )
        return JsonResponse({"ok": counts["failed"] == 0, **counts})
