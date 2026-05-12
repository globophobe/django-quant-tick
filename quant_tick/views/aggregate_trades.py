import logging
from datetime import datetime

import httpx
import pandas as pd
from django.db.models import QuerySet
from django.http import HttpRequest, JsonResponse
from django.views import View

from quant_tick.constants import FileData, Frequency, TaskType
from quant_tick.controllers import TradeDataIterator
from quant_tick.exchanges import api, candles_api
from quant_tick.lib import get_current_time, get_min_time
from quant_tick.lib.download import ArchiveDownloadError
from quant_tick.models import Symbol, TaskState, TradeData, WebSocketData
from quant_tick.services.aggregate_candles import aggregate_candle_data
from quant_tick.forms import (
    AggregateTradeRequestForm,
    TimeRangeRequestForm,
    format_form_errors,
)

logger = logging.getLogger(__name__)

TRANSIENT_COLLECTION_ERRORS = (httpx.TransportError,)
TRADE_COMPARE_COLUMNS = (
    "timestamp",
    "nanoseconds",
    "price",
    "volume",
    "notional",
    "tickRule",
    "ticks",
    "high",
    "low",
    "totalBuyVolume",
    "totalVolume",
    "totalBuyNotional",
    "totalNotional",
    "totalBuyTicks",
    "totalTicks",
)


def get_timestamp_range(delta: pd.Timedelta) -> tuple[pd.Timestamp, pd.Timestamp]:
    timestamp_to = get_min_time(get_current_time(), "1min")
    timestamp_from = get_min_time(timestamp_to - delta, "1d")
    return timestamp_from, timestamp_to


def get_request_params(request: HttpRequest) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Parse aggregate request params."""
    form = TimeRangeRequestForm(request.GET)
    if not form.is_valid():
        raise ValueError(format_form_errors(form))
    return get_timestamp_range(form.cleaned_data["time_ago"])


def get_candle_retry_min_timestamp_from(timestamp: datetime) -> datetime:
    return get_min_time(timestamp, "1h")


class AggregateTradeDataView(View):
    queryset = Symbol.objects.filter(is_active=True)

    def get_query_form(self, request: HttpRequest) -> AggregateTradeRequestForm:
        data = request.GET.copy()
        data["exchange"] = self.kwargs.get("exchange")
        form = AggregateTradeRequestForm(data)
        if not form.is_valid():
            raise ValueError(format_form_errors(form))
        return form

    def get_queryset(self, params: dict) -> QuerySet:
        queryset = self.queryset.filter(exchange=params["exchange"])
        api_symbol = params["api_symbol"]
        if api_symbol:
            queryset = queryset.filter(api_symbol=api_symbol)
        return queryset

    def get_task_state(self, symbol: Symbol) -> TaskState:
        task_state, _ = TaskState.objects.get_or_create(
            task_type=TaskType.AGGREGATE_TRADES,
            exchange=symbol.exchange,
            api_symbol=symbol.api_symbol,
        )
        return task_state

    def acquire_tasks(self, params: list[tuple]) -> tuple[list[tuple], dict[str, int]]:
        tasks = []
        skipped = {"backoff": 0, "locked": 0}
        for symbol, timestamp_from, timestamp_to in params:
            task_state = self.get_task_state(symbol)
            if not task_state.can_run():
                skipped["backoff"] += 1
                continue
            if not task_state.acquire():
                skipped["locked"] += 1
                continue
            tasks.append((task_state, symbol, timestamp_from, timestamp_to))
        return tasks, skipped

    def get_params(self, request: HttpRequest) -> list[tuple]:
        form = self.get_query_form(request)
        timestamp_from, timestamp_to = get_timestamp_range(
            form.cleaned_data["time_ago"]
        )
        queryset = self.get_queryset(form.cleaned_data)
        return [
            (
                symbol,
                timestamp_from,
                timestamp_to,
            )
            for symbol in queryset
        ]

    def get_recent_retry_window(
        self,
        timestamp_from,
        timestamp_to,
    ) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        retry_from = get_min_time(timestamp_to - pd.Timedelta("1h"), "1min")
        retry_from = max(timestamp_from, retry_from)
        if retry_from >= timestamp_to:
            return None
        return retry_from, timestamp_to

    def get_candle_retry_timestamp_from(
        self,
        symbol: Symbol,
        timestamp_from,
        timestamp_to,
    ) -> datetime | None:
        trade_data = (
            TradeData.objects.filter(
                symbol=symbol,
                timestamp__gte=timestamp_from,
                timestamp__lt=timestamp_to,
                ok=False,
            )
            .order_by("timestamp")
            .only("timestamp")
            .first()
        )
        return trade_data.timestamp if trade_data else None

    def get_candle_payload(
        self,
        symbol: Symbol,
        retry_from=None,
    ) -> dict:
        data = {
            "exchange": symbol.exchange,
            "api_symbol": symbol.api_symbol,
        }
        if retry_from is not None:
            data["timestamp_from"] = get_candle_retry_min_timestamp_from(retry_from)
        return data

    def get_websocket_data_rows(
        self,
        symbol: Symbol,
        timestamp_from,
        timestamp_to,
    ) -> list[WebSocketData]:
        if not TradeData.objects.filter(symbol=symbol).exists():
            return []
        rows = []
        for ts_from, ts_to in TradeDataIterator(symbol).iter_all(
            timestamp_from,
            timestamp_to,
        ):
            rows += list(
                WebSocketData.objects.for_symbol(symbol)
                .filter(timestamp__gte=ts_from, timestamp__lt=ts_to)
                .order_by("timestamp")
            )
        return sorted(rows, key=lambda row: row.timestamp)

    def get_rest_filtered_trades(
        self,
        symbol: Symbol,
        timestamp_from,
        timestamp_to,
    ) -> pd.DataFrame | None:
        trade_data = (
            TradeData.objects.overlapping(
                symbol,
                timestamp_from,
                timestamp_to,
                (Frequency.MINUTE, Frequency.HOUR, Frequency.DAY),
            )
            .order_by("frequency", "timestamp")
            .first()
        )
        if trade_data is None or not trade_data.has_data_frame(FileData.FILTERED):
            return None
        return TradeData._filter_frame(
            trade_data.get_data_frame(FileData.FILTERED),
            timestamp_from,
            timestamp_to,
        )

    @staticmethod
    def get_trade_compare_frame(data_frame: pd.DataFrame) -> pd.DataFrame:
        missing = set(TRADE_COMPARE_COLUMNS) - set(data_frame.columns)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"Missing trade comparison columns: {names}.")

        frame = data_frame.loc[:, TRADE_COMPARE_COLUMNS].copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        return frame.reset_index(drop=True)

    @classmethod
    def filtered_trade_frames_match(
        cls,
        websocket_trades: pd.DataFrame,
        rest_trades: pd.DataFrame,
    ) -> bool:
        websocket_frame = cls.get_trade_compare_frame(websocket_trades)
        rest_frame = cls.get_trade_compare_frame(rest_trades)
        try:
            pd.testing.assert_frame_equal(
                websocket_frame,
                rest_frame,
                check_dtype=False,
            )
        except AssertionError:
            return False
        return True

    def filtered_trades_match_rest(
        self,
        symbol: Symbol,
        timestamp_from,
        timestamp_to,
        filtered_trades: pd.DataFrame,
    ) -> bool | None:
        rest_filtered = self.get_rest_filtered_trades(
            symbol,
            timestamp_from,
            timestamp_to,
        )
        if rest_filtered is None:
            return None
        return self.filtered_trade_frames_match(filtered_trades, rest_filtered)

    def validate_websocket_data(self, symbol: Symbol, data: WebSocketData) -> str:
        timestamp_from = data.timestamp
        timestamp_to = timestamp_from + pd.Timedelta("1min")

        raw_trades, aggregated_trades, filtered_trades = data.get_data_frames(symbol)
        if (
            raw_trades is None
            and aggregated_trades is None
            and filtered_trades is None
        ):
            return "skipped"

        candles = candles_api(symbol, timestamp_from, timestamp_to, resolution="1m")
        ok = TradeData.validate(
            symbol,
            timestamp_from,
            timestamp_to,
            candles,
            raw_trades=raw_trades,
            aggregated_trades=aggregated_trades,
            filtered_trades=filtered_trades,
        )
        if ok is False:
            return "failed"
        rest_match = self.filtered_trades_match_rest(
            symbol,
            timestamp_from,
            timestamp_to,
            filtered_trades,
        )
        if rest_match is False:
            return "failed"
        if ok is True and rest_match is True:
            return "validated"
        return "unknown"

    def validate_websocket_data_rows(
        self,
        symbol: Symbol,
        timestamp_to,
        rows: list[WebSocketData],
    ) -> dict[str, int]:
        result = {"validated": 0, "failed": 0, "unknown": 0, "skipped": 0}
        for data in rows:
            bucket_to = data.timestamp + pd.Timedelta("1min")
            if bucket_to > timestamp_to:
                result["skipped"] += 1
                continue
            try:
                status = self.validate_websocket_data(symbol, data)
            except Exception:
                result["failed"] += 1
                logger.exception("%s: websocket data validation failed", symbol)
                continue
            result[status] += 1
        return result

    def get(self, request: HttpRequest, *args, **kwargs) -> JsonResponse:
        try:
            params = self.get_params(request)
        except ValueError as exc:
            return JsonResponse({"error": str(exc)}, status=400)
        tasks, skipped = self.acquire_tasks(params)
        if params and not tasks:
            skipped_reason = "backoff" if skipped["backoff"] else "locked"
            return JsonResponse({"ok": True, "skipped": skipped_reason})
        try:
            candle_payloads = []
            released = set()
            for task_state, symbol, timestamp_from, timestamp_to in tasks:
                try:
                    logger.info(
                        "{symbol}: starting...".format(**{"symbol": str(symbol)})
                    )
                    websocket_rows = self.get_websocket_data_rows(
                        symbol,
                        timestamp_from,
                        timestamp_to,
                    )
                    candle_retry_from = None
                    retry_window = self.get_recent_retry_window(
                        timestamp_from,
                        timestamp_to,
                    )
                    if retry_window is not None:
                        candle_retry_from = self.get_candle_retry_timestamp_from(
                            symbol,
                            *retry_window,
                        )
                        api(symbol, *retry_window, True)
                    api(symbol, timestamp_from, timestamp_to, False)
                    websocket_result = self.validate_websocket_data_rows(
                        symbol,
                        timestamp_to,
                        websocket_rows,
                    )
                    logger.info(
                        "%s: websocket data validation %s",
                        symbol,
                        websocket_result,
                    )
                    candle_payloads.append(
                        self.get_candle_payload(symbol, candle_retry_from)
                    )
                except ArchiveDownloadError:
                    raise
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
                    released.add(task_state.pk)
        except ArchiveDownloadError:
            raise
        finally:
            for task_state, *_rest in tasks:
                if task_state.pk not in locals().get("released", set()):
                    task_state.release()
        if candle_payloads:
            aggregate_candle_data(candle_payloads)
        return JsonResponse({"ok": True})
