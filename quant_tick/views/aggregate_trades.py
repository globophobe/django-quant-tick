import logging
from datetime import datetime

import httpx
import pandas as pd
from django.conf import settings
from django.db.models import QuerySet
from django.http import HttpRequest, JsonResponse
from django.views import View

from quant_tick.constants import TaskType
from quant_tick.exchanges import api
from quant_tick.lib import get_current_time, get_min_time
from quant_tick.lib.download import ArchiveDownloadError
from quant_tick.models import Symbol, TaskState, TradeData
from quant_tick.pubsub import (
    get_trade_pubsub_configs,
    ingest_trades_from_pubsub,
)
from quant_tick.forms import (
    AggregateTradeRequestForm,
    TimeRangeRequestForm,
    format_form_errors,
    parse_time_delta,
)

logger = logging.getLogger(__name__)

TRANSIENT_COLLECTION_ERRORS = (httpx.TransportError,)


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


def serialize_timestamp(timestamp: datetime) -> str:
    value = timestamp.isoformat()
    return value.removesuffix("+00:00") + "Z" if value.endswith("+00:00") else value


def get_candle_retry_min_timestamp_from(timestamp: datetime) -> datetime:
    timestamp_from = get_min_time(timestamp, "1h")
    midpoint = timestamp_from + pd.Timedelta("30min")
    if timestamp >= midpoint:
        return midpoint
    return timestamp_from


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

    def get_latest_trade_data(self, symbol: Symbol) -> TradeData | None:
        return (
            TradeData.objects.filter(symbol=symbol)
            .only("timestamp", "frequency")
            .order_by("-timestamp", "-frequency")
            .first()
        )

    def get_pubsub_window(
        self,
        symbol: Symbol,
        timestamp_from,
        timestamp_to,
    ) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        latest = self.get_latest_trade_data(symbol)
        if latest is None:
            return None
        latest_to = latest.timestamp + pd.Timedelta(f"{int(latest.frequency)}min")
        pubsub_from = max(timestamp_from, latest_to)
        if pubsub_from >= timestamp_to:
            return None
        return pubsub_from, timestamp_to

    def get_recent_retry_window(
        self,
        timestamp_from,
        timestamp_to,
    ) -> tuple[pd.Timestamp, pd.Timestamp] | None:
        time_ago = getattr(settings, "QUANT_TICK_TRADE_RECENT_RETRY_TIME_AGO", "1h")
        if not time_ago:
            return None
        retry_from = get_min_time(timestamp_to - parse_time_delta(time_ago), "1min")
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

    def get_candle_request(
        self,
        symbol: Symbol,
        retry_from=None,
    ) -> dict:
        data = {
            "exchange": symbol.exchange,
            "api_symbol": symbol.api_symbol,
        }
        if retry_from is not None:
            data["timestamp_from"] = serialize_timestamp(
                get_candle_retry_min_timestamp_from(retry_from)
            )
        return data

    def ingest_pubsub_trades(
        self,
        symbol: Symbol,
        timestamp_from,
        timestamp_to,
    ):
        configs = get_trade_pubsub_configs(symbol)
        if not configs:
            return None
        try:
            result = ingest_trades_from_pubsub(
                symbol=symbol,
                timestamp_from=timestamp_from,
                timestamp_to=timestamp_to,
                configs=configs,
            )
        except Exception:
            logger.exception("%s: pub/sub ingest failed", symbol)
            return None
        streams = ",".join(stream for stream, _subscription in configs)
        logger.info(
            "%s: %s pub/sub ingest pulled=%s processed=%s ok=%s "
            "failed=%s pending=%s ignored=%s",
            symbol,
            streams,
            result.pulled,
            result.processed,
            result.ok,
            result.failed,
            result.pending,
            result.ignored,
        )
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
            candle_requests = []
            for task_state, symbol, timestamp_from, timestamp_to in tasks:
                try:
                    logger.info(
                        "{symbol}: starting...".format(**{"symbol": str(symbol)})
                    )
                    pubsub_window = self.get_pubsub_window(
                        symbol,
                        timestamp_from,
                        timestamp_to,
                    )
                    if pubsub_window is not None:
                        self.ingest_pubsub_trades(symbol, *pubsub_window)
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
                    candle_requests.append(
                        self.get_candle_request(symbol, candle_retry_from)
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
        except ArchiveDownloadError:
            raise
        finally:
            for task_state, *_rest in tasks:
                task_state.release()
        response = {"ok": True}
        if len(candle_requests) == 1:
            response.update(candle_requests[0])
        else:
            response["candle_requests"] = candle_requests
        return JsonResponse(response)
