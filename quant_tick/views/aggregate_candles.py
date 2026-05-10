import logging
from datetime import UTC, datetime

from django.db.models import QuerySet
from django.http import HttpRequest, JsonResponse
from django.views import View

from quant_tick.constants import TaskType
from quant_tick.lib import get_current_time, get_min_time
from quant_tick.models import Candle, Symbol, TaskState
from quant_tick.forms import get_candle_request_data, parse_request_body

logger = logging.getLogger(__name__)

DEFAULT_TIMESTAMP_FROM = datetime(2009, 1, 3, tzinfo=UTC)


class AggregateCandleView(View):
    def get_task_state(self, symbol: Symbol) -> TaskState:
        task_state, _ = TaskState.objects.get_or_create(
            task_type=TaskType.AGGREGATE_CANDLES,
            exchange=symbol.exchange,
            api_symbol=symbol.api_symbol,
        )
        return task_state

    def get_queryset(self, payload: dict) -> QuerySet:
        queryset = Candle.objects.filter(is_active=True).select_related("symbol")
        exchange = payload.get("exchange") or ""
        if exchange:
            queryset = queryset.filter(symbol__exchange=exchange)
        api_symbol = payload.get("api_symbol")
        if api_symbol:
            queryset = queryset.filter(symbol__api_symbol=api_symbol)
        return queryset

    def get_request_params(self, payload: dict) -> tuple[datetime, datetime, bool]:
        timestamp_to = get_min_time(get_current_time(), "1min")
        if payload.get("timestamp_from"):
            timestamp_from = payload["timestamp_from"]
            retry = True
        elif payload.get("time_ago"):
            timestamp_from = get_min_time(timestamp_to - payload["time_ago"], "1d")
            retry = False
        else:
            timestamp_from = DEFAULT_TIMESTAMP_FROM
            retry = False
        return timestamp_from, timestamp_to, retry

    def get_params(self, payload: dict) -> list[tuple]:
        timestamp_from, timestamp_to, retry = self.get_request_params(payload)
        queryset = self.get_queryset(payload)
        return [
            (
                candle,
                timestamp_from,
                timestamp_to,
                retry,
            )
            for candle in queryset
        ]

    def acquire_tasks(self, params: list[tuple]) -> tuple[list[tuple], dict[str, int]]:
        groups = {}
        for candle, timestamp_from, timestamp_to, retry in params:
            key = candle.symbol_id
            groups.setdefault(key, []).append(
                (candle, timestamp_from, timestamp_to, retry)
            )
        tasks = []
        skipped = {"backoff": 0, "locked": 0}
        for items in groups.values():
            symbol = items[0][0].symbol
            task_state = self.get_task_state(symbol)
            if not task_state.can_run():
                skipped["backoff"] += len(items)
                continue
            if not task_state.acquire():
                skipped["locked"] += len(items)
                continue
            tasks.append((task_state, items))
        return tasks, skipped

    def handle(self, request: HttpRequest) -> JsonResponse:
        try:
            payload = parse_request_body(request)
            payloads = get_candle_request_data(payload)
            all_params = []
            skipped = {"backoff": 0, "locked": 0}
            tasks = []
            for candle_payload in payloads:
                params = self.get_params(candle_payload)
                all_params += params
                payload_tasks, payload_skipped = self.acquire_tasks(params)
                skipped["backoff"] += payload_skipped["backoff"]
                skipped["locked"] += payload_skipped["locked"]
                tasks += payload_tasks
            if all_params and not tasks:
                skipped_reason = "backoff" if skipped["backoff"] else "locked"
                return JsonResponse({"ok": True, "skipped": skipped_reason})
            processed = 0
            for task_state, items in tasks:
                try:
                    for candle, timestamp_from, timestamp_to, retry in items:
                        logger.info(
                            "{candle}: starting...".format(**{"candle": str(candle)})
                        )
                        candle.candles(timestamp_from, timestamp_to, retry)
                        processed += 1
                except Exception:
                    task_state.mark_recent_error()
                    raise
                else:
                    task_state.clear_recent_error()
                finally:
                    task_state.release()
        except ValueError as exc:
            return JsonResponse({"error": str(exc)}, status=400)
        finally:
            for task_state, _items in locals().get("tasks", []):
                task_state.release()
        return JsonResponse({"ok": True, "processed": processed})

    def get(self, request: HttpRequest, *args, **kwargs) -> JsonResponse:
        return self.handle(request)

    def post(self, request: HttpRequest, *args, **kwargs) -> JsonResponse:
        return self.handle(request)
