import logging

from django.db.models import QuerySet
from django_filters.rest_framework import DjangoFilterBackend
from rest_framework.exceptions import ValidationError
from rest_framework.generics import ListAPIView
from rest_framework.request import Request
from rest_framework.response import Response

from quant_tick.constants import Exchange
from quant_tick.exchanges import api
from quant_tick.filters import SymbolFilter
from quant_tick.models import Symbol, TaskState, TradeData
from quant_tick.serializers import TimeAgoWithRetrySerializer
from quant_tick.storage import convert_trade_data_to_daily

logger = logging.getLogger(__name__)


class TradeDataView(ListAPIView):
    """Trade data view."""

    queryset = TradeData.objects.all()
    filter_backends = (DjangoFilterBackend,)
    filterset_class = SymbolFilter


class AggregateTradeDataView(ListAPIView):
    """Aggregate trade data view."""

    queryset = Symbol.objects.filter(is_active=True)
    filter_backends = (DjangoFilterBackend,)
    filterset_class = SymbolFilter

    def get_queryset(self) -> QuerySet:
        """Get queryset."""
        return self.queryset.filter(exchange=self.get_exchange())

    def get_exchange(self) -> str:
        """Get the required path exchange."""
        exchange = self.kwargs.get("exchange")
        if exchange not in Exchange.values:
            raise ValidationError({"exchange": "Invalid exchange."})
        return exchange

    def get_task_state(self, exchange: str) -> TaskState:
        """Get the exchange-scoped task state."""
        task_state, _ = TaskState.objects.get_or_create(
            name="aggregate_trades",
            exchange=exchange,
        )
        return task_state

    def get_params(self, request: Request) -> list[tuple]:
        """Get params."""
        serializer = TimeAgoWithRetrySerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data
        queryset = self.get_queryset()
        return [
            (
                symbol,
                data["timestamp_from"],
                data["timestamp_to"],
                data["retry"],
            )
            for symbol in self.filter_queryset(queryset)
        ]

    def get(self, request: Request, *args, **kwargs) -> Response:
        """Get data for each symbol."""
        exchange = self.get_exchange()
        task_state = self.get_task_state(exchange)
        if not task_state.can_run():
            return Response({"ok": True, "skipped": "backoff"})
        if not task_state.acquire():
            return Response({"ok": True, "skipped": "locked"})
        params = self.get_params(request)
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
        return Response({"ok": True})
