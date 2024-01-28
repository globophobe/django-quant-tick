import logging

from django_filters.rest_framework import DjangoFilterBackend
from rest_framework.generics import ListAPIView
from rest_framework.request import Request
from rest_framework.response import Response

from quant_tick.exchanges import api
from quant_tick.filters import SymbolFilter
from quant_tick.models import Symbol
from quant_tick.serializers import TimeAgoWithRetrySerializer
from quant_tick.storage import convert_trade_data_to_daily

logger = logging.getLogger(__name__)


class AggregateTradeDataView(ListAPIView):
    """Aggregate trade data view."""

    queryset = Symbol.objects.filter(is_active=True)
    filter_backends = (DjangoFilterBackend,)
    filterset_class = SymbolFilter

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
        for symbol, timestamp_from, timestamp_to, retry in self.get_params(request):
            logger.info("{symbol}: starting...".format(**{"symbol": str(symbol)}))
            api(symbol, timestamp_from, timestamp_to, retry)
            convert_trade_data_to_daily(symbol, timestamp_from, timestamp_to)
        return Response({"ok": True})
