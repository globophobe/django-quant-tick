import logging

from django_filters.rest_framework import DjangoFilterBackend
from rest_framework.generics import ListAPIView
from rest_framework.request import Request
from rest_framework.response import Response

from quant_candles.controllers import aggregate_trade_summary
from quant_candles.exchanges import api
from quant_candles.filters import SymbolFilter
from quant_candles.models import Symbol
from quant_candles.serializers import TimeAgoWithRetrySerializer
from quant_candles.storage import convert_trade_data_to_hourly

logger = logging.getLogger(__name__)


class AggregateTradeDataView(ListAPIView):
    queryset = Symbol.objects.all()
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
            convert_trade_data_to_hourly(symbol, timestamp_from, timestamp_to)
            aggregate_trade_summary(symbol, timestamp_from, timestamp_to, retry)
        return Response({"ok": True})
