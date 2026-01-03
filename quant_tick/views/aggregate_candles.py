import logging

from django_filters.rest_framework import DjangoFilterBackend
from rest_framework.generics import ListAPIView
from rest_framework.request import Request
from rest_framework.response import Response

from quant_tick.controllers import aggregate_candles
from quant_tick.filters import CandleFilter
from quant_tick.models import Candle
from quant_tick.serializers import TimeAgoWithRetrySerializer
from quant_tick.storage import convert_candle_cache_to_daily

logger = logging.getLogger(__name__)


class AggregateCandleView(ListAPIView):
    """Aggregate candle view."""

    queryset = Candle.objects.filter(is_active=True).select_related("symbol")
    filter_backends = (DjangoFilterBackend,)
    filterset_class = CandleFilter

    def get_params(self, request: Request) -> list[tuple]:
        """Get params."""
        serializer = TimeAgoWithRetrySerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data
        queryset = self.get_queryset()
        return [
            (
                candle,
                data["timestamp_from"],
                data["timestamp_to"],
                data["retry"],
            )
            for candle in self.filter_queryset(queryset)
        ]

    def get(self, request: Request, *args, **kwargs) -> Response:
        """Get data for each symbol."""
        for candle, timestamp_from, timestamp_to, retry in self.get_params(request):
            logger.info("{candle}: starting...".format(**{"candle": str(candle)}))
            aggregate_candles(candle, timestamp_from, timestamp_to, retry)
            convert_candle_cache_to_daily(candle)
        return Response({"ok": True})
