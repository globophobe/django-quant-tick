import logging

from django_filters.rest_framework import DjangoFilterBackend
from rest_framework.generics import ListAPIView
from rest_framework.request import Request
from rest_framework.response import Response

from quant_candles.controllers import aggregate_candles
from quant_candles.filters import CandleFilter
from quant_candles.models import Candle
from quant_candles.serializers import TimeAgoWithRetrySerializer
from quant_candles.storage import convert_candle_cache_to_daily

logger = logging.getLogger(__name__)


class AggregateCandleView(ListAPIView):
    queryset = Candle.objects.filter(is_active=True).prefetch_related("symbols")
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
