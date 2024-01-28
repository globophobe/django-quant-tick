from django_filters.rest_framework import DjangoFilterBackend
from rest_framework.generics import ListAPIView, RetrieveAPIView
from rest_framework.request import Request
from rest_framework.response import Response

from quant_tick.filters import CandleFilter
from quant_tick.models import Candle
from quant_tick.serializers import (
    CandleDataSerializer,
    CandleSerializer,
    TimeFrameWithLimitSerializer,
)


class CandleView(ListAPIView):
    """Candle view."""

    queryset = Candle.objects.prefetch_related("symbols__global_symbol")
    filterset_class = CandleFilter
    filter_backends = (DjangoFilterBackend,)
    serializer_class = CandleSerializer


class CandleDataView(RetrieveAPIView):
    """Candle data view."""

    queryset = Candle.objects.all()
    lookup_field = "code_name"

    def get(self, request: Request, *args, **kwargs) -> Response:
        """Get candles."""
        serializer = TimeFrameWithLimitSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        params = serializer.validated_data
        candle = self.get_object()
        data = candle.get_data(
            params["timestamp_from"], params["timestamp_to"], limit=params["limit"]
        )
        return Response(CandleDataSerializer(data, many=True).data)
