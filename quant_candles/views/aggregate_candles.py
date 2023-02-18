from django_filters.rest_framework import DjangoFilterBackend
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from quant_candles.controllers import aggregate_candles
from quant_candles.filters import CandleFilter
from quant_candles.models import Candle
from quant_candles.serializers import TimeAgoWithRetrySerializer


class AggregateCandleView(APIView):
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
        for candle, timestamp_from, timestamp_to, retry in self.get_command_kwargs(
            request
        ):
            aggregate_candles(candle, timestamp_from, timestamp_to, retry)
        return Response({"ok": True})
