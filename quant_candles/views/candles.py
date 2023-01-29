from rest_framework.generics import ListAPIView
from rest_framework.request import Request
from rest_framework.response import Response

from quant_candles.models import Candle
from quant_candles.serializers import DataSerializer, LimitSerializer


class CandleView(ListAPIView):
    queryset = Candle.objects.all()
    lookup_field = "code_name"

    def get(self, request: Request, *args, **kwargs) -> Response:
        """Get candles."""
        serializer = LimitSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        params = serializer.validated_data
        candle = self.get_object()
        data = candle.get_data(
            params["timestamp_from"],
            params["timestamp_to"],
            limit=params["limit"],
        )
        return Response(DataSerializer(data, many=True).data)


class CandleTradeDataSummaryView(ListAPIView):
    queryset = Candle.objects.prefetch_related("symbols")
    lookup_field = "code_name"

    def get(self, request: Request, *args, **kwargs) -> Response:
        """Get candle trade data summary."""
        serializer = LimitSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        params = serializer.validated_data
        candle = self.get_object()
        data_frame = candle.get_trade_summary_data(
            params["timestamp_from"],
            params["timestamp_to"],
            limit=params["limit"],
        )
        data = data_frame.drop(columns=["index"]).todict()
        for index, d in enumerate(data):
            timestamp = d.pop["timestamp"]
            data[index] = {"timestamp": timestamp, "data": d}
        return Response(DataSerializer(data, many=True).data)
