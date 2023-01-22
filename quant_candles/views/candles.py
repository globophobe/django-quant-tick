from rest_framework.generics import ListAPIView
from rest_framework.request import Request
from rest_framework.response import Response

from quant_candles.models import Candle
from quant_candles.serializers import CandleParameterSerializer, CandleSerializer


class CandleView(ListAPIView):
    queryset = Candle.objects.all()
    lookup_field = "code_name"

    def get(self, request: Request, *args, **kwargs) -> Response:
        """Get candles."""
        serializer = CandleParameterSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        params = serializer.validated_data
        candle = self.get_object()
        data = candle.get_data(
            params["timestamp_from"],
            params["timestamp_to"],
            limit=params["limit"],
        )
        return Response(CandleSerializer(data, many=True).data)
