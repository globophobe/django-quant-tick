from django.core import management
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from quant_candles.serializers import QuantCandleParameterSerializer


class QuantCandleView(APIView):
    def post(self, request: Request, *args, **kwargs) -> Response:
        """Process each symbol."""
        serializer = QuantCandleParameterSerializer(data=request.data)
        serializer.is_valid()
        management.call_command(
            "quant_candles",
            "--exchange",
            " ".join(serializer.validated_data["exchange"]),
            "--time-ago",
            request.data["time_ago"],
        )
        return Response({"ok": True})
