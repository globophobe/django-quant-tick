from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.views import APIView

from cryptofeed_werks.exchanges import api
from cryptofeed_werks.serializers import TradeDataParameterSerializer


class TradeDataAPIView(APIView):
    """Trade data API view."""

    def get(self, request: Request) -> Response:
        """Get."""
        serializer = TradeDataParameterSerializer(data=request.query_params)
        serializer.is_valid(raise_exception=True)
        api(**serializer.validated_data)
        return Response({"ok": True})
