import pandas as pd
from django.db.models import Q, QuerySet
from django.http import Http404
from rest_framework.generics import ListAPIView
from rest_framework.request import Request
from rest_framework.response import Response

from quant_candles.exchanges import api
from quant_candles.models import TradeData
from quant_candles.serializers import OHLCSerializer, TimeFrameSerializer


class CandleView(ListAPIView):
    queryset = TradeData.objects.all()

    def get_queryset(self) -> QuerySet:
        """Get queryset."""
        queryset = super().get_queryset()
        queryset = queryset.filter(
            symbol__exchange=self.kwargs["exchange"],
            symbol__api_symbol=self.kwargs["symbol"],
        )
        serializer = TimeFrameSerializer(data=self.request.query_params)
        serializer.is_valid()
        data = serializer.validated_data
        timestamp_from = data["timestamp_from"] - pd.Timedelta("1t")
        timestamp_to = data["timestamp_to"] + pd.Timedelta("1t")
        queryset = queryset.filter(
            Q(timestamp__gte=timestamp_from) & Q(timestamp__lt=timestamp_to)
        )
        if not queryset.exists():
            raise Http404
        return queryset.order_by("-timestamp")

    def get(self, request: Request, *args, **kwargs) -> Response:
        """Get hourly candles for symbol."""
        serializer_class = OHLCSerializer
        for k in self.get_command_kwargs(request):
            api(**k)
        return Response({"ok": True})
