from django.db.models import QuerySet
from django.http import Http404
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


class TradeDataView(ListAPIView):
    queryset = Symbol.objects.all()
    filter_backends = (DjangoFilterBackend,)
    filterset_class = SymbolFilter

    def get_queryset(self) -> QuerySet:
        """Get queryset."""
        queryset = super().get_queryset()
        queryset = queryset.filter(exchange=self.kwargs["exchange"])
        queryset = self.filter_queryset(queryset)
        if not queryset.exists():
            raise Http404
        return queryset

    def get_params(self, request: Request) -> list[tuple]:
        """Get params."""
        serializer = TimeAgoWithRetrySerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data
        return [
            (
                symbol,
                data["timestamp_from"],
                data["timestamp_to"],
                data["retry"],
            )
            for symbol in self.get_queryset()
        ]

    def get(self, request: Request, *args, **kwargs) -> Response:
        """Get data for each symbol."""
        for symbol, timestamp_from, timestamp_to, retry in self.get_params(request):
            api(symbol, timestamp_from, timestamp_to, retry)
            convert_trade_data_to_hourly(symbol, timestamp_from, timestamp_to)
            aggregate_trade_summary(symbol, timestamp_from, timestamp_to, retry)
        return Response({"ok": True})
