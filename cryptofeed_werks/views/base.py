from django.db.models import QuerySet
from django.http import Http404
from django_filters.rest_framework import DjangoFilterBackend
from rest_framework.generics import ListAPIView

from cryptofeed_werks.filters import SymbolFilter
from cryptofeed_werks.models import Symbol
from cryptofeed_werks.serializers import TradeParameterSerializer


class BaseSymbolView(ListAPIView):
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

    def get_command_kwargs(self, request):
        """Get params."""
        serializer = TradeParameterSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data
        return [
            {
                "symbol": symbol,
                "timestamp_from": data["timestamp_from"],
                "timestamp_to": data["timestamp_to"],
                "retry": data["retry"],
            }
            for symbol in self.get_queryset()
        ]
