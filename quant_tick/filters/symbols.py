from django_filters import rest_framework as filters

from quant_tick.models import Symbol


class SymbolFilter(filters.FilterSet):
    class Meta:
        model = Symbol
        fields = ("exchange", "api_symbol")
