from django_filters import rest_framework as filters

from cryptofeed_werks.models import Symbol


class SymbolFilter(filters.FilterSet):
    symbol = filters.CharFilter(field_name="api_symbol")

    class Meta:
        model = Symbol
        fields = ("symbol",)
