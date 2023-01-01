from django_filters import rest_framework as filters

from quant_candles.models import Symbol


class SymbolFilter(filters.FilterSet):
    symbol = filters.CharFilter(field_name="api_symbol")
    aggregate = filters.BooleanFilter(field_name="should_aggregate_trades")
    filter = filters.NumberFilter(field_name="significant_trade_filter")

    class Meta:
        model = Symbol
        fields = ("symbol", "aggregate", "filter")
