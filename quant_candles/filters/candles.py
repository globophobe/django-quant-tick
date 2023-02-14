from django_filters import rest_framework as filters

from quant_candles.models import Candle, GlobalSymbol, Symbol


class CandleFilter(filters.FilterSet):
    global_symbol = filters.ModelMultipleChoiceFilter(
        field_name="symbol__global_symbol", queryset=GlobalSymbol.objects.all()
    )
    symbol = filters.ModelMultipleChoiceFilter(
        field_name="symbol", queryset=Symbol.objects.all()
    )

    class Meta:
        model = Candle
        fields = ("global_symbol", "symbol", "code_name")
