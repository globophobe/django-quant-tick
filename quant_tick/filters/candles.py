from django_filters import rest_framework as filters

from quant_tick.models import Candle


class CandleFilter(filters.FilterSet):
    class Meta:
        model = Candle
        fields = ("code_name",)
