from django_filters import rest_framework as filters

from quant_candles.models import Symbol


class CandleFilter(filters.FilterSet):
    name = filters.CharFilter(field_name="code_name")

    class Meta:
        model = Symbol
        fields = ("name",)
