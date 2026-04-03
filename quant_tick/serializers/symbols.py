from rest_framework import serializers

from quant_tick.models import Symbol


class SymbolSerializer(serializers.ModelSerializer):
    """Symbol serializer."""

    class Meta:
        model = Symbol
        fields = (
            "exchange",
            "symbol",
            "aggregate_trades",
            "significant_trade_filter",
        )
