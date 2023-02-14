from rest_framework import serializers

from quant_candles.models import Symbol


class SymbolSerializer(serializers.ModelSerializer):
    global_symbol = serializers.SerializerMethodField()

    def get_global_symbol(self, symbol: Symbol) -> str:
        """Get global symbol."""
        return symbol.global_symbol.name

    class Meta:
        model = Symbol
        fields = (
            "global_symbol",
            "exchange",
            "symbol",
            "significant_trade_filter",
            "should_aggregate_trades",
        )
