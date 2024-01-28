from rest_framework import serializers

from quant_tick.models import Candle

from .symbols import SymbolSerializer


class CandleSerializer(serializers.ModelSerializer):
    """Candle serializer."""

    code_name = serializers.CharField()
    symbols = SymbolSerializer(many=True)

    class Meta:
        model = Candle
        fields = ("code_name", "symbols")


class CandleDataSerializer(serializers.Serializer):
    """Candle data serializer."""

    timestamp = serializers.DateTimeField()
    data = serializers.JSONField(source="json_data")
