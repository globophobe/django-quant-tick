from rest_framework import serializers

from quant_tick.models import Candle, Strategy


class CandleSerializer(serializers.ModelSerializer):
    """Candle serializer."""

    class Meta:
        model = Candle
        fields = ("code_name",)


class StrategySerializer(serializers.ModelSerializer):
    """Strategy serializer."""

    class Meta:
        model = Strategy
        fields = ("code_name",)


class CandleDataSerializer(serializers.Serializer):
    """Candle data serializer."""

    timestamp = serializers.DateTimeField()
    data = serializers.JSONField(source="json_data")
