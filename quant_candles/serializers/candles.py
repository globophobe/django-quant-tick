from rest_framework import serializers

from .timeframe import TimeFrameSerializer


class CandleParameterSerializer(TimeFrameSerializer):
    limit = serializers.IntegerField(
        required=False, min_value=1, max_value=1000, default=1000
    )


class CandleSerializer(serializers.Serializer):
    timestamp = serializers.DateTimeField()
    data = serializers.JSONField(source="json_data")
