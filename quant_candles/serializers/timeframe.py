from datetime import time

from rest_framework import serializers

from quant_candles.lib import parse_period_from_to


class TimeFrameSerializer(serializers.Serializer):
    date_from = serializers.DateField(required=False)
    time_from = serializers.TimeField(required=False)
    date_to = serializers.DateField(required=False)
    time_to = serializers.TimeField(required=False)

    def validate(self, data: dict) -> dict:
        """Validate parameters."""
        date_from = data.pop("date_from")
        time_from = data.pop("time_from")
        date_to = data.pop("date_to")
        time_to = data.pop("time_to")
        if date_from is not None:
            date_from = date_from.isoformat()
        if time_from is not None:
            time_from = time(time_from.hour, time_from.minute).isoformat()
        if date_to is not None:
            date_to = date_to.isoformat()
        if time_to is not None:
            time_to = time(time_to.hour, time_to.minute).isoformat()
        timestamp_from, timestamp_to = parse_period_from_to(
            date_from, time_from, date_to, time_to
        )
        data["timestamp_from"] = timestamp_from
        data["timestamp_to"] = timestamp_to
        return data
