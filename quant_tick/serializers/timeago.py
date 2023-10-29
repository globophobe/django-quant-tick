import pandas as pd
from rest_framework import serializers

from quant_tick.lib import get_current_time, get_min_time
from quant_tick.utils import gettext_lazy as _


class TimeAgoSerializer(serializers.Serializer):
    time_ago = serializers.CharField(required=False, default="1d")

    def validate_time_ago(self, value: str) -> pd.Timedelta:
        """Validate time ago."""
        try:
            return pd.Timedelta(value)
        except ValueError:
            raise serializers.ValidationError(_(f"Cannot parse {value}."))

    def validate(self, data: dict) -> dict:
        """Validate."""
        time_ago = data.pop("time_ago")
        now = get_current_time()
        timestamp_to = get_min_time(now, "1t")
        timestamp_from = timestamp_to - pd.Timedelta(time_ago)
        data["timestamp_to"] = timestamp_to
        data["timestamp_from"] = timestamp_from
        return data


class TimeAgoWithRetrySerializer(TimeAgoSerializer):
    retry = serializers.BooleanField(required=False, default=False)
