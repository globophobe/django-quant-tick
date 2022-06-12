import pandas as pd
from rest_framework import serializers

from cryptofeed_werks.lib import get_current_time, get_min_time


class BaseParameterSerializer(serializers.Serializer):
    time_ago = serializers.CharField(required=False)

    def validate(self, data: dict) -> dict:
        """Validate."""
        time_ago = data.pop("time_ago")
        now = get_current_time()
        timestamp_to = get_min_time(now, "1t")
        timestamp_from = timestamp_to - pd.Timedelta(time_ago)
        data["timestamp_to"] = timestamp_to
        data["timestamp_from"] = timestamp_from
        return data
