import pandas as pd
from rest_framework import serializers

from quant_candles.constants import NUMERIC_PRECISION, NUMERIC_SCALE
from quant_candles.lib import get_current_time, get_min_time
from quant_candles.utils import gettext_lazy as _


def big_decimal() -> serializers.DecimalField:
    """Big decimal."""
    return serializers.DecimalField(
        max_digits=NUMERIC_PRECISION, decimal_places=NUMERIC_SCALE
    )


class BaseParameterSerializer(serializers.Serializer):
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
