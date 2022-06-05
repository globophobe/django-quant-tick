from datetime import date, datetime, time, timezone
from typing import Optional

import pandas as pd
from rest_framework import serializers

from cryptofeed_werks.constants import Exchange
from cryptofeed_werks.lib import get_current_time, get_min_time
from cryptofeed_werks.models import Symbol


class TradeDataParameterSerializer(serializers.Serializer):
    """Trade data parameter serializer."""

    exchange = serializers.ChoiceField(choices=Exchange.choices)
    symbol = serializers.CharField()
    date_to = serializers.DateField(required=False)
    time_to = serializers.TimeField(required=False, default=time.min)
    date_from = serializers.DateField(required=False, default=date(2010, 1, 1))
    time_from = serializers.TimeField(required=False, default=time.min)
    retry = serializers.BooleanField(required=False, default=False)

    def __init__(self, *args, **kwargs) -> None:
        """Initialize."""
        super().__init__(*args, **kwargs)
        self.now = get_current_time()
        self.tomorrow = self.now.date() + pd.Timedelta("1d")

    def validate_date_to(self, date_to: Optional[date]) -> date:
        """Validate date to."""
        return date_to or self.tomorrow

    def validate(self, data: dict) -> dict:
        d = {}
        exchange = data["exchange"]
        symbol = data["symbol"]
        try:
            d["symbol"] = Symbol.objects.get(
                exchange=data["exchange"], api_symbol=["symbol"]
            )
        except Symbol.DoesNotExist:
            raise serializers.ValidationError(f"{exchange} {symbol} not registered.")
        else:
            date_from = data["date_from"]
            time_from = data["time_from"]
            date_to = data["date_to"]
            time_to = data["time_to"]
            # UTC, please
            timestamp_from = datetime.combine(date_from, time_from).replace(
                tzinfo=timezone.utc
            )
            timestamp_to = datetime.combine(date_to, time_to).replace(
                tzinfo=timezone.utc
            )
            # Sane defaults
            d["timestamp_to"] = (
                get_min_time(self.now, "1t")
                if timestamp_to >= self.now
                else timestamp_to
            )
            d["timestamp_from"] = (
                timestamp_to if timestamp_from > timestamp_to else timestamp_from
            )
            return d
