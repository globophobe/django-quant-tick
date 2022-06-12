import datetime
from typing import Optional

import pandas as pd
from django.db.models import Q

from cryptofeed_werks.exchanges import candles_api
from cryptofeed_werks.lib import validate_data_frame
from cryptofeed_werks.models import AggregatedTradeData, Symbol


def convert_minute_to_hourly(
    symbol: Symbol,
    timestamp_from: Optional[datetime.datetime] = None,
    timestamp_to: Optional[datetime.datetime] = None,
    verbose: bool = False,
):
    """Convert minute data to hourly data."""
    queryset = AggregatedTradeData.objects.filter(symbol=symbol)
    q = Q()
    if timestamp_from:
        q |= Q(timestamp__gte=timestamp_from)
    if timestamp_to:
        q |= Q(timestamp__lt=timestamp_to)
    queryset = queryset.filter(q)
    hours = {}
    for minute in queryset.filter(is_hourly=False):
        d = minute.timestamp.date()
        t = datetime.time(minute.timestamp.time().hour, 0)
        hour = datetime.datetime.combine(d, t).replace(tzinfo=datetime.timezone.utc)
        minutes = hours.setdefault(hour, [])
        minutes.append(minute)

    for hour, minutes in hours.items():
        timestamps = [aggregated.timestamp for aggregated in minutes]
        values = set([timestamp.time().minute for timestamp in timestamps])
        is_complete = values == {i for i in range(60)}
        if is_complete:
            timestamp_from = timestamps[0]
            timestamp_to = timestamps[-1] + pd.Timedelta("1t")
            timestamps.sort()
            data_frames = [
                aggregated.data_frame
                for aggregated in minutes
                if aggregated.data_frame is not None
            ]
            if len(data_frames):
                filtered = pd.concat(data_frames)
            else:
                filtered = pd.DataFrame([])
            candles = candles_api(symbol, timestamp_from, timestamp_to)
            validated = validate_data_frame(
                timestamp_from, timestamp_to, filtered, candles
            )
            # Delete minutes
            pks = [aggregated.pk for aggregated in minutes]
            AggregatedTradeData.objects.filter(pk__in=pks).delete()
            # Create hourly
            AggregatedTradeData.write(
                symbol,
                timestamp_from,
                timestamp_to,
                filtered,
                validated=validated,
            )
