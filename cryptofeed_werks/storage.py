import datetime
import logging
from typing import Optional

import pandas as pd

from cryptofeed_werks.constants import Frequency
from cryptofeed_werks.exchanges import candles_api
from cryptofeed_werks.lib import validate_data_frame
from cryptofeed_werks.models import AggregatedTradeData, Symbol
from cryptofeed_werks.utils import gettext_lazy as _

logger = logging.getLogger(__name__)


def convert_aggregated_to_hourly(
    symbol: Symbol,
    timestamp_from: Optional[datetime.datetime] = None,
    timestamp_to: Optional[datetime.datetime] = None,
    verbose: bool = False,
):
    """Convert trade data aggregated by minute to hourly, to reduce file operations."""
    queryset = AggregatedTradeData.objects.filter(symbol=symbol).filter_by_timestamp(
        timestamp_from, timestamp_to
    )
    hours = {}
    for minute in queryset.filter(frequency=Frequency.MINUTE):
        d = minute.timestamp.date()
        t = datetime.time(minute.timestamp.time().hour, 0)
        hour = datetime.datetime.combine(d, t).replace(tzinfo=datetime.timezone.utc)
        minutes = hours.setdefault(hour, [])
        minutes.append(minute)

    for hour, minutes in hours.items():
        timestamps = [aggregated.timestamp for aggregated in minutes]
        values = set([timestamp.time().minute for timestamp in timestamps])
        is_complete = values == {i for i in range(Frequency.HOUR)}
        if is_complete:
            timestamp_from = timestamps[0]
            timestamp_to = timestamps[-1] + pd.Timedelta("1t")
            timestamps.sort()
            data_frames = [
                aggregated.get_data_frame()
                for aggregated in minutes
                if aggregated.data.name
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


def clean_aggregated_storage(
    symbol: Symbol,
    timestamp_from: Optional[datetime.datetime] = None,
    timestamp_to: Optional[datetime.datetime] = None,
    verbosity: int = 0,
):
    """Clean aggregated storage."""
    queryset = AggregatedTradeData.objects.filter(symbol=symbol).filter_by_timestamp(
        timestamp_from, timestamp_to
    )
    count = 0
    deleted = 0
    total = queryset.count()
    for obj in queryset:
        if obj.data.name and not obj.data.storage.exists(obj.data.name):
            obj.delete()
            deleted += 1
        if verbosity:
            count += 1
            print(
                _("Checked {count}/{total} objects").format(
                    **{"count": count, "total": total}
                )
            )
    if verbosity:
        print(_("Deleted {deleted} objects").format(**{"deleted": deleted}))
