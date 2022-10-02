import datetime
import logging
from typing import Optional

import pandas as pd

from quant_werks.constants import Frequency
from quant_werks.exchanges import candles_api
from quant_werks.lib import validate_data_frame
from quant_werks.models import AggregatedTradeData, Symbol
from quant_werks.utils import gettext_lazy as _

logger = logging.getLogger(__name__)


def convert_aggregated_to_hourly(
    symbol: Symbol,
    timestamp_from: Optional[datetime.datetime] = None,
    timestamp_to: Optional[datetime.datetime] = None,
    verbose: bool = False,
):
    """Convert trade data aggregated by minute to hourly."""
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
        timestamps = [aggregated_data.timestamp for aggregated_data in minutes]
        values = set([timestamp.time().minute for timestamp in timestamps])
        is_complete = values == {i for i in range(Frequency.HOUR)}
        if is_complete:
            timestamp_from = timestamps[0]
            timestamp_to = timestamps[-1] + pd.Timedelta("1t")
            timestamps.sort()
            data_frames = {
                aggregated_data: aggregated_data.get_data_frame()
                for aggregated_data in minutes
                if aggregated_data.file_data.name
            }
            for aggregated_data, data_frame in data_frames.items():
                data_frame["uid"] = ""
                # Set first index.
                uid = data_frame.columns.get_loc("uid")
                data_frame.iloc[:1, uid] = aggregated_data.uid
            if len(data_frames):
                filtered = pd.concat(data_frames.values())
            else:
                filtered = pd.DataFrame([])
            candles = candles_api(symbol, timestamp_from, timestamp_to)
            validated = validate_data_frame(
                timestamp_from, timestamp_to, filtered, candles
            )
            # Delete minutes
            pks = [aggregated_data.pk for aggregated_data in minutes]
            AggregatedTradeData.objects.filter(pk__in=pks).delete()
            # Create hourly
            AggregatedTradeData.write(
                symbol, timestamp_from, timestamp_to, filtered, validated
            )


def clean_aggregated_storage(
    symbol: Symbol,
    timestamp_from: Optional[datetime.datetime] = None,
    timestamp_to: Optional[datetime.datetime] = None,
):
    """Clean aggregated storage."""
    queryset = AggregatedTradeData.objects.filter(symbol=symbol).filter_by_timestamp(
        timestamp_from, timestamp_to
    )
    count = 0
    deleted = 0
    total = queryset.count()
    for obj in queryset:
        count += 1
        if obj.data.name and not obj.data.storage.exists(obj.data.name):
            obj.delete()
            deleted += 1
            logging.info(
                _("Checked {count}/{total} objects").format(
                    **{"count": count, "total": total}
                )
            )
    logging.info(_("Deleted {deleted} objects").format(**{"deleted": deleted}))
