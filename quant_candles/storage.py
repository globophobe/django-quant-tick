import datetime
import logging
import os
from typing import Optional

import pandas as pd

from quant_candles.constants import Frequency
from quant_candles.exchanges import candles_api
from quant_candles.lib import iter_timeframe, validate_data_frame
from quant_candles.models import Symbol, TradeData
from quant_candles.models.trades import upload_trade_data_to
from quant_candles.utils import gettext_lazy as _

logger = logging.getLogger(__name__)


def convert_trade_data_to_hourly(
    symbol: Symbol,
    timestamp_from: Optional[datetime.datetime] = None,
    timestamp_to: Optional[datetime.datetime] = None,
):
    """Convert trade data aggregated by minute to hourly."""
    queryset = TradeData.objects.filter(symbol=symbol).filter_by_timestamp(
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
        timestamps = [t.timestamp for t in minutes]
        values = set([timestamp.time().minute for timestamp in timestamps])
        is_complete = values == {i for i in range(Frequency.HOUR)}
        if is_complete:
            timestamp_from = timestamps[0]
            timestamp_to = timestamps[-1] + pd.Timedelta("1t")
            timestamps.sort()
            data_frames = {t: t.get_data_frame() for t in minutes if t.file_data.name}
            for t, data_frame in data_frames.items():
                data_frame["uid"] = ""
                # Set first index.
                uid = data_frame.columns.get_loc("uid")
                data_frame.iloc[:1, uid] = t.uid
            if len(data_frames):
                filtered = pd.concat(data_frames.values())
            else:
                filtered = pd.DataFrame([])
            candles = candles_api(symbol, timestamp_from, timestamp_to)
            validated = validate_data_frame(
                timestamp_from,
                timestamp_to,
                filtered,
                candles,
                symbol.should_aggregate_trades,
            )
            # Delete minutes
            pks = [aggregated_data.pk for aggregated_data in minutes]
            TradeData.objects.filter(pk__in=pks).delete()
            # Create hourly
            TradeData.write(symbol, timestamp_from, timestamp_to, filtered, validated)
            logging.info(
                _("Converted {timestamp_from} {timestamp_to} to hourly").format(
                    **{"timestamp_from": timestamp_from, "timestamp_to": timestamp_to}
                )
            )


def clean_trade_data_with_non_existing_files(
    symbol: Symbol,
    timestamp_from: Optional[datetime.datetime] = None,
    timestamp_to: Optional[datetime.datetime] = None,
) -> None:
    """Clean aggregated with non-existing files."""
    logging.info(_("Checking objects with non existent files"))

    queryset = (
        TradeData.objects.filter(symbol=symbol)
        .exclude(file_data="")
        .filter_by_timestamp(timestamp_from, timestamp_to)
    )
    count = 0
    deleted = 0
    total = queryset.count()
    for obj in queryset:
        count += 1
        if not obj.file_data.storage.exists(obj.file_data.name):
            obj.delete()
            deleted += 1
        logging.info(
            _("Checked {count}/{total} objects").format(
                **{"count": count, "total": total}
            )
        )

    logging.info(_("Deleted {deleted} objects").format(**{"deleted": deleted}))


def clean_unlinked_trade_data_files(
    symbol: Symbol,
    timestamp_from: Optional[datetime.datetime] = None,
    timestamp_to: Optional[datetime.datetime] = None,
) -> None:
    """Clean unlinked trade data files."""
    logging.info(_("Checking unlinked trade data files"))

    deleted = 0
    queryset = TradeData.objects.filter(symbol=symbol)
    first = queryset.first()
    last = queryset.last()
    if first and last:
        for daily_timestamp_from, daily_timestamp_to in iter_timeframe(
            first.timestamp, last.timestamp, value="1d"
        ):
            expected_files = [
                os.path.basename(obj.file_data.name)
                for obj in TradeData.objects.filter(symbol=symbol)
                .exclude(file_data="")
                .filter_by_timestamp(daily_timestamp_from, daily_timestamp_to)
            ]

            dummy = TradeData(symbol=symbol, timestamp=daily_timestamp_from)
            storage = dummy.file_data.storage
            directory, __ = os.path.split(upload_trade_data_to(dummy, "dummy.parquet"))
            __, filenames = storage.listdir(directory)

            should_delete = [
                filename for filename in filenames if filename not in expected_files
            ]
            for filename in should_delete:
                storage.delete(os.path.join(directory, filename))
                deleted += 1

            logging.info(
                _("Checked {date}").format(**{"date": daily_timestamp_from.date()})
            )

        logging.info(
            _("Deleted {deleted} unlinked files").format(**{"deleted": deleted})
        )
