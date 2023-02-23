import datetime
import logging
import os

import pandas as pd
from django.db import transaction
from django.db.models import Count
from django.db.models.functions import TruncDate

from quant_candles.constants import Frequency
from quant_candles.exchanges import candles_api
from quant_candles.lib import (
    get_existing,
    get_min_time,
    get_next_time,
    has_timestamps,
    iter_timeframe,
    validate_data_frame,
)
from quant_candles.models import Candle, CandleCache, Symbol, TradeData
from quant_candles.models.trades import upload_trade_data_to
from quant_candles.utils import gettext_lazy as _

logger = logging.getLogger(__name__)


def convert_candle_cache_to_daily(candle: Candle):
    """Convert candle cache, by minute or hour, to daily.

    * Convert, from past to present, in order.
    """
    candle_cache = CandleCache.objects.filter(candle=candle)
    last_daily_cache = (
        candle_cache.filter(frequency=Frequency.DAY)
        .only("timestamp")
        .order_by("-timestamp")
        .first()
    )
    if last_daily_cache:
        hourly_or_minute_cache = candle_cache.filter(
            timestamp__lt=last_daily_cache.timestamp,
            frequency__in=(Frequency.HOUR, Frequency.MINUTE),
        )
        unique_dates = (
            hourly_or_minute_cache.annotate(date=TruncDate("timestamp"))
            .values("date")
            .annotate(unique=Count("date"))
        )
        if unique_dates.count() <= 1:
            timestamp_from = last_daily_cache.timestamp
        else:
            timestamp_from = (
                hourly_or_minute_cache.only("timestamp").first().timestamp_from
            )
    else:
        any_cache = candle_cache.only("timestamp").first()
        if any_cache:
            timestamp_from = any_cache.timestamp
        else:
            timestamp_from = None
    if timestamp_from:
        timestamp_to = candle_cache.only("timestamp").last().timestamp
        for daily_ts_from, daily_ts_to in iter_timeframe(
            get_min_time(timestamp_from, value="1d"),
            get_next_time(timestamp_to, value="1d"),
            value="1d",
        ):
            delta = daily_ts_to - daily_ts_from
            total_minutes = delta.total_seconds() / Frequency.HOUR
            if total_minutes == Frequency.DAY:
                target_cache = CandleCache.objects.filter(
                    candle=candle,
                    timestamp__gte=daily_ts_from,
                    timestamp__lt=daily_ts_to,
                    frequency__in=(Frequency.HOUR, Frequency.MINUTE),
                )
                existing = get_existing(target_cache.values("timestamp", "frequency"))
                if has_timestamps(daily_ts_from, daily_ts_to, existing):
                    with transaction.atomic():
                        daily_cache, created = CandleCache.objects.get_or_create(
                            candle=candle,
                            timestamp=daily_ts_from,
                            frequency=Frequency.DAY,
                        )
                        daily_cache.json_data = (
                            target_cache.order_by("-timestamp").first().json_data
                        )
                        daily_cache.save()
                        target_cache.delete()
                    logging.info(
                        _("Converted {date_from} {date_to} to daily").format(
                            **{
                                "date_from": daily_ts_from.date(),
                                "date_to": daily_ts_to.date(),
                            }
                        )
                    )


def convert_trade_data_to_hourly(
    symbol: Symbol, timestamp_from: datetime.datetime, timestamp_to: datetime.datetime
):
    """Convert trade data, by minute, to hourly."""
    trade_data = TradeData.objects.filter(
        symbol=symbol,
        frequency=Frequency.MINUTE,
    )
    t = trade_data.filter(timestamp__gte=timestamp_from, timestamp__lte=timestamp_to)
    if t.exists():
        first = t.first()
        last = t.last()
        min_timestamp_from = first.timestamp
        if timestamp_from < min_timestamp_from:
            timestamp_from = min_timestamp_from
        max_timestamp_to = last.timestamp + pd.Timedelta(f"{last.frequency}t")
        if timestamp_to > max_timestamp_to:
            timestamp_to = max_timestamp_to
        for daily_ts_from, daily_ts_to in iter_timeframe(
            timestamp_from, timestamp_to, value="1d", reverse=True
        ):
            for hourly_ts_from, hourly_ts_to in iter_timeframe(
                daily_ts_from, daily_ts_to, value="1h", reverse=True
            ):
                delta = hourly_ts_to - hourly_ts_from
                total_minutes = delta.total_seconds() / Frequency.HOUR
                if total_minutes == Frequency.HOUR.value:
                    minutes = trade_data.filter(
                        timestamp__gte=hourly_ts_from, timestamp__lt=hourly_ts_to
                    )
                    if minutes.count() == Frequency.HOUR:
                        data_frames = {
                            t: t.get_data_frame() for t in minutes if t.file_data.name
                        }
                        for t, data_frame in data_frames.items():
                            data_frame["uid"] = ""
                            # Set first index.
                            uid = data_frame.columns.get_loc("uid")
                            data_frame.iloc[:1, uid] = t.uid
                        if len(data_frames):
                            filtered = pd.concat(data_frames.values())
                        else:
                            filtered = pd.DataFrame([])
                        candles = candles_api(symbol, hourly_ts_from, hourly_ts_to)
                        validated = validate_data_frame(
                            hourly_ts_from,
                            hourly_ts_to,
                            filtered,
                            candles,
                            symbol.should_aggregate_trades,
                        )
                        # First, delete minutes, as naming convention is same as hourly.
                        minutes.delete()
                        # Next, create hourly
                        TradeData.write(
                            symbol,
                            hourly_ts_from,
                            hourly_ts_to,
                            filtered,
                            validated,
                        )
                        logging.info(
                            _(
                                "Converted {timestamp_from} {timestamp_to} to hourly"
                            ).format(
                                **{
                                    "timestamp_from": hourly_ts_from,
                                    "timestamp_to": hourly_ts_to,
                                }
                            )
                        )


def clean_trade_data_with_non_existing_files(
    symbol: Symbol, timestamp_from: datetime.datetime, timestamp_to: datetime.datetime
) -> None:
    """Clean aggregated with non-existing files."""
    logging.info(_("Checking objects with non existent files"))

    trade_data = (
        TradeData.objects.filter(symbol=symbol)
        .exclude(file_data="")
        .filter(
            timestamp__gte=timestamp_from,
            timestamp__lte=timestamp_to,
        )
        .only("file_data")
    )
    count = 0
    deleted = 0
    total = trade_data.count()
    for obj in trade_data:
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
    symbol: Symbol, timestamp_from: datetime.datetime, timestamp_to: datetime.datetime
) -> None:
    """Clean unlinked trade data files."""
    logging.info(_("Checking unlinked trade data files"))

    deleted = 0
    trade_data = (
        TradeData.objects.filter(symbol=symbol).exclude(file_data="").only("file_data")
    )
    t = trade_data.filter(
        timestamp__gte=timestamp_from,
        timestamp__lte=timestamp_to,
    )
    if t.exists():
        min_timestamp_from = t.first().timestamp
        if timestamp_from < min_timestamp_from:
            timestamp_from = min_timestamp_from
        max_timestamp_to = t.last().timestamp
        if timestamp_to > max_timestamp_to:
            timestamp_to = max_timestamp_to
        for daily_timestamp_from, daily_timestamp_to in iter_timeframe(
            timestamp_from, timestamp_to, value="1d"
        ):
            expected_files = [
                os.path.basename(obj.file_data.name)
                for obj in trade_data.filter(
                    timestamp__gte=daily_timestamp_from,
                    timestamp__lte=daily_timestamp_to,
                )
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
