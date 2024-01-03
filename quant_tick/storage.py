import datetime
import logging
from pathlib import Path

import pandas as pd
from django.db import transaction
from django.db.models import Count, QuerySet
from django.db.models.functions import TruncDate

from quant_tick.constants import FileData, Frequency
from quant_tick.lib import (
    get_existing,
    get_min_time,
    get_next_time,
    has_timestamps,
    iter_timeframe,
)
from quant_tick.models import Candle, CandleCache, Symbol, TradeData
from quant_tick.models.trades import (
    upload_aggregated_data_to,
    upload_candle_data_to,
    upload_clustered_data_to,
    upload_filtered_data_to,
    upload_raw_data_to,
)
from quant_tick.utils import gettext_lazy as _

logger = logging.getLogger(__name__)


def convert_candle_cache_to_daily(candle: Candle) -> None:
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
            timestamp__lt=last_daily_cache.timestamp, frequency__lt=Frequency.DAY
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
                    frequency__lt=Frequency.DAY,
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
                        _("Converted {date} to daily").format(
                            **{"date": daily_ts_from.date()}
                        )
                    )


def convert_trade_data_to_daily(
    symbol: Symbol, timestamp_from: datetime.datetime, timestamp_to: datetime.datetime
) -> None:
    """Convert trade data by minute, or hourly, to daily.

    * Not necessary to convert in order.
    """
    queryset = TradeData.objects.filter(
        symbol=symbol, frequency__in=(Frequency.MINUTE, Frequency.HOUR)
    )
    trade_data = queryset.filter(
        timestamp__gte=timestamp_from, timestamp__lte=timestamp_to
    )
    if trade_data.exists():
        first = trade_data.first()
        last = trade_data.last()
        min_timestamp_from = first.timestamp
        if timestamp_from < min_timestamp_from:
            timestamp_from = min_timestamp_from
        max_timestamp_to = last.timestamp + pd.Timedelta(f"{last.frequency}t")
        if timestamp_to > max_timestamp_to:
            timestamp_to = max_timestamp_to
        for daily_ts_from, daily_ts_to in iter_timeframe(
            timestamp_from, timestamp_to, value="1d", reverse=True
        ):
            trade_data = queryset.filter(
                timestamp__gte=daily_ts_from, timestamp__lt=daily_ts_to
            )
            minutes = get_existing(trade_data.values("timestamp", "frequency"))
            # iter_timeframe may return timestamps not equal to a day.
            if len(minutes) == Frequency.DAY:
                convert_trade_data(trade_data, daily_ts_from, daily_ts_to)
            else:
                for hourly_ts_from, hourly_ts_to in iter_timeframe(
                    timestamp_from, timestamp_to, value="1h", reverse=True
                ):
                    t_data = queryset.filter(
                        timestamp__gte=hourly_ts_from,
                        timestamp__lt=hourly_ts_to,
                        frequency=Frequency.MINUTE,
                    )
                    if t_data.count() == Frequency.HOUR:
                        convert_trade_data(t_data, hourly_ts_from, hourly_ts_to)


def convert_trade_data(
    trade_data: QuerySet,
    timestamp_from: datetime.datetime,
    timestamp_to: datetime.datetime,
) -> None:
    """Convert trade data."""
    delta = timestamp_to - timestamp_from
    frequency = delta.total_seconds() / 60
    assert frequency in (Frequency.HOUR, Frequency.DAY)
    first = trade_data.first()
    with transaction.atomic():
        new_trade_data = TradeData.objects.create(
            symbol=first.symbol,
            timestamp=first.timestamp,
            uid=first.uid,
            frequency=frequency,
            ok=all(trade_data.values_list("ok", flat=True)),
        )
        for file_data in FileData:
            data_frames = {
                t: t.get_data_frame(file_data)
                for t in trade_data
                if getattr(t, file_data).name
            }

            if len(data_frames):
                data_frame = pd.concat(data_frames.values())
            else:
                data_frame = pd.DataFrame([])

            # First, delete minutes, as naming convention is same as hourly.
            for t in trade_data:
                getattr(t, file_data).delete()
            # Next, create hourly
            setattr(
                new_trade_data,
                file_data,
                TradeData.prepare_data(data_frame),
            )
            new_trade_data.save()

    logging.info(
        _("Converted {timestamp_from} {timestamp_to} to {frequency}").format(
            **{
                "timestamp_from": timestamp_from,
                "timestamp_to": timestamp_to,
                "frequency": frequency,
            }
        )
    )


def clean_trade_data_with_non_existing_files(
    symbol: Symbol, timestamp_from: datetime.datetime, timestamp_to: datetime.datetime
) -> None:
    """Clean trade data with non-existing files."""
    logging.info(_("Checking objects with non existent files"))

    mapping = dict(
        zip(
            FileData,
            [
                "save_raw",
                "save_aggregated",
                "save_filtered",
                "save_clustered",
                "save_candles",
            ],
            strict=True,
        )
    )

    trade_data = (
        TradeData.objects.filter(symbol=symbol)
        .exclude(file_data="")
        .filter(
            timestamp__gte=timestamp_from,
            timestamp__lte=timestamp_to,
        )
        .only(*list(mapping.keys()) + list(mapping.values()))
    )
    count = 0
    deleted = 0
    total = trade_data.count()
    for obj in trade_data:
        count += 1
        for file_data, attr in mapping.items():
            if getattr(obj, attr):
                name = getattr(obj, file_data).name
                if not obj.file_data.storage.exists(name):
                    obj.delete()
                    break
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

    mapping = dict(
        zip(
            FileData,
            [
                upload_raw_data_to,
                upload_aggregated_data_to,
                upload_filtered_data_to,
                upload_clustered_data_to,
                upload_candle_data_to,
            ],
            strict=True,
        )
    )

    deleted = 0
    trade_data = (
        TradeData.objects.filter(symbol=symbol).exclude(file_data="").only(*FileData)
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
            for file_data, upload_to in mapping.items():
                expected_files = [
                    Path(getattr(obj, file_data).name).name
                    for obj in trade_data.filter(
                        timestamp__gte=daily_timestamp_from,
                        timestamp__lte=daily_timestamp_to,
                    )
                ]

                dummy = TradeData(symbol=symbol, timestamp=daily_timestamp_from)
                storage = getattr(dummy, file_data).storage
                directory = Path(upload_to(dummy, "dummy.parquet")).stem
                __, filenames = storage.listdir(directory)

                should_delete = [
                    filename for filename in filenames if filename not in expected_files
                ]
                for filename in should_delete:
                    storage.delete(Path(directory) / filename)
                    deleted += 1

            logging.info(
                _("Checked {date}").format(**{"date": daily_timestamp_from.date()})
            )

        logging.info(
            _("Deleted {deleted} unlinked files").format(**{"deleted": deleted})
        )
