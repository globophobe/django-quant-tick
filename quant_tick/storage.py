import datetime
import logging
import warnings
from pathlib import Path

import pandas as pd
from django.db import transaction
from django.db.models import Count, Q, QuerySet
from django.db.models.functions import TruncDate

from quant_tick.constants import FileData, Frequency
from quant_tick.lib import (
    combine_clustered_trades,
    get_existing,
    get_min_time,
    get_next_time,
    has_timestamps,
    is_decimal_close,
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


def convert_trade_data_to_hourly(
    symbol: Symbol, timestamp_from: datetime.datetime, timestamp_to: datetime.datetime
) -> None:
    """Convert trade data by minute, to hourly."""
    queryset = TradeData.objects.filter(symbol=symbol, frequency=Frequency.MINUTE)
    trade_data = queryset.filter(
        timestamp__gte=timestamp_from, timestamp__lte=timestamp_to
    )
    if trade_data.exists():
        first = trade_data.first()
        last = trade_data.last()
        min_timestamp_from = first.timestamp
        if timestamp_from < min_timestamp_from:
            timestamp_from = min_timestamp_from
        max_timestamp_to = last.timestamp + pd.Timedelta(f"{last.frequency}min")
        if timestamp_to > max_timestamp_to:
            timestamp_to = max_timestamp_to
        for daily_ts_from, daily_ts_to in iter_timeframe(
            timestamp_from, timestamp_to, value="1d", reverse=True
        ):
            for hourly_ts_from, hourly_ts_to in iter_timeframe(
                daily_ts_from, daily_ts_to, value="1h", reverse=True
            ):
                minute_trade_data = queryset.filter(
                    timestamp__gte=hourly_ts_from,
                    timestamp__lt=hourly_ts_to,
                    frequency=Frequency.MINUTE,
                )
                if minute_trade_data.count() == Frequency.HOUR:
                    convert_trade_data(minute_trade_data, hourly_ts_from, hourly_ts_to)
    logger.info("{symbol}: done".format(symbol=str(symbol)))


def convert_trade_data(
    trade_data: QuerySet,
    timestamp_from: datetime.datetime,
    timestamp_to: datetime.datetime,
) -> None:
    """Convert trade data."""
    delta = timestamp_to - timestamp_from
    frequency = delta.total_seconds() / 60
    assert frequency == Frequency.HOUR
    first = trade_data.first()

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
            # Ignore: The behavior of DataFrame concatenation with empty or
            # all-NA entries is deprecated.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                data_frame = pd.concat(data_frames.values())

            key = (
                "notional"
                if file_data in (FileData.RAW, FileData.AGGREGATED, FileData.CANDLE)
                else "totalNotional"
            )
            expected = sum([getattr(df, key).sum() for df in data_frames.values()])
            actual = data_frame[key].sum()
            assert is_decimal_close(expected, actual)

            if file_data == FileData.CLUSTERED:
                data_frame = combine_clustered_trades(data_frame)

            # First, delete minutes, as naming convention is same as hourly or daily.
            for t in trade_data:
                getattr(t, file_data).delete()

            # Next, create hourly or daily.
            if file_data != FileData.CANDLE:
                data_frame.reset_index(inplace=True)

            if len(data_frame):
                setattr(
                    new_trade_data,
                    file_data,
                    TradeData.prepare_data(data_frame),
                )
                if not new_trade_data.json_data:
                    candles = pd.DataFrame(
                        [t.json_data["candle"] for t in trade_data if t.json_data]
                    )
                    first_row = candles.iloc[0]
                    last_row = candles.iloc[-1]
                    candle = {
                        "timestamp": first_row.timestamp,
                        "open": first_row.open,
                        "high": candles.high.max(),
                        "low": candles.low.min(),
                        "close": last_row.close,
                        "volume": candles.volume.sum(),
                        "buyVolume": candles.buyVolume.sum(),
                        "notional": candles.notional.sum(),
                        "buyNotional": candles.buyNotional.sum(),
                        "ticks": candles.ticks.sum(),
                        "buyTicks": candles.buyTicks.sum(),
                    }
                    new_trade_data.json_data = {"candle": candle}
                new_trade_data.save()

    # Completely delete minutes.
    trade_data.delete()

    logging.info(
        _("Converted {timestamp_from} {timestamp_to} to {frequency}").format(
            **{
                "timestamp_from": timestamp_from,
                "timestamp_to": timestamp_to,
                "frequency": "hourly",
            }
        )
    )


def clean_trade_data_with_non_existing_files(
    symbol: Symbol, timestamp_from: datetime.datetime, timestamp_to: datetime.datetime
) -> None:
    """Clean trade data with non-existing files."""
    logging.info(_("Checking objects with non existent files"))

    fields = [FileData.CANDLE]
    for file_data, attr in {
        FileData.RAW: "save_raw",
        FileData.AGGREGATED: "save_aggregated",
        FileData.FILTERED: "save_filtered",
        FileData.CLUSTERED: "save_clustered",
    }.items():
        if getattr(symbol, attr):
            fields.append(file_data)

    exclude = Q()
    for f in fields:
        exclude |= Q(**{f: ""})

    trade_data = (
        TradeData.objects.filter(symbol=symbol)
        .exclude(exclude)
        .filter(
            timestamp__gte=timestamp_from,
            timestamp__lte=timestamp_to,
        )
        .only(*fields)
    )
    count = 0
    deleted = 0
    total = trade_data.count()
    for obj in trade_data:
        count += 1
        for field in fields:
            if getattr(obj, field):
                f = getattr(obj, field)
                if not f.storage.exists(f.name):
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

    fields = [FileData.CANDLE]
    for file_data, attr in {
        FileData.RAW: "save_raw",
        FileData.AGGREGATED: "save_aggregated",
        FileData.FILTERED: "save_filtered",
        FileData.CLUSTERED: "save_clustered",
    }.items():
        if getattr(symbol, attr):
            fields.append(file_data)

    exclude = Q()
    for f in fields:
        exclude |= Q(**{f: ""})

    trade_data = TradeData.objects.filter(symbol=symbol).exclude(exclude).only(*fields)
    t = trade_data.filter(
        timestamp__gte=timestamp_from,
        timestamp__lte=timestamp_to,
    )

    deleted = 0
    mapping = {
        k: v
        for k, v in {
            FileData.RAW: upload_raw_data_to,
            FileData.AGGREGATED: upload_aggregated_data_to,
            FileData.FILTERED: upload_filtered_data_to,
            FileData.CLUSTERED: upload_clustered_data_to,
            FileData.CANDLE: upload_candle_data_to,
        }.items()
        if k in fields
    }
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
    logger.info("{symbol}: done".format(symbol=str(symbol)))
