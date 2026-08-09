import datetime
import logging
import warnings
from collections.abc import Callable
from pathlib import Path

import pandas as pd
from django.db import transaction
from django.db.models import Count, Q, QuerySet
from django.db.models.functions import TruncDate, TruncHour
from django.utils.translation import gettext_lazy as _

from quant_tick.constants import FileData, Frequency
from quant_tick.lib import (
    aggregate_candle,
    get_existing,
    get_min_time,
    get_next_time,
    has_timestamps,
    is_decimal_close,
    iter_timeframe,
    merge_cache,
)
from quant_tick.models import Candle, CandleCache, Symbol, TradeData
from quant_tick.models.trades import (
    upload_aggregated_data_to,
    upload_filtered_data_to,
    upload_raw_data_to,
)

logger = logging.getLogger(__name__)

COMPACT_RECENT_DELAY = pd.Timedelta("2h")


def get_compact_max_timestamp_to(timestamp_to: datetime.datetime) -> datetime.datetime:
    return get_min_time(timestamp_to - COMPACT_RECENT_DELAY, "1h")


def convert_candle_cache_to_daily(
    candle: Candle,
    assert_lease_owned: Callable[[], None] | None = None,
) -> None:
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
            timestamp_from = hourly_or_minute_cache.only("timestamp").first().timestamp
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
                        if assert_lease_owned is not None:
                            assert_lease_owned()
                        daily_cache, _created = CandleCache.objects.get_or_create(
                            candle=candle,
                            timestamp=daily_ts_from,
                            frequency=Frequency.DAY,
                        )
                        daily_cache.json_data = (
                            target_cache.order_by("-timestamp").first().json_data
                        )
                        daily_cache.save()
                        target_cache.delete()
                    logger.info(
                        _("Converted {date} to daily").format(date=daily_ts_from.date())
                    )


def convert_trade_data_to_daily(
    symbol: Symbol,
    timestamp_from: datetime.datetime,
    timestamp_to: datetime.datetime,
    assert_lease_owned: Callable[[], None] | None = None,
) -> None:
    """Convert trade data by minute, or hourly, to daily.

    * Convert any order.
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
        timestamp_from = max(timestamp_from, min_timestamp_from)
        max_timestamp_to = last.timestamp + pd.Timedelta(f"{last.frequency}min")
        timestamp_to = min(timestamp_to, max_timestamp_to)
        for daily_ts_from, daily_ts_to in iter_timeframe(
            timestamp_from, timestamp_to, value="1d", reverse=True
        ):
            # First, convert any complete hours of minute data to hourly.
            minute_trade_data = queryset.filter(
                timestamp__gte=daily_ts_from,
                timestamp__lt=daily_ts_to,
                frequency=Frequency.MINUTE,
            )
            if minute_trade_data.exists():
                complete_hours = (
                    minute_trade_data.annotate(hour=TruncHour("timestamp"))
                    .values("hour")
                    .annotate(count=Count("id"))
                    .filter(count=Frequency.HOUR)
                    .order_by("-hour")
                )
                for item in complete_hours:
                    hourly_ts_from = item["hour"]
                    hourly_ts_to = hourly_ts_from + pd.Timedelta("1h")
                    hour_minute_data = minute_trade_data.filter(
                        timestamp__gte=hourly_ts_from,
                        timestamp__lt=hourly_ts_to,
                    )
                    convert_trade_data(
                        symbol,
                        hour_minute_data,
                        hourly_ts_from,
                        hourly_ts_to,
                        assert_lease_owned=assert_lease_owned,
                    )

            # Then, convert hourly to daily if complete.
            hourly_trade_data = queryset.filter(
                timestamp__gte=daily_ts_from,
                timestamp__lt=daily_ts_to,
                frequency=Frequency.HOUR,
            )
            is_complete_day = daily_ts_from == get_min_time(
                daily_ts_from, "1d"
            ) and daily_ts_to == daily_ts_from + pd.Timedelta("1d")
            hourly_values = list(hourly_trade_data.values("timestamp", "frequency"))
            hourly_existing = get_existing(hourly_values)
            if (
                is_complete_day
                and len(hourly_values) == Frequency.DAY // Frequency.HOUR
                and has_timestamps(daily_ts_from, daily_ts_to, hourly_existing)
            ):
                convert_trade_data(
                    symbol,
                    hourly_trade_data,
                    daily_ts_from,
                    daily_ts_to,
                    assert_lease_owned=assert_lease_owned,
                )
    logger.info(f"{symbol!s}: done")


def convert_trade_data(
    symbol: Symbol,
    trade_data: QuerySet,
    timestamp_from: datetime.datetime,
    timestamp_to: datetime.datetime,
    assert_lease_owned: Callable[[], None] | None = None,
) -> None:
    """Convert trade data."""
    objs = list(trade_data)
    delta = timestamp_to - timestamp_from
    frequency = int(delta.total_seconds() / 60)
    assert frequency in (Frequency.HOUR, Frequency.DAY)
    first = objs[0]

    prepared_files = {}
    candle_df = None
    trade_candles = [
        dict(obj.json_data["candle"])
        for obj in objs
        if obj.json_data is not None and "candle" in obj.json_data
    ]
    for file_data in symbol.trade_data_fields:
        data_frames = {
            obj: obj.get_data_frame(file_data)
            for obj in objs
            if getattr(obj, file_data).name
        }

        if len(data_frames):
            # Ignore: The behavior of DataFrame concatenation with empty or
            # all-NA entries is deprecated.
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                df = pd.concat(data_frames.values())

            key = (
                "notional"
                if file_data in (FileData.RAW, FileData.AGGREGATED)
                else "totalNotional"
            )
            expected = sum(getattr(frame, key).sum() for frame in data_frames.values())
            actual = df[key].sum()
            assert is_decimal_close(expected, actual)

            df.reset_index(inplace=True)
            if len(df):
                prepared_files[file_data] = TradeData.prepare_data(df)
                if file_data in (
                    FileData.RAW,
                    FileData.AGGREGATED,
                    FileData.FILTERED,
                ):
                    candle_df = df

    json_data = None
    if len(trade_candles) == len(objs) and trade_candles:
        candle = trade_candles[0]
        for payload in trade_candles[1:]:
            candle = merge_cache(candle, dict(payload))
        json_data = {"candle": candle}
    elif candle_df is not None:
        candle = aggregate_candle(
            candle_df,
            min_volume_exponent=1,
            min_notional_exponent=1,
        )
        json_data = {"candle": candle}

    validation_states = {obj.ok for obj in objs}
    if False in validation_states:
        validation_state = False
    elif None in validation_states:
        validation_state = None
    else:
        validation_state = True

    target = TradeData(
        symbol=symbol,
        timestamp=first.timestamp,
        uid=first.uid,
        frequency=frequency,
        ok=validation_state,
        json_data=json_data,
    )
    source_state = {
        obj.pk: (
            obj.timestamp,
            obj.frequency,
            obj.uid,
            obj.raw_data.name,
            obj.aggregated_data.name,
            obj.filtered_data.name,
            obj.json_data,
            obj.ok,
        )
        for obj in objs
    }
    target_names = {
        target._meta.get_field(str(file_data)).generate_filename(
            target,
            prepared.name,
        )
        for file_data, prepared in prepared_files.items()
    }
    source_names = {
        getattr(obj, file_data).name
        for obj in objs
        for file_data in FileData
        if getattr(obj, file_data).name
    }
    if target_names & source_names:
        raise RuntimeError("Compacted target file conflicts with a source file.")

    if assert_lease_owned is not None:
        assert_lease_owned()
    with transaction.atomic():
        locked_objs = list(
            TradeData.objects.select_for_update().filter(pk__in=source_state)
        )
        locked_state = {
            obj.pk: (
                obj.timestamp,
                obj.frequency,
                obj.uid,
                obj.raw_data.name,
                obj.aggregated_data.name,
                obj.filtered_data.name,
                obj.json_data,
                obj.ok,
            )
            for obj in locked_objs
        }
        if locked_state != source_state:
            raise RuntimeError("TradeData changed during compaction.")
        if assert_lease_owned is not None:
            assert_lease_owned()

        target.save(force_insert=True)
        update_fields = []
        for file_data, prepared in prepared_files.items():
            getattr(target, file_data).save(prepared.name, prepared, save=False)
            update_fields.append(str(file_data))
        if assert_lease_owned is not None:
            assert_lease_owned()
        if update_fields:
            target.save(update_fields=update_fields)

        for obj in locked_objs:
            obj._skip_signal = True
            obj.delete()

    logger.info(
        _("Converted {timestamp_from} {timestamp_to} to {frequency}").format(
            timestamp_from=timestamp_from,
            timestamp_to=timestamp_to,
            frequency="daily" if frequency == Frequency.DAY else "hourly",
        )
    )


def clean_trade_data_with_non_existing_files(
    symbol: Symbol, timestamp_from: datetime.datetime, timestamp_to: datetime.datetime
) -> None:
    """Clean trade data with non-existing files."""
    logger.info(_("Checking objects with non existent files"))

    fields = symbol.trade_data_fields
    if not fields:
        return

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
        .only("timestamp", *fields)
    )
    deleted = 0
    total = trade_data.count()
    first_row = trade_data.first()
    next_progress_year = (
        first_row.timestamp.year + 1 if first_row else timestamp_from.year + 1
    )
    for count, obj in enumerate(trade_data):
        row_timestamp = obj.timestamp
        while row_timestamp.year >= next_progress_year:
            logger.info(
                f"{symbol!s}: checked {next_progress_year - 1}, {count}/{total} items, "
                f"deleted {deleted} items"
            )
            next_progress_year += 1

        for field in fields:
            if getattr(obj, field):
                f = getattr(obj, field)
                if not f.storage.exists(f.name):
                    deleted += 1
                    obj.delete()
                    break


def clean_unlinked_trade_data_files(
    symbol: Symbol, timestamp_from: datetime.datetime, timestamp_to: datetime.datetime
) -> None:
    """Clean unlinked trade data files."""
    logger.info(_("Checking unlinked trade data files"))

    fields = symbol.trade_data_fields
    if not fields:
        return

    trade_data = TradeData.objects.filter(symbol=symbol).only(
        *fields,
        "timestamp",
        "frequency",
    )
    t = trade_data.filter(
        timestamp__gte=timestamp_from,
        timestamp__lte=timestamp_to,
    )

    deleted = 0
    checked_days = 0
    scanned_files = 0
    mapping = {
        k: v
        for k, v in {
            FileData.RAW: upload_raw_data_to,
            FileData.AGGREGATED: upload_aggregated_data_to,
            FileData.FILTERED: upload_filtered_data_to,
        }.items()
        if k in fields
    }
    if t.exists():
        min_timestamp_from = t.first().timestamp
        timestamp_from = max(timestamp_from, min_timestamp_from)
        last = t.last()
        max_timestamp_to = last.timestamp + pd.Timedelta(minutes=last.frequency)
        timestamp_to = min(timestamp_to, max_timestamp_to)
        next_progress_year = timestamp_from.year + 1
        for daily_timestamp_from, daily_timestamp_to in iter_timeframe(
            timestamp_from, timestamp_to, value="1d"
        ):
            checked_days += 1
            directory_timestamp_from = get_min_time(
                daily_timestamp_from,
                "1d",
            )
            directory_timestamp_to = directory_timestamp_from + pd.Timedelta("1d")
            for file_data, upload_to in mapping.items():
                expected_files = {
                    Path(getattr(obj, file_data).name).name
                    for obj in trade_data.exclude(**{file_data: ""}).filter(
                        timestamp__gte=directory_timestamp_from,
                        timestamp__lt=directory_timestamp_to,
                    )
                }

                dummy = TradeData(
                    symbol=symbol,
                    timestamp=directory_timestamp_from,
                    frequency=Frequency.MINUTE,
                )
                storage = getattr(dummy, file_data).storage
                directory = Path(upload_to(dummy, "dummy.parquet")).parent
                __, filenames = storage.listdir(directory)
                scanned_files += len(filenames)

                should_delete = [
                    filename for filename in filenames if filename not in expected_files
                ]
                for filename in should_delete:
                    storage.delete(Path(directory) / filename)
                    deleted += 1

            if daily_timestamp_to.year >= next_progress_year:
                logger.info(
                    f"{symbol!s}: checked {next_progress_year - 1}, {checked_days} items, "
                    f"deleted {deleted} items"
                )
                next_progress_year = daily_timestamp_to.year + 1


def _clean_overlapping_trade_data_rows(
    *,
    symbol: Symbol,
    rows: list[TradeData],
    coverage_delta: pd.Timedelta,
    cleanup_frequency: Frequency | tuple[Frequency, ...],
    label: str,
) -> int:
    deleted = 0
    total = len(rows)
    if not rows:
        return deleted

    next_progress_year = rows[0].timestamp.year + 1
    for count, row in enumerate(rows):
        while row.timestamp.year >= next_progress_year:
            logger.info(
                f"{symbol!s}: checked {label} overlap rows {next_progress_year - 1}, "
                f"{count}/{total} items, deleted {deleted} items"
            )
            next_progress_year += 1

        deleted += TradeData.objects.cleanup(
            symbol,
            row.timestamp,
            row.timestamp + coverage_delta,
            cleanup_frequency,
        )

    return deleted


def clean_trade_data_overlaps(
    symbol: Symbol, timestamp_from: datetime.datetime, timestamp_to: datetime.datetime
) -> int:
    """Delete lower-frequency rows already covered by higher-frequency TradeData."""
    logger.info(_("Checking overlapping trade data rows"))

    daily_rows = list(
        TradeData.objects.overlapping(
            symbol, timestamp_from, timestamp_to, Frequency.DAY
        ).only("timestamp")
    )
    hourly_rows = list(
        TradeData.objects.overlapping(
            symbol, timestamp_from, timestamp_to, Frequency.HOUR
        ).only("timestamp")
    )

    deleted = _clean_overlapping_trade_data_rows(
        symbol=symbol,
        rows=daily_rows,
        coverage_delta=pd.Timedelta("1d"),
        cleanup_frequency=(Frequency.HOUR, Frequency.MINUTE),
        label="daily",
    )
    deleted += _clean_overlapping_trade_data_rows(
        symbol=symbol,
        rows=hourly_rows,
        coverage_delta=pd.Timedelta("1h"),
        cleanup_frequency=Frequency.MINUTE,
        label="hourly",
    )

    if not daily_rows and not hourly_rows:
        logger.info(f"{symbol!s}: no overlapping trade-data rows")
    return deleted
