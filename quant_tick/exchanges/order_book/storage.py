"""Compact complete minute summaries without resampling them.

Writers lock the symbol before reading partitions, including collector retries.
New files have immutable names; database changes publish their references before
old files are removed. A failed write leaves existing partitions readable.
"""

import logging
from datetime import UTC
from functools import partial

import pandas as pd
from django.db import transaction
from django.db.models import Count
from django.db.models.functions import TruncDay, TruncHour

from quant_tick.constants import Frequency
from quant_tick.models import OrderBookData, Symbol
from quant_tick.signals.files import discard

logger = logging.getLogger(__name__)


def read_partition(partition):
    frame = partition.get_data_frame()
    expected = pd.date_range(
        partition.timestamp, periods=partition.frequency, freq="min"
    )
    if "timestamp" not in frame or not pd.DatetimeIndex(frame.timestamp).equals(
        expected
    ):
        raise ValueError("Order book partition does not cover its complete interval")
    return frame


def iter_rows(symbol, timestamp_from=None, timestamp_to=None):
    partitions = OrderBookData.objects.overlapping(symbol, timestamp_from, timestamp_to)
    for partition in partitions.iterator(chunk_size=100):
        frame = read_partition(partition)
        if timestamp_from is not None:
            frame = frame[frame.timestamp >= timestamp_from]
        if timestamp_to is not None:
            frame = frame[frame.timestamp < timestamp_to]
        yield from frame.to_dict("records")


def replace_partition(partitions, frame, frequency, staged):
    first = partitions[0]
    stored = OrderBookData(
        symbol_id=first.symbol_id, timestamp=first.timestamp, frequency=frequency
    )
    stored.data.save("data.parquet", OrderBookData.prepare_data(frame), save=False)
    staged.append(stored)
    OrderBookData.objects.filter(pk=first.pk).update(
        data=stored.data.name, json_data=None, frequency=frequency
    )
    if first.data.name:
        transaction.on_commit(partial(discard, first.data.storage, first.data.name))
    OrderBookData.objects.filter(pk__in=[row.pk for row in partitions[1:]]).delete()


def write_minutes(symbol, rows, staged):
    """Caller holds the symbol lock and cleans staged uploads on rollback."""
    if not rows:
        return
    moments = pd.DatetimeIndex([moment for moment, _ in rows])
    if (
        not moments.is_monotonic_increasing
        or not moments.is_unique
        or moments.tz is None
        or not moments.equals(moments.floor("min"))
    ):
        raise ValueError("Order book summaries must be ordered unique UTC minutes")
    pending = dict(rows)
    partitions = OrderBookData.objects.overlapping(
        symbol, moments[0], moments[-1] + pd.Timedelta(minutes=1)
    ).exclude(frequency=Frequency.MINUTE)
    for partition in partitions:
        end = partition.timestamp + pd.Timedelta(minutes=partition.frequency)
        updates = {
            moment: values
            for moment, values in pending.items()
            if partition.timestamp <= moment < end
        }
        if not updates:
            continue
        frame = read_partition(partition).set_index("timestamp")
        for moment, values in updates.items():
            if set(values) != set(frame.columns):
                raise ValueError("Order book retry has a different summary schema")
            frame.loc[moment] = values
            del pending[moment]
        replace_partition([partition], frame.reset_index(), partition.frequency, staged)
    OrderBookData.objects.bulk_create(
        [
            OrderBookData(symbol=symbol, timestamp=moment, json_data=values)
            for moment, values in pending.items()
        ],
        update_conflicts=True,
        update_fields=["json_data"],
        unique_fields=["symbol", "timestamp"],
        batch_size=1000,
    )


def compact_partition(
    symbol, start, source_frequency, frequency, *, assert_lease_owned=None
):
    staged = []
    end = start + pd.Timedelta(minutes=frequency)
    try:
        with transaction.atomic():
            Symbol.objects.select_for_update().get(pk=symbol.pk)
            if assert_lease_owned:
                assert_lease_owned()
            partitions = list(OrderBookData.objects.overlapping(symbol, start, end))
            if len(partitions) != frequency // source_frequency or any(
                row.frequency != source_frequency for row in partitions
            ):
                return
            frame = pd.concat(
                [read_partition(row) for row in partitions], ignore_index=True
            )
            expected = pd.date_range(start, periods=frequency, freq="min")
            if not pd.DatetimeIndex(frame.timestamp).equals(expected):
                raise ValueError("Order book compaction requires a complete interval")
            replace_partition(partitions, frame, frequency, staged)
            if assert_lease_owned:
                assert_lease_owned()
    except Exception:
        for stored in staged:
            discard(stored.data.storage, stored.data.name)
        raise
    logger.info("%s: compacted order book %s to %s", symbol, start, end)


def convert_order_book_data_to_daily(
    symbol, timestamp_from, timestamp_to, *, assert_lease_owned=None
):
    for source, target, truncate in (
        (Frequency.MINUTE, Frequency.HOUR, TruncHour),
        (Frequency.HOUR, Frequency.DAY, TruncDay),
    ):
        candidates = (
            OrderBookData.objects.filter(
                symbol=symbol,
                frequency=source,
                timestamp__gte=timestamp_from,
                timestamp__lt=timestamp_to,
            )
            .annotate(boundary=truncate("timestamp", tzinfo=UTC))
            .values("boundary")
            .annotate(count=Count("pk"))
            .filter(count=target // source)
            .order_by("-boundary")
        )
        for candidate in candidates.iterator():
            start = candidate["boundary"]
            if start + pd.Timedelta(minutes=target) <= timestamp_to:
                compact_partition(
                    symbol, start, source, target, assert_lease_owned=assert_lease_owned
                )
