import datetime
from pathlib import Path

import pandas as pd
from django.db import models
from django.db.models import QuerySet
from pandas import DataFrame

from quant_tick.constants import FileData, Frequency
from quant_tick.lib import (
    aggregate_candle,
    aggregate_trades,
    cluster_trades_with_time_window,
    filter_by_timestamp,
    get_existing,
    get_missing,
    get_next_time,
    has_timestamps,
    validate_data_frame,
    volume_filter_with_time_window,
)
from quant_tick.utils import gettext_lazy as _

from .base import AbstractDataStorage, JSONField
from .symbols import Symbol


def upload_raw_data_to(instance: "TradeData", filename: str) -> str:
    """Upload raw data to."""
    return instance.upload_path("raw", filename)


def upload_aggregated_data_to(instance: "TradeData", filename: str) -> str:
    """Upload aggregated data to."""
    return instance.upload_path("aggregated", filename)


def upload_filtered_data_to(instance: "TradeData", filename: str) -> str:
    """Upload filtered data to."""
    return instance.upload_path("filtered", filename)


def upload_clustered_data_to(instance: "TradeData", filename: str) -> str:
    """Upload trade clustered data to."""
    return instance.upload_path("clustered", filename)


def upload_candle_data_to(instance: "TradeData", filename: str) -> str:
    """Upload candle data to."""
    return instance.upload_path("candles", filename)


class TradeDataQuerySet(QuerySet):
    """Trade data queryset."""

    def get_last_uid(self, symbol: Symbol, timestamp: datetime.datetime) -> str:
        """Get last uid."""
        queryset = self.filter(symbol=symbol, timestamp__gte=timestamp)
        trade_data = queryset.exclude(uid="").order_by("timestamp").first()
        if trade_data:
            return trade_data.uid

    def get_max_timestamp(
        self, symbol: Symbol, timestamp: datetime.datetime
    ) -> datetime.datetime:
        """Get max timestamp."""
        trade_data = self.filter(symbol=symbol).only("timestamp").last()
        if trade_data and trade_data.timestamp < timestamp:
            return trade_data.timestamp
        return timestamp

    def get_min_timestamp(
        self, symbol: Symbol, timestamp: datetime.datetime
    ) -> datetime.datetime:
        """Get min timestamp."""
        trade_data = self.filter(symbol=symbol).only("timestamp").first()
        if trade_data and trade_data.timestamp > timestamp:
            return trade_data.timestamp
        return timestamp

    def has_timestamps(
        self, symbol: Symbol, timestamp_from: datetime, timestamp_to: datetime
    ) -> bool:
        """Has timestamps."""
        trade_data = (
            TradeData.objects.filter(
                symbol=symbol,
                timestamp__gte=timestamp_from,
                timestamp__lt=timestamp_to,
            )
            .only("timestamp", "frequency")
            .values("timestamp", "frequency")
        )
        existing = get_existing(trade_data.values("timestamp", "frequency"))
        return has_timestamps(timestamp_from, timestamp_to, existing)


class TradeData(AbstractDataStorage):
    """Trade data."""

    symbol = models.ForeignKey(
        "quant_tick.Symbol", related_name="trade_data", on_delete=models.CASCADE
    )
    timestamp = models.DateTimeField(_("timestamp"), db_index=True)
    uid = models.CharField(_("uid"), blank=True, max_length=255)
    frequency = models.PositiveIntegerField(
        _("frequency"),
        choices=[c for c in Frequency.choices if c[0] != Frequency.WEEK],
        db_index=True,
    )
    raw_data = models.FileField(_("raw data"), blank=True, upload_to=upload_raw_data_to)
    aggregated_data = models.FileField(
        _("aggregated data"), blank=True, upload_to=upload_aggregated_data_to
    )
    filtered_data = models.FileField(
        _("filtered data"), blank=True, upload_to=upload_filtered_data_to
    )
    clustered_data = models.FileField(
        _("clustered data"), blank=True, upload_to=upload_clustered_data_to
    )
    candle_data = models.FileField(
        _("candle data"), blank=True, upload_to=upload_candle_data_to
    )
    json_data = JSONField(_("json data"), null=True)
    ok = models.BooleanField(_("ok"), null=True, default=False, db_index=True)
    objects = TradeDataQuerySet.as_manager()

    def upload_path(self, directory: str, filename: str) -> str:
        """Upload data to.

        Example: {path_prefix} / {directory} / 2022-01-01 / 0000.parquet
        """
        path = self.symbol.upload_path + [directory, self.timestamp.date().isoformat()]
        fname = self.timestamp.time().strftime("%H%M")
        ext = Path(filename).suffix
        path.append(f"{fname}{ext}")
        return "/".join(path)

    @classmethod
    def write(
        cls,
        symbol: Symbol,
        timestamp_from: datetime.datetime,
        timestamp_to: datetime.datetime,
        trades: DataFrame,
        candles: DataFrame,
    ) -> None:
        """Write data."""
        delta = timestamp_to - timestamp_from
        frequency = delta.total_seconds() / 60
        assert frequency <= Frequency.HOUR
        is_first_minute = timestamp_from.time().minute == 0
        is_hourly = timestamp_from == timestamp_to - pd.Timedelta("1h")
        if is_first_minute and is_hourly:
            cls.write_hour(symbol, timestamp_from, timestamp_to, trades, candles)
        else:
            cls.write_minutes(symbol, timestamp_from, timestamp_to, trades, candles)

    @classmethod
    def write_hour(
        cls,
        symbol: Symbol,
        timestamp_from: datetime.datetime,
        timestamp_to: datetime.datetime,
        trades: DataFrame,
        candles: DataFrame,
    ) -> None:
        """Write hourly data."""
        params = {
            "symbol": symbol,
            "timestamp": timestamp_from,
            "frequency": Frequency.HOUR,
        }
        try:
            obj = cls.objects.get(**params)
        except cls.DoesNotExist:
            obj = cls(**params)
        finally:
            cls.write_data_frame(obj, trades, candles)

    @classmethod
    def write_minutes(
        cls,
        symbol: Symbol,
        timestamp_from: datetime.datetime,
        timestamp_to: datetime.datetime,
        trades: DataFrame,
        candles: DataFrame,
    ) -> None:
        """Write minute data."""
        queryset = cls.objects.filter(
            symbol=symbol, timestamp__gte=timestamp_from, timestamp__lt=timestamp_to
        )
        existing = get_existing(queryset.values("timestamp", "frequency"))
        timestamps = get_missing(timestamp_from, timestamp_to, existing)
        existing = {
            obj.timestamp: obj
            for obj in cls.objects.filter(
                symbol=symbol,
                timestamp__in=timestamps,
                frequency=Frequency.MINUTE,
            )
        }
        for timestamp in timestamps:
            if timestamp in existing:
                obj = existing[timestamp]
            else:
                obj = cls(
                    symbol=symbol,
                    timestamp=timestamp,
                    frequency=Frequency.MINUTE,
                )
            cls.write_data_frame(
                obj,
                filter_by_timestamp(trades, timestamp, get_next_time(timestamp, "1t")),
                filter_by_timestamp(candles, timestamp, get_next_time(timestamp, "1t")),
            )

    @classmethod
    def write_data_frame(
        cls, obj: "TradeData", trades: DataFrame, candles: DataFrame
    ) -> None:
        """Write data frame."""
        # Delete previously saved data.
        if obj.pk:
            for file_data in FileData:
                getattr(obj, file_data).delete()

        # Set TradeData.uid as first uid.
        if len(trades):
            obj.uid = trades.iloc[0].uid

        # Are there any trades?
        aggregated = pd.DataFrame([])
        filtered = pd.DataFrame([])
        clustered = pd.DataFrame([])
        if len(trades):
            symbol = obj.symbol
            if symbol.save_raw:
                obj.raw_data = cls.prepare_data(trades)
            if symbol.save_aggregated or symbol.save_filtered or symbol.save_clustered:
                aggregated = aggregate_trades(trades)
                obj.aggregated_data = cls.prepare_data(aggregated)
            if symbol.save_filtered or symbol.save_clustered:
                if symbol.significant_trade_filter:
                    filtered = volume_filter_with_time_window(
                        aggregated, min_volume=symbol.significant_trade_filter
                    )
                else:
                    filtered = aggregated
                obj.filtered_data = cls.prepare_data(filtered)
            if symbol.save_clustered:
                clustered = cluster_trades_with_time_window(filtered)
                obj.clustered_data = cls.prepare_data(clustered)

        data_frame = pd.DataFrame([])
        if len(clustered):
            data_frame = clustered
        elif len(filtered):
            data_frame = filtered
        elif len(aggregated):
            data_frame = aggregated
        else:
            data_frame = trades

        obj.json_data = {"candle": aggregate_candle(data_frame)}

        # TODO: Candles
        validated = validate_data_frame(
            obj.timestamp,
            obj.timestamp + pd.Timedelta(f"{obj.frequency}t"),
            data_frame,
            candles,
        )
        if len(validated):
            values = validated.values()
            all_true = all([v is True for v in values])
            some_false = True in [isinstance(v, dict) for v in values]
            some_none = None in values
            if all_true:
                obj.ok = True
            elif some_false or some_none:
                if some_false:
                    obj.ok = False
                else:
                    obj.ok = None
            else:
                raise NotImplementedError

        obj.save()

    class Meta:
        db_table = "quant_candles_trade_data"
        ordering = ("timestamp",)
        verbose_name = verbose_name_plural = _("trade data")
