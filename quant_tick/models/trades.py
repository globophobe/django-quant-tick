import datetime
from pathlib import Path

import pandas as pd
from django.conf import settings
from django.db import models, transaction
from django.db.models import QuerySet
from django.utils.translation import gettext_lazy as _
from pandas import DataFrame

from quant_tick.constants import FileData, Frequency
from quant_tick.lib import (
    aggregate_candle,
    aggregate_candles,
    aggregate_trades,
    filter_by_timestamp,
    get_existing,
    has_timestamps,
    is_decimal_close,
    iter_window,
    validate_aggregated_candles,
    volume_filter_with_time_window,
)

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

    def cleanup(
        self,
        symbol: Symbol,
        timestamp_from: datetime.datetime,
        timestamp_to: datetime.datetime,
        frequency: Frequency,
    ) -> None:
        """Delete existing objects."""
        queryset = self.filter(
            symbol=symbol,
            timestamp__gte=timestamp_from,
            timestamp__lt=timestamp_to,
            frequency=frequency,
        )
        objs = list(queryset)
        with transaction.atomic():
            for obj in objs:
                obj._skip_signal = True
                obj.delete()
        for obj in objs:
            for file_data in FileData:
                getattr(obj, file_data).delete(save=False)


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
    candle_data = models.FileField(
        _("candle data"), blank=True, upload_to=upload_candle_data_to
    )
    json_data = JSONField(_("json data"), null=True)
    ok = models.BooleanField(_("ok"), null=True, default=False, db_index=True)
    objects = TradeDataQuerySet.as_manager()

    def upload_path(self, directory: str, filename: str) -> str:
        """Upload path.

        Example:
        trades / coinbase / BTCUSD / blaring-crocodile / raw / 2022-01-01 / 0000.parquet
        """
        path = ["test-trades"] if settings.TEST else ["trades"]
        path += self.symbol.upload_path + [directory, self.timestamp.date().isoformat()]
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
        assert frequency <= Frequency.DAY
        is_first_hour = timestamp_from.time() == datetime.time.min
        is_daily = timestamp_from == timestamp_to - pd.Timedelta("1d")
        is_first_minute = timestamp_from.time().minute == 0
        is_hourly = timestamp_from == timestamp_to - pd.Timedelta("1h")
        if is_first_hour and is_daily:
            cls.write_day(symbol, timestamp_from, timestamp_to, trades, candles)
        elif is_first_minute and is_hourly:
            cls.write_hour(symbol, timestamp_from, timestamp_to, trades, candles)
        else:
            cls.write_minutes(symbol, timestamp_from, timestamp_to, trades, candles)

    @classmethod
    def write_day(
        cls,
        symbol: Symbol,
        timestamp_from: datetime.datetime,
        timestamp_to: datetime.datetime,
        trades: DataFrame,
        candles: DataFrame,
    ) -> None:
        """Write daily data."""
        cls.objects.cleanup(symbol, timestamp_from, timestamp_to, Frequency.DAY)
        cls.write_data_frame(
            TradeData(symbol=symbol, timestamp=timestamp_from, frequency=Frequency.DAY),
            trades,
            candles,
        )

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
        cls.objects.cleanup(symbol, timestamp_from, timestamp_to, Frequency.HOUR)
        cls.write_data_frame(
            TradeData(symbol=symbol, timestamp=timestamp_from, frequency=Frequency.HOUR),
            trades,
            candles,
        )

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
        for ts_from, ts_to in iter_window(timestamp_from, timestamp_to, value="1min"):
            cls.objects.cleanup(symbol, ts_from, ts_to, Frequency.MINUTE)
            cls.write_data_frame(
                TradeData(symbol=symbol, timestamp=ts_from, frequency=Frequency.MINUTE),
                filter_by_timestamp(trades, ts_from, ts_to),
                filter_by_timestamp(candles, ts_from, ts_to),
            )

    @classmethod
    def write_data_frame(
        cls, obj: "TradeData", trades: DataFrame, candles: DataFrame
    ) -> None:
        """Write data frame."""
        uid = ""
        json_data = None
        raw_data = None
        aggregated_data = None
        filtered_data = None
        candle_data = None

        # Are there any trades?
        aggregated_candles = pd.DataFrame([])
        if len(trades):
            # Set TradeData.uid as first uid.
            uid = trades.iloc[0].uid
            symbol = obj.symbol
            if symbol.save_aggregated or symbol.save_filtered:
                aggregated = aggregate_trades(trades)
                filtered = aggregated
            if symbol.save_raw:
                raw_data = cls.prepare_data(trades)
            if symbol.save_aggregated:
                aggregated_data = cls.prepare_data(aggregated)
            if symbol.save_filtered:
                if symbol.save_filtered and symbol.significant_trade_filter:
                    filtered = volume_filter_with_time_window(
                        aggregated, min_volume=symbol.significant_trade_filter
                    )
                    filtered_data = cls.prepare_data(filtered)
            aggregated_candles = aggregate_candles(
                trades,
                obj.timestamp,
                obj.timestamp + pd.Timedelta(f"{obj.frequency}min"),
            )
            assert is_decimal_close(
                aggregated_candles.notional.sum(), trades.notional.sum()
            )

            json_data = {"candle": aggregate_candle(trades)}

        aggregated_candles, ok = validate_aggregated_candles(
            aggregated_candles,
            candles,
        )
        if len(aggregated_candles):
            candle_data = cls.prepare_data(aggregated_candles)

        with transaction.atomic():
            obj.uid = uid
            obj.json_data = json_data
            obj.raw_data = raw_data
            obj.aggregated_data = aggregated_data
            obj.filtered_data = filtered_data
            obj.candle_data = candle_data
            obj.ok = ok
            obj.save()

    class Meta:
        db_table = "quant_tick_trade_data"
        ordering = ("timestamp",)
        unique_together = (("symbol", "timestamp", "frequency"),)
        verbose_name = verbose_name_plural = _("trade data")
