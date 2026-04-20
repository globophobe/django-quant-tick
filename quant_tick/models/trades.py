import datetime
from pathlib import Path

import pandas as pd
from django.conf import settings
from django.db import models, transaction
from django.db.models import Q, QuerySet
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
    return instance.upload_path("raw", filename)


def upload_aggregated_data_to(instance: "TradeData", filename: str) -> str:
    return instance.upload_path("aggregated", filename)


def upload_filtered_data_to(instance: "TradeData", filename: str) -> str:
    return instance.upload_path("filtered", filename)


class TradeDataQuerySet(QuerySet):
    """QuerySet helpers for TradeData."""

    def overlapping(
        self,
        symbol: Symbol,
        timestamp_from: datetime.datetime,
        timestamp_to: datetime.datetime,
        frequency: Frequency | tuple[Frequency, ...],
    ) -> QuerySet:
        """Return rows whose coverage overlaps the requested range."""
        frequencies = frequency if isinstance(frequency, tuple) else (frequency,)
        query = Q()
        for item in frequencies:
            query |= Q(
                frequency=item,
                timestamp__gt=timestamp_from - pd.Timedelta(f"{int(item)}min"),
                timestamp__lt=timestamp_to,
            )
        return self.filter(symbol=symbol).filter(query)

    def get_last_uid(self, symbol: Symbol, timestamp: datetime.datetime) -> str:
        """Get the first later non-empty uid for pagination recovery."""
        queryset = self.filter(symbol=symbol, timestamp__gte=timestamp)
        trade_data = queryset.exclude(uid="").order_by("timestamp").first()
        if trade_data:
            return trade_data.uid

    def get_max_timestamp(
        self, symbol: Symbol, timestamp: datetime.datetime
    ) -> datetime.datetime:
        """Clamp to the latest stored TradeData timestamp for the symbol."""
        trade_data = self.filter(symbol=symbol).only("timestamp").last()
        if trade_data and trade_data.timestamp < timestamp:
            return trade_data.timestamp
        return timestamp

    def get_min_timestamp(
        self, symbol: Symbol, timestamp: datetime.datetime
    ) -> datetime.datetime:
        """Clamp to the earliest stored TradeData timestamp for the symbol."""
        trade_data = self.filter(symbol=symbol).only("timestamp").first()
        if trade_data and trade_data.timestamp > timestamp:
            return trade_data.timestamp
        return timestamp

    def has_timestamps(
        self, symbol: Symbol, timestamp_from: datetime, timestamp_to: datetime
    ) -> bool:
        """Whether the symbol has complete TradeData coverage for the range."""
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
        frequency: Frequency | tuple[Frequency, ...],
    ) -> int:
        """Delete overlapping rows and their stored files for the range."""
        objs = list(self.overlapping(symbol, timestamp_from, timestamp_to, frequency))
        if not objs:
            return 0
        with transaction.atomic():
            for obj in objs:
                obj._skip_signal = True
                obj.delete()
        for obj in objs:
            for file_data in FileData:
                getattr(obj, file_data).delete(save=False)
        return len(objs)


class TradeData(AbstractDataStorage):
    """Stored trade-data slice and derived parquet artifacts."""

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
    json_data = JSONField(_("json data"), null=True)
    ok = models.BooleanField(_("ok"), null=True, default=False, db_index=True)
    objects = TradeDataQuerySet.as_manager()

    def upload_path(self, directory: str, filename: str) -> str:
        """Build a deterministic storage path for a file field.

        Example:
        trades / coinbase / BTCUSD / blaring-crocodile / raw / 2022-01-01 / 0000.parquet
        """
        path = ["test-trades"] if settings.TEST else ["trades"]
        path += self.symbol.upload_path + [directory, self.timestamp.date().isoformat()]
        fname = self.timestamp.time().strftime("%H%M")
        ext = Path(filename).suffix
        path.append(f"{fname}{ext}")
        return "/".join(path)

    def get_candle_source_data(self) -> str | None:
        """Return the source field of the trade candle."""
        for file_data in (FileData.FILTERED, FileData.AGGREGATED, FileData.RAW):
            if self.has_data_frame(file_data):
                return file_data
        return None

    @classmethod
    def write(
        cls,
        symbol: Symbol,
        timestamp_from: datetime.datetime,
        timestamp_to: datetime.datetime,
        trades: DataFrame,
        candles: DataFrame,
    ) -> None:
        """Dispatch writes by slice frequency."""
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
        """Rewrite the daily TradeData row for the range."""
        cls.objects.cleanup(
            symbol,
            timestamp_from,
            timestamp_to,
            (Frequency.MINUTE, Frequency.HOUR, Frequency.DAY),
        )
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
        """Rewrite the hourly TradeData row for the range."""
        cls.objects.cleanup(
            symbol,
            timestamp_from,
            timestamp_to,
            (Frequency.MINUTE, Frequency.HOUR, Frequency.DAY),
        )
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
        """Rewrite minute TradeData rows for each minute in the range."""
        for ts_from, ts_to in iter_window(timestamp_from, timestamp_to, value="1min"):
            cls.objects.cleanup(
                symbol,
                ts_from,
                ts_to,
                (Frequency.MINUTE, Frequency.HOUR, Frequency.DAY),
            )
            cls.write_data_frame(
                TradeData(symbol=symbol, timestamp=ts_from, frequency=Frequency.MINUTE),
                filter_by_timestamp(trades, ts_from, ts_to),
                filter_by_timestamp(candles, ts_from, ts_to),
            )

    @classmethod
    def write_data_frame(
        cls, obj: "TradeData", trades: DataFrame, candles: DataFrame
    ) -> None:
        """Serialize and persist one TradeData row and its derived artifacts."""
        uid = ""
        json_data = None
        raw_data = None
        aggregated_data = None
        filtered_data = None
        # Are there any trades?
        aggregated_candles = pd.DataFrame([])
        if len(trades):
            # Set TradeData.uid as first uid.
            uid = trades.iloc[0].uid
            symbol = obj.symbol
            validation = trades
            if symbol.save_raw:
                raw_data = cls.prepare_data(trades)
            if symbol.save_aggregated or symbol.significant_trade_filter:
                aggregated = aggregate_trades(trades)
                validation = aggregated
                if symbol.significant_trade_filter:
                    filtered = volume_filter_with_time_window(
                        aggregated,
                        min_volume=symbol.significant_trade_filter,
                    )
                    validation = filtered
                    filtered_data = cls.prepare_data(filtered)
                if symbol.save_aggregated:
                    aggregated_data = cls.prepare_data(aggregated)
            aggregated_candles = aggregate_candles(
                validation,
                obj.timestamp,
                obj.timestamp + pd.Timedelta(f"{obj.frequency}min"),
            )
            assert is_decimal_close(
                aggregated_candles.notional.sum(), trades.notional.sum()
            )
            json_data = {
                "candle": aggregate_candle(
                    validation,
                    min_volume_exponent=1,
                    min_notional_exponent=1,
                )
            }

        ok = validate_aggregated_candles(aggregated_candles, candles)
        with transaction.atomic():
            obj.uid = uid
            obj.json_data = json_data
            obj.raw_data = raw_data
            obj.aggregated_data = aggregated_data
            obj.filtered_data = filtered_data
            obj.ok = ok
            obj.save()

    class Meta:
        db_table = "quant_tick_trade_data"
        ordering = ("timestamp",)
        unique_together = (("symbol", "timestamp", "frequency"),)
        verbose_name = verbose_name_plural = _("trade data")
