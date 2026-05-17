import datetime
from pathlib import Path

import pandas as pd
from django.db import models, transaction
from django.db.models import Q, QuerySet
from django.utils.translation import gettext_lazy as _
from pandas import DataFrame

from quant_tick.constants import FileData, Frequency
from quant_tick.testing import is_test
from quant_tick.lib import (
    aggregate_candle,
    aggregate_candles,
    aggregate_trades,
    exchange_omits_zero_trade_candles,
    filter_by_timestamp,
    get_existing,
    get_min_time,
    has_timestamps,
    iter_window,
    validate_aggregated_candles,
    validate_totals,
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
    json_data = JSONField(_("json data"), null=True)
    ok = models.BooleanField(_("ok"), null=True, default=False, db_index=True)
    objects = TradeDataQuerySet.as_manager()

    def upload_path(self, directory: str, filename: str) -> str:
        """Upload path.

        Example:
        trades / coinbase / BTCUSD / blaring-crocodile / raw / 2022-01-01 / 0000.parquet
        """
        path = ["test-trades"] if is_test() else ["trades"]
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
        candles: DataFrame,
        *,
        raw_trades: DataFrame | None = None,
        aggregated_trades: DataFrame | None = None,
        filtered_trades: DataFrame | None = None,
    ) -> list["TradeData"]:
        """Write trades and validate them against exchange candles."""
        if raw_trades is None and aggregated_trades is None and filtered_trades is None:
            raise ValueError(
                "Either raw_trades, aggregated_trades, or filtered_trades is required."
            )
        return cls._write_range(
            symbol,
            timestamp_from,
            timestamp_to,
            candles,
            raw_trades=raw_trades,
            aggregated_trades=aggregated_trades,
            filtered_trades=filtered_trades,
        )

    @classmethod
    def validate(
        cls,
        symbol: Symbol,
        timestamp_from: datetime.datetime,
        timestamp_to: datetime.datetime,
        candles: DataFrame,
        *,
        raw_trades: DataFrame | None = None,
        aggregated_trades: DataFrame | None = None,
        filtered_trades: DataFrame | None = None,
    ) -> bool | None:
        """Validate trades against exchange candles without writing TradeData."""
        if raw_trades is None and aggregated_trades is None and filtered_trades is None:
            raise ValueError(
                "Either raw_trades, aggregated_trades, or filtered_trades is required."
            )
        cls._get_write_frequency(timestamp_from, timestamp_to)
        raw_trades, aggregated_trades, filtered_trades = cls._prepare_partition_data(
            symbol,
            timestamp_from,
            timestamp_to,
            raw_trades=raw_trades,
            aggregated_trades=aggregated_trades,
            filtered_trades=filtered_trades,
        )
        return cls._validate_trade_data(
            symbol,
            timestamp_from,
            timestamp_to,
            filter_by_timestamp(candles, timestamp_from, timestamp_to),
            raw_trades=raw_trades,
            aggregated_trades=aggregated_trades,
            filtered_trades=filtered_trades,
        )

    @classmethod
    def _get_write_frequency(
        cls,
        timestamp_from: datetime.datetime,
        timestamp_to: datetime.datetime,
    ) -> Frequency:
        delta = timestamp_to - timestamp_from
        if delta <= pd.Timedelta(0):
            raise ValueError("timestamp_to must be after timestamp_from.")
        frequency = delta.total_seconds() / 60
        assert frequency <= Frequency.DAY
        if (
            timestamp_from == get_min_time(timestamp_from, "1d")
            and timestamp_from == timestamp_to - pd.Timedelta("1d")
        ):
            return Frequency.DAY
        if (
            timestamp_from == get_min_time(timestamp_from, "1h")
            and timestamp_from == timestamp_to - pd.Timedelta("1h")
        ):
            return Frequency.HOUR
        if timestamp_from != get_min_time(timestamp_from, "1min") or (
            timestamp_to != get_min_time(timestamp_to, "1min")
        ):
            raise ValueError("minimum timestamp boundary is 1 minute.")
        return Frequency.MINUTE

    @classmethod
    def _write_range(
        cls,
        symbol: Symbol,
        timestamp_from: datetime.datetime,
        timestamp_to: datetime.datetime,
        candles: DataFrame,
        *,
        raw_trades: DataFrame | None = None,
        aggregated_trades: DataFrame | None = None,
        filtered_trades: DataFrame | None = None,
    ) -> list["TradeData"]:
        frequency = cls._get_write_frequency(timestamp_from, timestamp_to)
        if frequency in (Frequency.DAY, Frequency.HOUR):
            return [
                cls._write_partition(
                    symbol,
                    timestamp_from,
                    timestamp_to,
                    frequency,
                    candles,
                    raw_trades=raw_trades,
                    aggregated_trades=aggregated_trades,
                    filtered_trades=filtered_trades,
                )
            ]
        return [
            cls._write_partition(
                symbol,
                ts_from,
                ts_to,
                Frequency.MINUTE,
                candles,
                raw_trades=raw_trades,
                aggregated_trades=aggregated_trades,
                filtered_trades=filtered_trades,
            )
            for ts_from, ts_to in iter_window(
                timestamp_from,
                timestamp_to,
                value="1min",
            )
        ]

    @classmethod
    def _write_partition(
        cls,
        symbol: Symbol,
        timestamp_from: datetime.datetime,
        timestamp_to: datetime.datetime,
        frequency: Frequency,
        candles: DataFrame,
        *,
        raw_trades: DataFrame | None = None,
        aggregated_trades: DataFrame | None = None,
        filtered_trades: DataFrame | None = None,
    ) -> "TradeData":
        raw_trades, aggregated_trades, filtered_trades = cls._prepare_partition_data(
            symbol,
            timestamp_from,
            timestamp_to,
            raw_trades=raw_trades,
            aggregated_trades=aggregated_trades,
            filtered_trades=filtered_trades,
        )

        cls.objects.cleanup(
            symbol,
            timestamp_from,
            timestamp_to,
            (Frequency.MINUTE, Frequency.HOUR, Frequency.DAY),
        )
        obj = TradeData(symbol=symbol, timestamp=timestamp_from, frequency=frequency)
        return cls._save_trade_data(
            obj,
            filter_by_timestamp(candles, timestamp_from, timestamp_to),
            raw_trades=raw_trades,
            aggregated_trades=aggregated_trades,
            filtered_trades=filtered_trades,
        )

    @classmethod
    def _prepare_partition_data(
        cls,
        symbol: Symbol,
        timestamp_from: datetime.datetime,
        timestamp_to: datetime.datetime,
        *,
        raw_trades: DataFrame | None = None,
        aggregated_trades: DataFrame | None = None,
        filtered_trades: DataFrame | None = None,
    ) -> tuple[DataFrame | None, DataFrame | None, DataFrame | None]:
        raw_trades = cls._filter_frame(raw_trades, timestamp_from, timestamp_to)
        aggregated_trades = cls._filter_frame(
            aggregated_trades,
            timestamp_from,
            timestamp_to,
        )
        filtered_trades = cls._filter_frame(
            filtered_trades,
            timestamp_from,
            timestamp_to,
        )
        raw_trades, aggregated_trades, filtered_trades = cls._prepare_trade_data(
            symbol,
            raw_trades=raw_trades,
            aggregated_trades=aggregated_trades,
            filtered_trades=filtered_trades,
        )
        validate_totals(
            raw_trades=raw_trades,
            aggregated_trades=aggregated_trades,
            filtered_trades=filtered_trades,
        )
        return raw_trades, aggregated_trades, filtered_trades

    @staticmethod
    def _filter_frame(
        data: DataFrame | None,
        timestamp_from: datetime.datetime,
        timestamp_to: datetime.datetime,
    ) -> DataFrame | None:
        if data is None:
            return None
        return filter_by_timestamp(data, timestamp_from, timestamp_to)

    @classmethod
    def _prepare_trade_data(
        cls,
        symbol: Symbol,
        *,
        raw_trades: DataFrame | None = None,
        aggregated_trades: DataFrame | None = None,
        filtered_trades: DataFrame | None = None,
    ) -> tuple[DataFrame | None, DataFrame | None, DataFrame | None]:
        if raw_trades is not None and len(raw_trades):
            if (
                aggregated_trades is None
                and (symbol.save_aggregated or symbol.significant_trade_filter)
            ):
                aggregated_trades = aggregate_trades(raw_trades)
            if symbol.significant_trade_filter and filtered_trades is None:
                filtered_trades = volume_filter_with_time_window(
                    aggregated_trades,
                    min_volume=symbol.significant_trade_filter,
                )

        if aggregated_trades is not None and len(aggregated_trades):
            if symbol.significant_trade_filter and filtered_trades is None:
                filtered_trades = volume_filter_with_time_window(
                    aggregated_trades,
                    min_volume=symbol.significant_trade_filter,
                )

        return raw_trades, aggregated_trades, filtered_trades

    @classmethod
    def _save_trade_data(
        cls,
        obj: "TradeData",
        candles: DataFrame,
        *,
        raw_trades: DataFrame | None = None,
        aggregated_trades: DataFrame | None = None,
        filtered_trades: DataFrame | None = None,
    ) -> "TradeData":
        """Populate and save one TradeData row."""
        uid = ""
        json_data = None
        raw_data = None
        aggregated_data = None
        filtered_data = None

        if raw_trades is not None and len(raw_trades):
            uid = raw_trades.iloc[0].uid
            if obj.symbol.save_raw:
                raw_data = cls.prepare_data(raw_trades)

        if aggregated_trades is not None and len(aggregated_trades):
            if not uid:
                uid = aggregated_trades.iloc[0].uid
            if obj.symbol.save_aggregated:
                aggregated_data = cls.prepare_data(aggregated_trades)

        if filtered_trades is not None and len(filtered_trades):
            if not uid:
                uid = filtered_trades.iloc[0].uid
            if obj.symbol.significant_trade_filter:
                filtered_data = cls.prepare_data(filtered_trades)

        validation = cls._get_validation_frame(
            raw_trades=raw_trades,
            aggregated_trades=aggregated_trades,
            filtered_trades=filtered_trades,
        )

        aggregated_candles = pd.DataFrame([])
        if validation is not None and len(validation):
            timestamp_to = obj.timestamp + pd.Timedelta(f"{obj.frequency}min")
            aggregated_candles = aggregate_candles(
                validation,
                obj.timestamp,
                timestamp_to,
            )
            json_data = {"candle": aggregate_candle(validation)}

        ok = cls._validate_candles(obj.symbol, aggregated_candles, candles)
        with transaction.atomic():
            obj.uid = uid
            obj.json_data = json_data
            obj.raw_data = raw_data
            obj.aggregated_data = aggregated_data
            obj.filtered_data = filtered_data
            obj.ok = ok
            obj.save()
        return obj

    @staticmethod
    def _get_validation_frame(
        *,
        raw_trades: DataFrame | None = None,
        aggregated_trades: DataFrame | None = None,
        filtered_trades: DataFrame | None = None,
    ) -> DataFrame | None:
        if filtered_trades is not None:
            return filtered_trades
        if aggregated_trades is not None:
            return aggregated_trades
        return raw_trades

    @classmethod
    def _validate_trade_data(
        cls,
        symbol: Symbol,
        timestamp_from: datetime.datetime,
        timestamp_to: datetime.datetime,
        candles: DataFrame,
        *,
        raw_trades: DataFrame | None = None,
        aggregated_trades: DataFrame | None = None,
        filtered_trades: DataFrame | None = None,
    ) -> bool | None:
        validation = cls._get_validation_frame(
            raw_trades=raw_trades,
            aggregated_trades=aggregated_trades,
            filtered_trades=filtered_trades,
        )
        aggregated_candles = pd.DataFrame([])
        if validation is not None and len(validation):
            aggregated_candles = aggregate_candles(
                validation,
                timestamp_from,
                timestamp_to,
            )
        return cls._validate_candles(symbol, aggregated_candles, candles)

    @staticmethod
    def _validate_candles(
        symbol: Symbol,
        aggregated_candles: DataFrame,
        exchange_candles: DataFrame,
    ) -> bool | None:
        return validate_aggregated_candles(
            aggregated_candles,
            exchange_candles,
            missing_candles_are_zero=exchange_omits_zero_trade_candles(
                symbol.exchange
            ),
        )

    class Meta:
        db_table = "quant_tick_trade_data"
        ordering = ("timestamp",)
        unique_together = (("symbol", "timestamp", "frequency"),)
        verbose_name = verbose_name_plural = _("trade data")
