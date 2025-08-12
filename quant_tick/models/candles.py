import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from django.db import models
from django.db.models import QuerySet
from pandas import DataFrame
from polymorphic.models import PolymorphicModel
from scipy import stats

from quant_tick.constants import FileData, Frequency
from quant_tick.lib import (
    filter_by_timestamp,
    get_existing,
    get_min_time,
    has_timestamps,
    parse_datetime,
)
from quant_tick.utils import gettext_lazy as _

from .base import AbstractCodeName, JSONField
from .trades import TradeData


class Candle(AbstractCodeName, PolymorphicModel):
    """Candle."""

    symbols = models.ManyToManyField(
        "quant_tick.Symbol",
        db_table="quant_tick_candle_symbol",
        verbose_name=_("symbols"),
    )
    date_from = models.DateField(_("date from"), null=True)
    date_to = models.DateField(_("date to"), null=True)
    json_data = JSONField(_("json data"), default=dict)
    is_active = models.BooleanField(_("active"), default=True)

    def initialize(
        self, timestamp_from: datetime, timestamp_to: datetime, retry: bool = False
    ) -> tuple[datetime, datetime, dict]:
        """Initialize."""
        # Is there a specific date from?
        if self.date_from:
            min_timestamp_from = parse_datetime(self.date_from)
            ts_from = (
                min_timestamp_from
                if timestamp_from < min_timestamp_from
                else timestamp_from
            )
        else:
            ts_from = timestamp_from
        # Is there a specific date to?
        if self.date_to:
            max_timestamp_to = parse_datetime(self.date_to)
            ts_to = (
                max_timestamp_to if timestamp_to < max_timestamp_to else timestamp_to
            )
        else:
            ts_to = timestamp_to
        # Does it have trade data?
        max_ts_to = min(
            [
                t.timestamp + pd.Timedelta(f"{t.frequency}min")
                for symbol in self.symbols.all()
                if (
                    t := TradeData.objects.filter(symbol=symbol)
                    .only("timestamp", "frequency")
                    .last()
                )
            ],
            default=ts_to,
        )
        ts_to = ts_to if max_ts_to > ts_to else max_ts_to
        # Does it have a cache?
        candle_cache = (
            CandleCache.objects.filter(
                candle=self,
                timestamp__lte=ts_to,
            )
            .order_by("-timestamp")
            .only("timestamp", "frequency", "json_data")
            .first()
        )
        if candle_cache:
            if not retry:
                timestamp = candle_cache.timestamp + pd.Timedelta(
                    f"{candle_cache.frequency}min"
                )
                ts_from = timestamp if timestamp > ts_from else ts_from
            data = candle_cache.json_data
        else:
            data = self.get_initial_cache(ts_from)
        return ts_from, ts_to, data

    def get_initial_cache(self, timestamp: datetime) -> dict:
        """Get initial cache."""
        return {}

    def get_cache_data(self, timestamp: datetime, data: dict) -> dict:
        """Get cache data."""
        return data

    def get_trade_data(
        self,
        timestamp_from: datetime,
        timestamp_to: datetime,
        only: list | str | None = None,
    ) -> QuerySet:
        """Get trade data."""
        # Trade data may be daily, so timestamp from >= daily timestamp.
        if isinstance(only, str):
            only = [only]
        only = only or []
        return (
            TradeData.objects.filter(
                symbol__in=self.symbols.all(),
                timestamp__gte=get_min_time(timestamp_from, value="1d"),
                timestamp__lt=timestamp_to,
            )
            .select_related("symbol")
            .only(*["timestamp", "frequency"] + only)
        )

    def get_expected_daily_candles(
        self, timestamp_from: datetime, timestamp_to: datetime
    ) -> dict:
        """Get expected daily candles."""
        trade_data = self.get_trade_data(timestamp_from, timestamp_to, only="json_data")
        return trade_data.values_list("json_data", flat=True)

    def get_data_frame(
        self, timestamp_from: datetime, timestamp_to: datetime
    ) -> DataFrame:
        """Get data frame."""
        trade_data = self.get_trade_data(
            timestamp_from, timestamp_to, only=["symbol"] + list(FileData)
        )
        data_frames = []
        for symbol in self.symbols.all():
            target = sorted(
                [obj for obj in trade_data if obj.symbol == symbol],
                key=lambda obj: obj.timestamp,
            )
            dfs = []
            for t in target:
                # Query may contain trade data by minute.
                # Only target timestamps.
                if timestamp_from <= t.timestamp + pd.Timedelta(f"{t.frequency}min"):
                    df = t.get_data_frame(self.json_data["source_data"])
                    if df is not None:
                        dfs.append(df)
            if dfs:
                # Ignore: The behavior of DataFrame concatenation with empty or
                # all-NA entries is deprecated.
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=FutureWarning)
                    df = pd.concat(dfs)
                df.insert(2, "exchange", symbol.exchange)
                df.insert(3, "symbol", symbol.symbol)
                data_frames.append(df)
        if data_frames:
            df = pd.concat(data_frames)
            sort_values = ["timestamp"]
            if "nanoseconds" in df.columns:
                sort_values.append("nanoseconds")
            df = df.sort_values(sort_values)
            return (
                filter_by_timestamp(df, timestamp_from, timestamp_to)
                .reset_index()
                .drop(columns=["index"])
            )
        else:
            return pd.DataFrame([])

    def can_aggregate(self, timestamp_from: datetime, timestamp_to: datetime) -> bool:
        """Can aggregate?"""
        values = []
        for symbol in self.symbols.all():
            # Trade data may be daily, so timestamp from >= daily timestmap.
            trade_data = TradeData.objects.filter(
                symbol=symbol,
                timestamp__gte=get_min_time(timestamp_from, value="1d"),
                timestamp__lt=timestamp_to,
            )
            # Only target timestamps.
            existing = [
                t
                for t in get_existing(trade_data.values("timestamp", "frequency"))
                if t >= timestamp_from and t < timestamp_to
            ]
            values.append(has_timestamps(timestamp_from, timestamp_to, existing))
        return all(values)

    def aggregate(
        self,
        timestamp_from: datetime,
        timestamp_to: datetime,
        data_frame: DataFrame,
        cache_data: dict | None = None,
    ) -> tuple[list, dict | None]:
        """Aggregate."""
        raise NotImplementedError

    def write_cache(
        self,
        timestamp_from: datetime,
        timestamp_to: datetime,
        data: dict | None = None,
        data_frame: DataFrame | None = None,
    ) -> None:
        """Write cache.

        Delete previously saved data.
        """
        queryset = CandleCache.objects.filter(
            candle=self, timestamp__gte=timestamp_from, timestamp__lt=timestamp_from
        )
        queryset.delete()
        delta = timestamp_to - timestamp_from
        # FIXME: Can't define every value in constants.
        frequency = delta.total_seconds() / 60
        CandleCache.objects.create(
            candle=self, timestamp=timestamp_from, frequency=frequency, json_data=data
        )

    def write_data(
        self, timestamp_from: datetime, timestamp_to: datetime, json_data: list[dict]
    ) -> None:
        """Write data.

        Delete previously saved data.
        """
        CandleData.objects.filter(
            candle=self, timestamp__gte=timestamp_from, timestamp__lt=timestamp_to
        ).delete()
        data = []
        for j in json_data:
            timestamp = j.pop("timestamp")
            kwargs = {"timestamp": timestamp, "json_data": j}
            c = CandleData(candle=self, **kwargs)
            data.append(c)
        CandleData.objects.bulk_create(data)

    def get_candle_data(self) -> DataFrame:
        """Get candle data."""
        return DataFrame(
            [
                {"timestamp": data["timestamp"], **data["json_data"]}
                for data in self.candledata_set.order_by("timestamp").values(
                    "timestamp", "json_data"
                )
            ]
        )

    def distribution_analysis(self, data_frame: DataFrame) -> str:
        """López de Prado style distribution analysis."""
        returns = data_frame["close"].pct_change().dropna()
        standardized_returns = (returns - returns.mean()) / returns.std()

        def ascii_histogram(data: list, bins: int = 30, width: int = 40) -> list:
            """ASCII histogram"""
            data = data[(data >= -3) & (data <= 3)]
            if len(data) == 0:
                return ["No data in range"]

            hist, edges = np.histogram(data, bins=bins, density=True)
            max_count = max(hist) if max(hist) > 0 else 1

            lines = []
            for i, count in enumerate(hist):
                bar_length = int((count / max_count) * width)
                bar = "█" * bar_length
                bin_center = (edges[i] + edges[i + 1]) / 2
                lines.append(f"{bin_center:6.2f} |{bar}")
            return lines

        sample_type = self.json_data["sample_type"]
        target_candles = self.json_data.get("target_candles_per_day", "")

        output = []
        output.append(f"Distribution Analysis - {sample_type.title()} Sampling")
        output.append("=" * 50)

        if target_candles:
            output.append(f"Target candles per day: {target_candles}")

        output.append(
            f"\nStandardized Returns ({len(standardized_returns)} observations):"
        )
        for line in ascii_histogram(standardized_returns):
            output.append(line)

        skew_val = stats.skew(standardized_returns)
        kurt_val = stats.kurtosis(standardized_returns)

        output.append("\nStatistics:")
        output.append(f"  Skewness: {skew_val:.3f}")
        output.append(f"  Kurtosis: {kurt_val:.3f}")
        output.append(f"  Sample size: {len(standardized_returns)}")

        # Normality test
        if len(standardized_returns) > 7:  # Minimum for shapiro test
            if len(standardized_returns) <= 5000:
                _, p_val = stats.shapiro(standardized_returns)
                test_name = "Shapiro-Wilk"
            else:
                _, p_val = stats.kstest(standardized_returns, "norm")
                test_name = "Kolmogorov-Smirnov"

            output.append(f"  {test_name} p-value: {p_val:.4f}")
            output.append(f"  Normal? {'No' if p_val < 0.05 else 'Yes'} (α=0.05)")

        return "\n".join(output)

    class Meta:
        db_table = "quant_tick_candle"
        verbose_name = _("candle")
        verbose_name_plural = _("candles")


class CandleCache(models.Model):
    """Candle cache."""

    candle = models.ForeignKey(
        "quant_tick.Candle", related_name="data", on_delete=models.CASCADE
    )
    timestamp = models.DateTimeField(_("timestamp"), db_index=True)
    frequency = models.PositiveIntegerField(
        _("frequency"), choices=Frequency.choices, db_index=True
    )
    json_data = JSONField(_("json data"), default=dict)

    class Meta:
        db_table = "quant_tick_candle_cache"
        ordering = ("timestamp",)
        verbose_name = verbose_name_plural = _("candle cache")


class CandleData(models.Model):
    """Candle data."""

    candle = models.ForeignKey(
        "quant_tick.Candle", on_delete=models.CASCADE, verbose_name=_("candle")
    )
    timestamp = models.DateTimeField(_("timestamp"), db_index=True)
    json_data = JSONField(_("json data"), default=dict)

    class Meta:
        db_table = "quant_tick_candle_data"
        ordering = ("timestamp",)
        verbose_name = verbose_name_plural = _("candle data")
