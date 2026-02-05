from datetime import datetime
from decimal import Decimal

import pandas as pd
from pandas import DataFrame

from quant_tick.utils import gettext_lazy as _

from ..candles import CacheResetMixin, Candle
from .cluster_mixin import ClusterBucketMixin


class ConstantCandle(ClusterBucketMixin, CacheResetMixin, Candle):
    """Fixed-threshold bars that close after accumulating a constant amount of activity.

    These bars close when a specific measure of market activity hits a fixed threshold.
    Unlike time-based bars that close every N minutes regardless of activity, constant
    bars adapt to market pace: they close faster during busy periods and slower during
    quiet periods.

    Common types:
    - Tick bars: Close after N ticks (e.g., 1000 ticks per bar)
    - Volume bars: Close after N contracts traded (e.g., 10,000 BTC)
    - Dollar bars: Close after $N notional traded (e.g., $1M USD)

    Why use constant bars instead of time bars? Market activity is not uniform. During
    volatile periods, more information arrives per minute. During quiet periods, less
    happens. Constant bars ensure each bar contains roughly the same amount of market
    activity, making them more stationary and better for statistical analysis.

    Based on AFML Chapter 2 - constant bars are the foundation for more advanced
    information-driven sampling (imbalance bars, run bars). They're simple, effective,
    and widely used in quantitative finance.
    """

    def get_initial_cache(self, timestamp: datetime) -> dict:
        """Get initial cache."""
        cache = {}
        if "cache_reset" in self.json_data:
            cache["date"] = timestamp.date()
        cache["sample_value"] = 0
        return cache

    def aggregate(
        self,
        timestamp_from: datetime,
        timestamp_to: datetime,
        data_frame: DataFrame,
        cache_data: dict,
    ) -> tuple[list, dict | None]:
        """Aggregate."""
        # Preprocess: cluster trades if cluster_bucket_stats enabled
        data_frame = self._preprocess_data(data_frame, cache_data)
        use_clusters = self.json_data.get("cluster_bucket_stats") and not data_frame.empty
        halflife = self.json_data.get("cluster_ema_halflife", 1000)

        start = 0
        data = []
        for index, row in data_frame.iterrows():
            cache_data["sample_value"] += self.get_sample_value(row)
            if self.should_aggregate_candle(cache_data):
                df = data_frame.loc[start:index]
                # Assign buckets for complete candle
                if use_clusters:
                    df = self._assign_buckets_and_update_ema(df, cache_data, halflife)
                candle = self._build_candle(df)
                if "next" in cache_data:
                    previous = cache_data.pop("next")
                    if "open" in previous:
                        candle = self._merge_cache(previous, candle)
                # Only emit if EMA is warmed up
                if self._is_cluster_ema_ready(cache_data):
                    data.append(candle)
                # Reinitialize cache
                cache_data["sample_value"] = 0
                # Next index
                start = index + 1

        # Cache incomplete candle
        is_last_row = start == len(data_frame)
        if not is_last_row:
            df = data_frame.loc[start:]
            # Defer last cluster before bucket assignment
            if use_clusters:
                df = self._defer_last_cluster(df, cache_data)
                if not df.empty:
                    df = self._assign_buckets_and_update_ema(df, cache_data, halflife)
            cache_data = self._get_next_cache(df, cache_data)

        data, cache_data = self.get_incomplete_candle(timestamp_to, data, cache_data)
        return data, cache_data

    def get_sample_value(self, row: tuple) -> Decimal | int:
        """Get sample value.

        Raw trades have totalVolume/totalNotional/totalTicks.
        Clustered data has volume/notional/ticks.
        """
        sample_type = self.json_data["sample_type"]
        # Try clustered column first, fall back to raw trade column
        if sample_type in row.index:
            return row[sample_type]
        return row["total" + sample_type.title()]

    def should_aggregate_candle(self, data: dict) -> bool:
        """Should aggregate candle."""
        return data["sample_value"] >= self.json_data["target_value"]

    def get_incomplete_candle(
        self, timestamp: datetime, data: list, cache_data: dict
    ) -> tuple[list, dict]:
        """Get incomplete candle.

        Saved only if cache resets next iteration.
        """
        ts = timestamp + pd.Timedelta("1us")
        if self.should_reset_cache(ts, cache_data):
            if "next" in cache_data:
                candle = cache_data.pop("next")
                # Flatten bucket_stats into candle if present
                if "bucket_stats" in candle:
                    candle.update(candle.pop("bucket_stats"))
                candle["incomplete"] = True
                # Only emit if EMA is warmed up
                if self._is_cluster_ema_ready(cache_data):
                    data.append(candle)
        return data, cache_data

    class Meta:
        proxy = True
        verbose_name = _("constant candle")
        verbose_name_plural = _("constant candles")
