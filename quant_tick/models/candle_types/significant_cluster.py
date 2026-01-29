from datetime import datetime
from decimal import Decimal

import pandas as pd
from pandas import DataFrame

from quant_tick.lib import (
    aggregate_candle,
    cluster_trades,
    combine_clustered_trades,
    get_next_cache,
    merge_cache,
)
from quant_tick.utils import gettext_lazy as _

from ..candles import CacheResetMixin, Candle


class SignificantCluster(CacheResetMixin, Candle):
    """Candle that triggers when cumulative directional notional reaches threshold.

    This candle type aggregates trades when the cumulative notional value in the
    same direction reaches a configurable threshold. It implements a cumulative
    directional flow signal: significant selling or buying pressure in one direction
    can predict mean reversion.

    json_data config:
    - source_data: FileData.FILTERED (filtered trade data)
    - target_cumulative_notional: Decimal threshold for triggering candle
    - significant_cluster_filter: Minimum notional for a cluster to count toward
      cumulative (small clusters still included in candle data)
    - cache_reset: Optional Frequency.DAY or Frequency.WEEK

    Key distinction:
    - significant_cluster_filter: Only affects which clusters count toward threshold
    - All clusters still included in the emitted candle's aggregated OHLCV data
    """

    def get_initial_cache(self, timestamp: datetime) -> dict:
        """Get initial cache."""
        cache = {}
        if "cache_reset" in self.json_data:
            cache["date"] = timestamp.date()
        cache["cumulative_notional"] = Decimal("0")
        cache["current_direction"] = None
        return cache

    def get_target_cumulative_notional(self, cache_data: dict) -> Decimal:
        """Get target cumulative notional threshold."""
        return Decimal(str(self.json_data["target_cumulative_notional"]))

    def get_significant_cluster_filter(self, cache_data: dict) -> Decimal:
        """Get significant cluster filter threshold."""
        return Decimal(str(self.json_data["significant_cluster_filter"]))

    def get_clusters(self, data_frame: DataFrame, cache_data: dict) -> DataFrame:
        """Cluster trades, handling boundary combining."""
        clusters = cluster_trades(data_frame)

        if "partial_cluster" in cache_data and not clusters.empty:
            partial = pd.DataFrame([cache_data.pop("partial_cluster")])
            clusters = combine_clustered_trades(
                pd.concat([partial, clusters], ignore_index=True)
            )

        return clusters

    def aggregate(
        self,
        timestamp_from: datetime,
        timestamp_to: datetime,
        data_frame: DataFrame,
        cache_data: dict,
    ) -> tuple[list, dict | None]:
        """Aggregate trades into candles based on cumulative directional notional."""
        clusters = self.get_clusters(data_frame, cache_data)

        if clusters.empty:
            return self.get_incomplete_candle(timestamp_to, [], cache_data)

        start = 0
        data = []
        target = self.get_target_cumulative_notional(cache_data)
        sig_filter = self.get_significant_cluster_filter(cache_data)

        for index, row in clusters.iterrows():
            tick_rule = row.get("tickRule")
            notional = row.get("totalNotional", row.get("notional", Decimal("0")))

            if cache_data["current_direction"] is None:
                cache_data["current_direction"] = tick_rule
                if notional >= sig_filter:
                    cache_data["cumulative_notional"] = notional
            elif tick_rule == cache_data["current_direction"]:
                if notional >= sig_filter:
                    cache_data["cumulative_notional"] += notional
            else:
                cache_data["current_direction"] = tick_rule
                if notional >= sig_filter:
                    cache_data["cumulative_notional"] = notional
                else:
                    cache_data["cumulative_notional"] = Decimal("0")

            if cache_data["cumulative_notional"] >= target:
                df = clusters.loc[start:index]
                candle = aggregate_candle(df)
                candle["direction"] = int(cache_data["current_direction"])
                candle["cumulativeNotional"] = cache_data["cumulative_notional"]
                if "next" in cache_data:
                    previous = cache_data.pop("next")
                    candle = merge_cache(previous, candle)
                data.append(candle)
                cache_data["cumulative_notional"] = Decimal("0")
                start = index + 1

        is_last_row = start == len(clusters)
        if not is_last_row:
            cache_data["partial_cluster"] = clusters.iloc[-1].to_dict()
            df = clusters.loc[start : len(clusters) - 2]
            if not df.empty:
                cache_data = get_next_cache(df, cache_data)
        return self.get_incomplete_candle(timestamp_to, data, cache_data)

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
                candle["incomplete"] = True
                data.append(candle)
        return data, cache_data

    class Meta:
        proxy = True
        verbose_name = _("significant cluster")
        verbose_name_plural = _("significant clusters")
