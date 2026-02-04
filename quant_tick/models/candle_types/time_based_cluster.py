from datetime import datetime
from decimal import Decimal

import pandas as pd
from pandas import DataFrame

from quant_tick.lib import (
    aggregate_candle,
    cluster_trades,
    combine_clustered_trades,
    filter_by_timestamp,
    get_min_time,
    get_next_cache,
    iter_window,
    merge_cache,
)
from quant_tick.utils import gettext_lazy as _

from .time_based_candles import TimeBasedCandle

PERCENTILES = [0, 25, 50, 75, 90, 95, 99]
THRESHOLDS = [25, 50, 75, 90, 95, 99]


class TimeBasedClusterCandle(TimeBasedCandle):
    """Time-based candle with cluster percentile buckets.

    Emits candles at fixed time intervals (like TimeBasedCandle) with
    cluster statistics broken down by volume percentile buckets.

    json_data config:
    - window: Time period string (e.g., "1h", "4h", "1d")

    Added fields per candle (p0, p25, p50, p75, p90, p95, p99):
    - p{N}.buyClusterCount: Buy clusters in this volume bucket
    - p{N}.totalClusterCount: Total clusters in this volume bucket
    - p{N}.buyClusterNotional: Notional from buy clusters
    - p{N}.totalClusterNotional: Total notional in bucket

    Each bucket captures from its percentile to the next:
    - p0: 0th-25th percentile (smallest clusters)
    - p25: 25th-50th percentile
    - p50: 50th-75th percentile
    - p75: 75th-90th percentile
    - p90: 90th-95th percentile
    - p95: 95th-99th percentile
    - p99: 99th-100th percentile (largest clusters)
    """

    def _is_same_window(self, ts1: datetime, ts2: datetime) -> bool:
        """Check if two timestamps are in the same window."""
        window = self.json_data["window"]
        return get_min_time(ts1, window) == get_min_time(ts2, window)

    def _compute_cluster_stats(self, clusters: DataFrame) -> dict:
        """Compute cluster statistics by volume percentile bucket."""
        if clusters.empty:
            return {}

        stats = {}
        volume_col = "volume" if "volume" in clusters.columns else None
        notional_col = "notional" if "notional" in clusters.columns else None

        if volume_col is None:
            return {}

        volumes = clusters[volume_col].astype(float)
        thresholds = {t: volumes.quantile(t / 100) for t in THRESHOLDS}

        for i, p in enumerate(PERCENTILES):
            is_first = i == 0
            is_last = i == len(PERCENTILES) - 1

            if is_first:
                bucket_mask = volumes <= thresholds[THRESHOLDS[0]]
            elif is_last:
                bucket_mask = volumes > thresholds[THRESHOLDS[-1]]
            else:
                lower = thresholds[THRESHOLDS[i - 1]]
                upper = thresholds[THRESHOLDS[i]]
                bucket_mask = (volumes > lower) & (volumes <= upper)

            bucket = clusters[bucket_mask]
            buy_mask = bucket["tickRule"] == 1

            buy_notional = Decimal(0)
            total_notional = Decimal(0)
            if notional_col and not bucket.empty:
                buy_notional = bucket.loc[buy_mask, notional_col].sum()
                total_notional = bucket[notional_col].sum()

            stats[f"p{p}"] = {
                "buyClusterCount": int(buy_mask.sum()),
                "totalClusterCount": len(bucket),
                "buyClusterNotional": buy_notional,
                "totalClusterNotional": total_notional,
            }

        return stats

    def _build_cluster_candle(
        self,
        clusters: DataFrame,
        timestamp: datetime,
        cache_data: dict,
    ) -> dict:
        """Build a time-based cluster candle."""
        candle = aggregate_candle(clusters, timestamp=timestamp)

        stats = self._compute_cluster_stats(clusters)
        candle.update(stats)

        if "next" in cache_data:
            previous = cache_data.pop("next")
            # Only merge if previous has OHLCV data
            if "open" in previous:
                candle = merge_cache(previous, candle)

        return candle

    def aggregate(
        self,
        timestamp_from: datetime,
        timestamp_to: datetime,
        data_frame: DataFrame,
        cache_data: dict | None = None,
    ) -> tuple[list, dict | None]:
        """Aggregate trades into time-based cluster candles.

        1. Cluster all trades in the batch
        2. Iterate over time windows
        3. For each window, filter clusters and compute percentile stats
        """
        cache_data = cache_data or {}
        data = []
        window = self.json_data["window"]

        if "partial_cluster" in cache_data:
            partial = pd.DataFrame([cache_data.pop("partial_cluster")])
            partial_ts = pd.to_datetime(partial["timestamp"].iloc[0])

            if not data_frame.empty:
                clusters = cluster_trades(data_frame)
                batch_start = pd.to_datetime(data_frame["timestamp"].min())

                if self._is_same_window(partial_ts, batch_start):
                    # Same window: merge if same direction
                    if not clusters.empty:
                        combined = pd.concat([partial, clusters], ignore_index=True)
                        clusters = combine_clustered_trades(combined)
                    else:
                        clusters = partial
                else:
                    # Different window: keep separate (partial is flushed)
                    clusters = pd.concat([partial, clusters], ignore_index=True)
            else:
                clusters = partial
        elif not data_frame.empty:
            clusters = cluster_trades(data_frame)
        else:
            clusters = pd.DataFrame()

        if clusters.empty:
            return [], cache_data

        if "timestamp" in clusters.columns:
            clusters["timestamp"] = pd.to_datetime(clusters["timestamp"])

        if "next" in cache_data:
            ts_from = cache_data["next"]["timestamp"]
        else:
            ts_from = timestamp_from

        max_ts_to = self.get_max_timestamp_to(ts_from, timestamp_to)
        ts_to = None

        for win_from, win_to in iter_window(ts_from, max_ts_to, window):
            ts_to = win_to
            win_clusters = filter_by_timestamp(clusters, win_from, win_to)

            if not win_clusters.empty:
                candle = self._build_cluster_candle(
                    win_clusters, win_from, cache_data
                )
                data.append(candle)
            elif "next" in cache_data:
                candle = cache_data.pop("next")
                data.append(candle)

        could_not_iterate = ts_to is None
        could_not_complete = ts_to and ts_to != timestamp_to

        if could_not_iterate or could_not_complete:
            cache_ts_from = ts_from if could_not_iterate else ts_to
            cache_clusters = filter_by_timestamp(
                clusters, cache_ts_from, timestamp_to
            )
            if not cache_clusters.empty:
                # Track last cluster as partial for potential merging
                cache_data["partial_cluster"] = cache_clusters.iloc[-1].to_dict()

                # For cache["next"], exclude the last cluster to avoid double-counting
                # when partial_cluster is re-added in next batch
                confirmed_clusters = cache_clusters.iloc[:-1]
                if not confirmed_clusters.empty:
                    cache_data = get_next_cache(
                        confirmed_clusters, cache_data, timestamp=cache_ts_from
                    )
                    stats = self._compute_cluster_stats(confirmed_clusters)
                    if "next" in cache_data:
                        cache_data["next"].update(stats)
                else:
                    # Still need timestamp for ts_from in next batch
                    cache_data["next"] = {"timestamp": cache_ts_from}

        return data, cache_data

    class Meta:
        proxy = True
        verbose_name = _("time based cluster candle")
        verbose_name_plural = _("time based cluster candles")
