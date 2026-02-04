from datetime import datetime

import numpy as np
import pandas as pd
from pandas import DataFrame

from quant_tick.lib import (
    WARMUP_MIN_CLUSTERS,
    aggregate_candle,
    cluster_trades,
    combine_clustered_trades,
    compute_bucket_stats,
    filter_by_timestamp,
    get_cluster_bucket,
    get_min_time,
    get_next_cache,
    iter_window,
    merge_bucket_stats,
    merge_cache,
    update_cluster_ema,
)
from quant_tick.utils import gettext_lazy as _

from .time_based_candles import TimeBasedCandle


class TimeBasedClusterCandle(TimeBasedCandle):
    """Time-based candle with cluster percentile buckets.

    Emits candles at fixed time intervals (like TimeBasedCandle) with
    cluster statistics broken down by volume percentile buckets.

    Buckets are assigned based on rolling EMA of cluster volumes:
    - p0: smallest clusters (bottom 25% relative to recent history)
    - p25-p95: intermediate buckets
    - p99: largest clusters (top 1% relative to recent history)

    json_data config:
    - window: Time period string (e.g., "1h", "4h", "1d")
    - cluster_ema_halflife: Halflife for EMA in clusters (default 1000)

    Added fields per candle (p0, p25, p50, p75, p90, p95, p99):
    - p{N}.buyClusterCount: Buy clusters in this volume bucket
    - p{N}.totalClusterCount: Total clusters in this volume bucket
    - p{N}.buyClusterNotional: Notional from buy clusters
    - p{N}.totalClusterNotional: Total notional in bucket
    """

    def _is_same_window(self, ts1: datetime, ts2: datetime) -> bool:
        """Check if two timestamps are in the same window."""
        window = self.json_data["window"]
        return get_min_time(ts1, window) == get_min_time(ts2, window)

    def _assign_buckets_and_update_ema(
        self, clusters: DataFrame, cache_data: dict, halflife: int
    ) -> DataFrame:
        """Assign buckets and update EMA for clusters."""
        ema_mean = float(cache_data.get("cluster_ema_mean", 0))
        ema_var = float(cache_data.get("cluster_ema_var", 0))
        ema_count = int(cache_data.get("cluster_ema_count", 0))
        ema_std = np.sqrt(ema_var) if ema_var > 0 else 0.0

        buckets = []
        for _idx, cluster in clusters.iterrows():
            vol = float(cluster["volume"])

            # Guard against invalid volume (zero, negative, NaN, inf)
            if not np.isfinite(vol) or vol <= 0:
                buckets.append(50)
                continue

            # Warm-up: force median bucket
            if ema_count < WARMUP_MIN_CLUSTERS or ema_std <= 0:
                bucket = 50
            else:
                bucket = get_cluster_bucket(vol, ema_mean, ema_std)

            buckets.append(bucket)
            update_cluster_ema(cache_data, vol, halflife)

            # Update for next cluster
            ema_mean = float(cache_data["cluster_ema_mean"])
            ema_var = float(cache_data["cluster_ema_var"])
            ema_count = int(cache_data["cluster_ema_count"])
            ema_std = np.sqrt(ema_var) if ema_var > 0 else 0.0

        clusters = clusters.copy()
        clusters["bucket"] = buckets
        return clusters

    def aggregate(
        self,
        timestamp_from: datetime,
        timestamp_to: datetime,
        data_frame: DataFrame,
        cache_data: dict | None = None,
    ) -> tuple[list, dict | None]:
        """Aggregate trades into time-based cluster candles with EMA buckets."""
        cache_data = cache_data or {}
        data = []
        window = self.json_data["window"]
        halflife = self.json_data.get("cluster_ema_halflife", 1000)

        # Cluster this batch
        if not data_frame.empty:
            batch_clusters = cluster_trades(data_frame)
        else:
            batch_clusters = pd.DataFrame()

        # Handle partial cluster from previous batch
        if "partial_cluster" in cache_data:
            partial = pd.DataFrame([cache_data.pop("partial_cluster")])
            partial_ts = pd.to_datetime(partial["timestamp"].iloc[0])

            if not batch_clusters.empty:
                batch_start = pd.to_datetime(batch_clusters["timestamp"].iloc[0])
                if self._is_same_window(partial_ts, batch_start):
                    # Same window: merge partial with first cluster
                    combined = pd.concat([partial, batch_clusters], ignore_index=True)
                    batch_clusters = combine_clustered_trades(combined)
                else:
                    # Different window: partial closes previous window
                    batch_clusters = pd.concat(
                        [partial, batch_clusters], ignore_index=True
                    )
            else:
                # No new clusters, partial continues alone
                batch_clusters = partial

        # Convert timestamps
        if not batch_clusters.empty and "timestamp" in batch_clusters.columns:
            batch_clusters["timestamp"] = pd.to_datetime(batch_clusters["timestamp"])

        # Setup for iteration
        if "next" in cache_data:
            ts_from = cache_data["next"]["timestamp"]
        else:
            ts_from = timestamp_from

        max_ts_to = self.get_max_timestamp_to(ts_from, timestamp_to)
        ts_to = None

        # Process windows
        for win_from, win_to in iter_window(ts_from, max_ts_to, window):
            ts_to = win_to
            if not batch_clusters.empty:
                win_clusters = filter_by_timestamp(batch_clusters, win_from, win_to)
            else:
                win_clusters = pd.DataFrame()

            if not win_clusters.empty:
                # Assign buckets and update EMA for complete window
                win_clusters = self._assign_buckets_and_update_ema(
                    win_clusters, cache_data, halflife
                )

                # Build candle
                candle = aggregate_candle(win_clusters, timestamp=win_from)
                stats = compute_bucket_stats(win_clusters)

                # Merge with cached stats from previous batches
                if "next" in cache_data:
                    prev = cache_data.pop("next")
                    # Only merge OHLCV if prev has it (might only have timestamp)
                    if "open" in prev:
                        candle = merge_cache(prev, candle)
                    if "bucket_stats" in prev:
                        stats = merge_bucket_stats(prev["bucket_stats"], stats)

                candle.update(stats)
                data.append(candle)
            elif "next" in cache_data:
                # Window closes but no new clusters - emit cached candle
                candle = cache_data.pop("next")
                if "bucket_stats" in candle:
                    stats = candle.pop("bucket_stats")
                    candle.update(stats)
                data.append(candle)

        # Handle incomplete window
        could_not_iterate = ts_to is None
        could_not_complete = ts_to and ts_to != timestamp_to

        if could_not_iterate or could_not_complete:
            cache_ts_from = ts_from if could_not_iterate else ts_to
            if not batch_clusters.empty:
                cache_clusters = filter_by_timestamp(
                    batch_clusters, cache_ts_from, timestamp_to
                )
            else:
                cache_clusters = pd.DataFrame()

            if not cache_clusters.empty:
                # Identify partial: last cluster in incomplete window
                confirmed = cache_clusters.iloc[:-1]
                partial = cache_clusters.iloc[-1]

                # Stash partial for next batch (don't count it yet!)
                cache_data["partial_cluster"] = partial.to_dict()

                # Only process confirmed clusters
                if len(confirmed) > 0:
                    confirmed = self._assign_buckets_and_update_ema(
                        confirmed, cache_data, halflife
                    )
                    stats = compute_bucket_stats(confirmed)
                    cache_data = get_next_cache(
                        confirmed, cache_data, timestamp=cache_ts_from
                    )

                    # Merge stats into cache
                    if "bucket_stats" in cache_data.get("next", {}):
                        cache_data["next"]["bucket_stats"] = merge_bucket_stats(
                            cache_data["next"]["bucket_stats"], stats
                        )
                    else:
                        cache_data["next"]["bucket_stats"] = stats
                elif "next" not in cache_data:
                    # Only partial, just set timestamp for ts_from
                    cache_data["next"] = {"timestamp": cache_ts_from}

        return data, cache_data

    class Meta:
        proxy = True
        verbose_name = _("time based cluster candle")
        verbose_name_plural = _("time based cluster candles")
