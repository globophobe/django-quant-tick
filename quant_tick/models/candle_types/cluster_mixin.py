import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from pandas import DataFrame

from quant_tick.lib import (
    WARMUP_MIN_CLUSTERS,
    cluster_trades,
    combine_clustered_trades,
    compute_bucket_stats,
    get_cluster_bucket,
    merge_bucket_stats,
    update_cluster_ema,
)


class ClusterBucketMixin:
    """Adds cluster percentile bucket stats to any candle type.

    Enable via json_data: {"cluster_bucket_stats": true, "cluster_ema_halflife": 1000}
    """

    def _preprocess_data(self, data_frame: DataFrame, cache_data: dict) -> DataFrame:
        """Cluster trades, handle partial from previous batch, normalize timestamps."""
        if not self.json_data.get("cluster_bucket_stats"):
            return super()._preprocess_data(data_frame, cache_data)

        if data_frame.empty:
            # May still have partial_cluster to process
            if "partial_cluster" in cache_data:
                partial = pd.DataFrame([cache_data.pop("partial_cluster")])
                partial["timestamp"] = pd.to_datetime(partial["timestamp"])
                return partial
            return data_frame

        clusters = cluster_trades(data_frame)

        # Merge partial cluster from previous batch
        if "partial_cluster" in cache_data:
            partial = pd.DataFrame([cache_data.pop("partial_cluster")])
            partial["timestamp"] = pd.to_datetime(partial["timestamp"])
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                if self._should_merge_partial_cluster(partial, clusters, cache_data):
                    combined = pd.concat([partial, clusters], ignore_index=True)
                    clusters = combine_clustered_trades(combined)
                else:
                    # Partial belongs to previous candle, prepend it
                    clusters = pd.concat([partial, clusters], ignore_index=True)

        # Normalize timestamps for window comparisons
        if "timestamp" in clusters.columns:
            clusters["timestamp"] = pd.to_datetime(clusters["timestamp"])

        return clusters

    def _should_merge_partial_cluster(
        self, partial: DataFrame, new_clusters: DataFrame, cache_data: dict
    ) -> bool:
        """Whether to merge partial with first new cluster.

        Override in subclasses:
        - TimeBasedCandle: merge if same window
        - ConstantCandle/Adaptive: always merge (no window concept)
        """
        return True  # Default: always merge

    def _is_cluster_ema_ready(self, cache_data: dict) -> bool:
        """Check if EMA has enough samples to produce meaningful buckets.

        Warmup candles are intentionally dropped.
        """
        if not self.json_data.get("cluster_bucket_stats"):
            return True  # Not using cluster stats, no warmup needed
        ema_count = int(cache_data.get("cluster_ema_count", 0))
        return ema_count >= WARMUP_MIN_CLUSTERS

    def _build_candle(self, df: DataFrame, timestamp: datetime | None = None) -> dict:
        """Build candle with bucket stats."""
        candle = super()._build_candle(df, timestamp)
        if self.json_data.get("cluster_bucket_stats") and "bucket" in df.columns:
            candle.update(compute_bucket_stats(df))
        return candle

    def _merge_cache(self, prev: dict, curr: dict) -> dict:
        """Merge OHLCV and bucket stats from previous batch.

        lib merge_cache() only merges OHLCV. We must explicitly merge bucket stats.
        """
        # First merge OHLCV via parent
        merged = super()._merge_cache(prev, curr)

        if not self.json_data.get("cluster_bucket_stats"):
            return merged

        # Extract bucket stats from both
        prev_stats = prev.get("bucket_stats", {})
        curr_stats = {k: v for k, v in curr.items() if k.startswith("p")}

        # Merge bucket stats
        if prev_stats or curr_stats:
            merged_stats = merge_bucket_stats(prev_stats, curr_stats)
            merged.update(merged_stats)

        return merged

    def _get_next_cache(
        self,
        df: DataFrame,
        cache_data: dict,
        timestamp: datetime | None = None,
    ) -> dict:
        """Cache incomplete candle with bucket stats.

        bucket_stats are stored separately in cache["next"]["bucket_stats"] and
        merged via _merge_cache when the candle completes.
        """
        if not self.json_data.get("cluster_bucket_stats"):
            return super()._get_next_cache(df, cache_data, timestamp)

        # Handle empty df (e.g., after deferring last cluster)
        if df.empty:
            if "next" not in cache_data:
                cache_data["next"] = {"timestamp": timestamp}
            return cache_data

        # Call parent to cache OHLCV
        cache_data = super()._get_next_cache(df, cache_data, timestamp)

        if "bucket" in df.columns:
            stats = compute_bucket_stats(df)
            if "next" in cache_data:
                # Merge with existing bucket_stats if present
                existing = cache_data["next"].get("bucket_stats", {})
                if existing:
                    stats = merge_bucket_stats(existing, stats)
                cache_data["next"]["bucket_stats"] = stats

        return cache_data

    def _defer_last_cluster(self, df: DataFrame, cache_data: dict) -> DataFrame:
        """Stash last cluster as partial for next batch. Returns df without last row."""
        if not self.json_data.get("cluster_bucket_stats"):
            return df
        if len(df) > 0:
            cache_data["partial_cluster"] = df.iloc[-1].to_dict()
            return df.iloc[:-1]
        return df

    def _assign_buckets_and_update_ema(
        self, clusters: DataFrame, cache_data: dict, halflife: int
    ) -> DataFrame:
        """Assign buckets based on rolling EMA of cluster volume."""
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
