import warnings
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

TOTAL_COLS = (
    "totalVolume",
    "totalBuyVolume",
    "totalNotional",
    "totalBuyNotional",
    "totalTicks",
    "totalBuyTicks",
)


class SignificantCluster(CacheResetMixin, Candle):
    """Significant Cluster.

    Emits a candle every time a cluster's volume passes the significant cluster
    filter threshold. The candle aggregates all clusters since the last emission
    up to and including the triggering cluster.

    json_data config:
    - source_data: FileData.FILTERED
    - significant_cluster_filter: Minimum volume for a cluster to trigger candle
    - cache_reset: Optional Frequency.DAY or Frequency.WEEK
    """

    @property
    def snapshot_window(self) -> int:
        """Snapshot window."""
        n = int(self.json_data.get("snapshot_window", 25))
        return max(n, 0)

    def get_significant_cluster_filter(self, cache_data: dict) -> Decimal:
        """Get significant cluster filter."""
        return Decimal(str(self.json_data["significant_cluster_filter"]))

    def get_initial_cache(self, timestamp: datetime) -> dict:
        """Get initial cache."""
        cache = {}
        if "cache_reset" in self.json_data:
            cache["date"] = timestamp.date()
        return cache

    def _extract_snapshots(self, df: DataFrame) -> list[dict]:
        """Extract snapshots."""
        return [self._extract_snapshot(row) for _, row in df.iterrows()]

    @staticmethod
    def _extract_snapshot(row: pd.Series) -> dict:
        """Extract snapshot."""
        return {
            "timestamp": str(row["timestamp"]),
            "close": row["close"],
            "tickRule": int(row["tickRule"]),
            "volume": row["volume"],
            "notional": row["notional"],
            "ticks": int(row["ticks"]),
            "totalVolume": row["totalVolume"],
            "totalBuyVolume": row["totalBuyVolume"],
            "totalNotional": row["totalNotional"],
            "totalBuyNotional": row["totalBuyNotional"],
            "totalTicks": int(row["totalTicks"]),
            "totalBuyTicks": int(row["totalBuyTicks"]),
        }

    @staticmethod
    def _merge_partial_trades(old: dict, new: dict) -> dict:
        """Merge partial trades."""
        merged = new.copy()
        merged["high"] = max(old["high"], new["high"])
        merged["low"] = min(old["low"], new["low"])
        for col in TOTAL_COLS:
            merged[col] = (old.get(col) or 0) + (new.get(col) or 0)
        return merged

    def get_clusters(self, data_frame: DataFrame, cache_data: dict) -> DataFrame:
        """Get clusters."""
        if "partial_trade" in cache_data and not data_frame.empty:
            partial = cache_data.pop("partial_trade")
            first = data_frame.iloc[0].to_dict()
            merged = self._merge_partial_trades(partial, first)
            data_frame = data_frame.copy()
            merged_df = pd.DataFrame([merged])
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                data_frame = pd.concat(
                    [merged_df, data_frame.iloc[1:]], ignore_index=True
                )
        elif "partial_trade" in cache_data:
            return pd.DataFrame()

        # Pop carry-forward row before clustering
        if not data_frame.empty and data_frame.iloc[-1].get("tickRule") is None:
            new_partial = data_frame.iloc[-1].to_dict()
            data_frame = data_frame.iloc[:-1]

            if data_frame.empty:
                cache_data["partial_trade"] = new_partial
                return pd.DataFrame()

            # Only tickRule=None rows remain: merge into new carry-forward
            if not data_frame["tickRule"].isin([1, -1]).any():
                old_partial = data_frame.iloc[0].to_dict()
                cache_data["partial_trade"] = self._merge_partial_trades(
                    old_partial, new_partial
                )
                return pd.DataFrame()

            cache_data["partial_trade"] = new_partial

        if data_frame.empty:
            return pd.DataFrame()

        clusters = cluster_trades(data_frame)

        if "partial_cluster" in cache_data and not clusters.empty:
            partial = pd.DataFrame([cache_data.pop("partial_cluster")])
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning)
                combined = pd.concat([partial, clusters], ignore_index=True)
            clusters = combine_clustered_trades(combined)

        return clusters

    def _aggregate_clusters(
        self,
        clusters: DataFrame,
        cache_data: dict,
        timestamp_to: datetime,
        sig_filter: Decimal,
    ) -> tuple[list, dict | None]:
        """Inner aggregation loop over clusters."""
        n = self.snapshot_window
        start = 0
        data = []

        if n > 0:
            next_snaps = cache_data.pop("next_snapshots", [])
            prev_snaps = cache_data.pop("prev_snapshots", [])
        else:
            cache_data.pop("next_snapshots", None)
            cache_data.pop("prev_snapshots", None)

        for index, row in clusters.iterrows():
            volume = row.get("volume", Decimal("0"))

            if volume >= sig_filter:
                self._stash_cluster_metadata(row, volume, cache_data)

                df = clusters.loc[start:index]
                if n > 0:
                    if index > start:
                        chunk_snaps = self._extract_snapshots(
                            clusters.loc[start : min(start + n - 1, index - 1)]
                        )
                    else:
                        chunk_snaps = []
                    post_open = (next_snaps + chunk_snaps)[:n]
                    pre_close = prev_snaps[-n:]
                    data.append(
                        self._build_candle(df, cache_data, post_open, pre_close)
                    )
                    next_snaps = []
                    prev_snaps = []
                else:
                    data.append(self._build_candle(df, cache_data))
                start = index + 1
            else:
                if n > 0 and index < len(clusters) - 1:
                    prev_snaps = (prev_snaps + [self._extract_snapshot(row)])[-n:]

        is_last_row = start == len(clusters)
        if not is_last_row:
            cache_data["partial_cluster"] = clusters.iloc[-1].to_dict()
            df = clusters.loc[start : len(clusters) - 2]
            if not df.empty:
                cache_data = get_next_cache(df, cache_data)

            if n > 0:
                leftover_snaps = self._extract_snapshots(
                    clusters.loc[start : len(clusters) - 2]
                )
                cache_data["next_snapshots"] = (next_snaps + leftover_snaps)[:n]
                cache_data["prev_snapshots"] = prev_snaps[-n:]
        return self.get_incomplete_candle(timestamp_to, data, cache_data)

    def _build_candle(
        self,
        df: DataFrame,
        cache_data: dict,
        post_open_snapshots: list[dict] | None = None,
        pre_close_snapshots: list[dict] | None = None,
    ) -> dict:
        """Build candle."""
        candle = aggregate_candle(df)
        candle["clusterTimestamp"] = str(cache_data["cluster_timestamp"])
        candle["clusterTotalSeconds"] = cache_data["cluster_total_seconds"]
        candle["clusterTickRule"] = int(cache_data["cluster_tick_rule"])
        candle["clusterVolume"] = cache_data["cluster_volume"]
        candle["clusterNotional"] = cache_data["cluster_notional"]
        if post_open_snapshots is not None:
            candle["postOpenSnapshots"] = post_open_snapshots
        if pre_close_snapshots is not None:
            candle["preCloseSnapshots"] = pre_close_snapshots
        if "next" in cache_data:
            previous = cache_data.pop("next")
            candle = merge_cache(previous, candle)
        return candle

    def _stash_cluster_metadata(
        self, row: pd.Series, volume: Decimal, cache_data: dict
    ) -> None:
        """Stash triggering cluster metadata into cache."""
        cache_data["cluster_timestamp"] = row.get("timestamp")
        cache_data["cluster_total_seconds"] = row.get("totalSeconds", 0)
        cache_data["cluster_tick_rule"] = int(row.get("tickRule"))
        cache_data["cluster_volume"] = volume
        cache_data["cluster_notional"] = row.get("notional", Decimal("0"))

    def aggregate(
        self,
        timestamp_from: datetime,
        timestamp_to: datetime,
        data_frame: DataFrame,
        cache_data: dict,
    ) -> tuple[list, dict | None]:
        """Aggregate trades into candles triggered by big clusters."""
        clusters = self.get_clusters(data_frame, cache_data)

        if clusters.empty:
            return self.get_incomplete_candle(timestamp_to, [], cache_data)

        sig_filter = self.get_significant_cluster_filter(cache_data)
        return self._aggregate_clusters(clusters, cache_data, timestamp_to, sig_filter)

    def get_incomplete_candle(
        self, timestamp: datetime, data: list, cache_data: dict
    ) -> tuple[list, dict]:
        """Get incomplete candle.

        * Only if cache reset next iteration.
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
