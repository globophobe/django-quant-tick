from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal

import numpy as np
import pandas as pd
from pandas import DataFrame

from quant_tick.lib import aggregate_candle, get_next_cache, merge_cache
from quant_tick.utils import gettext_lazy as _

from .significant_cluster import SignificantCluster

RELATIVE_PRECISION = 0.02
LOG_BASE = 1 + RELATIVE_PRECISION


class AdaptiveSignificantCluster(SignificantCluster):
    """SignificantCluster with adaptive thresholds based on recent percentiles.

    Instead of fixed thresholds, adapts to recent market conditions:
    - significant_cluster_filter = Nth percentile of recent cluster sizes
    - target_cumulative_notional = Mth percentile of recent cumulative runs

    Uses log-spaced histogram bins for efficient percentile computation.

    json_data config:
    - source_data: FileData.FILTERED
    - moving_average_number_of_days: 7 (lookback for percentile calculation)
    - significant_cluster_percentile: 99 (top 1% of clusters count)
    - target_cumulative_percentile: 90 (trigger at 90th percentile of runs)
    - cache_reset: Optional Frequency.DAY or Frequency.WEEK
    """

    @property
    def significant_cluster_percentile(self) -> float:
        """Percentile for significant cluster filter."""
        if self.json_data and self.json_data.get("significant_cluster_percentile"):
            return float(self.json_data["significant_cluster_percentile"])
        return 99.0

    @property
    def target_cumulative_percentile(self) -> float:
        """Percentile for target cumulative threshold."""
        if self.json_data and self.json_data.get("target_cumulative_percentile"):
            return float(self.json_data["target_cumulative_percentile"])
        return 90.0

    def get_initial_cache(self, timestamp: datetime) -> dict:
        """Get initial cache."""
        cache = super().get_initial_cache(timestamp)
        cache["significant_cluster_filter"] = None
        cache["target_cumulative_notional"] = None
        return cache

    def is_warmup_complete(self, timestamp: datetime, cache_data: dict) -> bool:
        """Check if we have enough days of data."""
        first_ts = cache_data.get("first_cluster_timestamp")
        if first_ts is None:
            return False
        days_elapsed = (timestamp - first_ts).days
        return days_elapsed >= self.json_data["moving_average_number_of_days"]

    def get_target_cumulative_notional(self, cache_data: dict) -> Decimal:
        """Get target cumulative notional from cache."""
        return cache_data["target_cumulative_notional"]

    def get_significant_cluster_filter(self, cache_data: dict) -> Decimal:
        """Get significant cluster filter from cache."""
        return cache_data["significant_cluster_filter"]

    def _get_bin_index(self, value: float) -> int | None:
        """Get histogram bin index for a value."""
        if value <= 0:
            return None
        return int(np.floor(np.log(value) / np.log(LOG_BASE)))

    def _add_to_histogram(self, histogram: dict, value: float) -> None:
        """Add value to histogram."""
        bin_idx = self._get_bin_index(value)
        if bin_idx is not None:
            histogram[bin_idx] = histogram.get(bin_idx, 0) + 1

    def _percentile_from_histogram(
        self, histogram: dict, percentile: float
    ) -> float | None:
        """Compute percentile from histogram."""
        if not histogram:
            return None
        int_hist = {int(k): v for k, v in histogram.items()}
        total = sum(int_hist.values())
        target_count = total * percentile / 100.0
        cumulative = 0
        for bin_idx in sorted(int_hist.keys()):
            cumulative += int_hist[bin_idx]
            if cumulative >= target_count:
                return LOG_BASE ** (bin_idx + 0.5)
        return LOG_BASE ** (max(int_hist.keys()) + 0.5)

    def _merge_histograms(self, histograms: list[dict]) -> dict:
        """Merge multiple histograms."""
        merged = {}
        for h in histograms:
            for k, count in h.items():
                bin_idx = int(k)
                merged[bin_idx] = merged.get(bin_idx, 0) + count
        return merged

    def _compute_run_totals_and_histogram(
        self,
        clusters: DataFrame,
        sig_filter: float | None,
        run_hist: dict,
        cache_data: dict,
        timestamp_from: datetime,
        cutoff_ts: str,
    ) -> None:
        """Compute run totals matching runtime logic, add to histogram."""
        if sig_filter is None:
            return

        run_state = cache_data.setdefault(
            "run_state", {"dir": None, "run": 0.0, "start_ts": None}
        )

        if run_state.get("start_ts") and run_state["start_ts"] < cutoff_ts:
            run_state["dir"] = None
            run_state["run"] = 0.0
            run_state["start_ts"] = None

        for idx, row in clusters.iterrows():
            tick_rule = row.get("tickRule")
            notional = float(row.get("totalNotional", 0))
            row_ts = row["timestamp"].isoformat()

            if run_state["dir"] is None:
                run_state["dir"] = tick_rule
                run_state["start_ts"] = row_ts
                if notional >= sig_filter:
                    run_state["run"] = notional
            elif tick_rule == run_state["dir"]:
                if notional >= sig_filter:
                    run_state["run"] += notional
            else:
                if run_state["run"] > 0:
                    if run_state.get("start_ts") and run_state["start_ts"] >= cutoff_ts:
                        self._add_to_histogram(run_hist, run_state["run"])
                run_state["dir"] = tick_rule
                run_state["start_ts"] = row_ts
                if notional >= sig_filter:
                    run_state["run"] = notional
                else:
                    run_state["run"] = 0.0

    def aggregate(
        self,
        timestamp_from: datetime,
        timestamp_to: datetime,
        data_frame: DataFrame,
        cache_data: dict,
    ) -> tuple[list, dict | None]:
        """Aggregate trades with adaptive thresholds."""
        clusters = self.get_clusters(data_frame, cache_data)

        if "first_cluster_timestamp" not in cache_data and not clusters.empty:
            cache_data["first_cluster_timestamp"] = timestamp_from

        if clusters.empty:
            return self.get_incomplete_candle(timestamp_to, [], cache_data)

        hour_key = timestamp_from.strftime("%Y-%m-%d-%H")
        hourly_data = cache_data.setdefault("hourly_data", {})
        if hour_key not in hourly_data:
            hourly_data[hour_key] = {"notional_hist": {}, "run_hist": {}}

        for idx, row in clusters.iterrows():
            notional = float(row.get("totalNotional", 0))
            if notional > 0:
                self._add_to_histogram(
                    hourly_data[hour_key]["notional_hist"], notional
                )

        days = self.json_data["moving_average_number_of_days"]
        cutoff = timestamp_from - pd.Timedelta(f"{days}d")
        cutoff_key = cutoff.strftime("%Y-%m-%d-%H")
        cutoff_ts = cutoff.isoformat()
        cache_data["hourly_data"] = {
            k: v for k, v in hourly_data.items() if k >= cutoff_key
        }
        hourly_data = cache_data["hourly_data"]

        all_notional_hist = self._merge_histograms(
            [h["notional_hist"] for h in hourly_data.values()]
        )
        current_sig = self._percentile_from_histogram(
            all_notional_hist, self.significant_cluster_percentile
        )

        self._compute_run_totals_and_histogram(
            clusters,
            current_sig,
            hourly_data[hour_key]["run_hist"],
            cache_data,
            timestamp_from,
            cutoff_ts,
        )

        start = 0
        data = []

        if self.is_warmup_complete(timestamp_from, cache_data):
            sig_filter, target = self.compute_adaptive_thresholds(cache_data)
            if sig_filter is not None and target is not None:
                cache_data["significant_cluster_filter"] = sig_filter
                cache_data["target_cumulative_notional"] = target

                for index, row in clusters.iterrows():
                    tick_rule = row.get("tickRule")
                    notional = row.get(
                        "totalNotional", row.get("notional", Decimal("0"))
                    )

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

    def compute_adaptive_thresholds(
        self, cache_data: dict
    ) -> tuple[Decimal | None, Decimal | None]:
        """Compute thresholds from pooled histograms."""
        hourly_data = cache_data.get("hourly_data", {})
        if not hourly_data:
            return None, None

        all_notional_hist = self._merge_histograms(
            [h["notional_hist"] for h in hourly_data.values()]
        )
        all_run_hist = self._merge_histograms(
            [h["run_hist"] for h in hourly_data.values()]
        )

        sig_raw = self._percentile_from_histogram(
            all_notional_hist, self.significant_cluster_percentile
        )
        target_raw = self._percentile_from_histogram(
            all_run_hist, self.target_cumulative_percentile
        )

        if sig_raw is None or target_raw is None:
            return None, None

        sig_filter = Decimal(str(sig_raw)).quantize(
            Decimal("0.1"), rounding=ROUND_HALF_UP
        )
        target = Decimal(str(target_raw)).quantize(
            Decimal("0.1"), rounding=ROUND_HALF_UP
        )

        return sig_filter, target

    class Meta:
        proxy = True
        verbose_name = _("adaptive significant cluster")
        verbose_name_plural = _("adaptive significant clusters")
