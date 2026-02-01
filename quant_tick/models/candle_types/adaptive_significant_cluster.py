import math
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal

import pandas as pd
from pandas import DataFrame
from scipy.stats import norm

from quant_tick.utils import gettext_lazy as _

from .significant_cluster import SignificantCluster

DEFAULT_HALFLIFE_DAYS = 90
DEFAULT_MIN_WARMUP_SAMPLES = 100


class AdaptiveSignificantCluster(SignificantCluster):
    """Adaptive Significant Cluster.

    Emits a candle every time a cluster passes the adaptive significant cluster
    filter threshold. The candle aggregates all clusters since the last emission
    up to and including the triggering cluster.

    The threshold adapts to recent market conditions using EMA z-score on
    log(volume). A cluster triggers when its log(volume) exceeds
    ema_log_mean + k * sqrt(ema_log_var), where k is derived from the
    configured percentile.

    Each candle includes clusterPercentile, the triggering cluster's percentile
    rank from the normal CDF of its z-score, allowing downstream filtering at
    stricter thresholds.

    json_data config:
    - source_data: FileData.FILTERED
    - halflife_days: EMA halflife in calendar days (default 90)
    - significant_cluster_percentile: 99
    - min_warmup_samples: Minimum clusters before emission (default 100)
    - cache_reset: Optional Frequency.DAY or Frequency.WEEK
    """

    @property
    def significant_cluster_percentile(self) -> float:
        """Significant cluster percentile."""
        if self.json_data and "significant_cluster_percentile" in self.json_data:
            value = float(self.json_data["significant_cluster_percentile"])
            if value <= 0 or value >= 100:
                msg = f"significant_cluster_percentile must be in (0, 100), got {value}"
                raise ValueError(msg)
            return value
        return 99.0

    @property
    def halflife_days(self) -> float:
        """EMA halflife in calendar days."""
        if self.json_data and "halflife_days" in self.json_data:
            value = float(self.json_data["halflife_days"])
        else:
            value = float(DEFAULT_HALFLIFE_DAYS)
        if value <= 0:
            msg = f"halflife_days must be positive, got {value}"
            raise ValueError(msg)
        return value

    @property
    def z_threshold(self) -> float:
        """Z-score threshold derived from percentile."""
        return float(norm.ppf(self.significant_cluster_percentile / 100.0))

    @property
    def min_warmup_samples(self) -> int:
        """Minimum clusters before warmup is complete."""
        if self.json_data and "min_warmup_samples" in self.json_data:
            return int(self.json_data["min_warmup_samples"])
        return DEFAULT_MIN_WARMUP_SAMPLES

    def get_initial_cache(self, timestamp: datetime) -> dict:
        """Get initial cache."""
        cache = {}
        if "cache_reset" in self.json_data:
            cache["date"] = timestamp.date()
        cache["significant_cluster_filter"] = None
        cache["ema_log_mean"] = None
        cache["ema_log_var"] = 0.0
        cache["ema_count"] = 0
        cache["ema_timestamp"] = None
        return cache

    def is_warmup_complete(self, cache_data: dict) -> bool:
        """Is warmup complete?"""
        return cache_data.get("ema_count", 0) >= self.min_warmup_samples

    def _update_ema(self, cache_data: dict, volume: float, timestamp: datetime) -> None:
        """Update EMA statistics with a new cluster volume."""
        if volume <= 0:
            return
        x = math.log(volume)
        if cache_data["ema_log_mean"] is None:
            cache_data["ema_log_mean"] = x
            cache_data["ema_log_var"] = 0.0
            cache_data["ema_timestamp"] = str(timestamp)
        else:
            prev_ts = pd.Timestamp(cache_data["ema_timestamp"])
            dt_seconds = max((timestamp - prev_ts).total_seconds(), 1.0)
            halflife_seconds = self.halflife_days * 86400
            alpha = 1.0 - math.exp(-math.log(2) * dt_seconds / halflife_seconds)
            delta = x - cache_data["ema_log_mean"]
            cache_data["ema_log_mean"] += alpha * delta
            cache_data["ema_log_var"] = max(
                0.0,
                (1.0 - alpha) * (cache_data["ema_log_var"] + alpha * delta * delta),
            )
            cache_data["ema_timestamp"] = str(max(timestamp, prev_ts))
        cache_data["ema_count"] = cache_data.get("ema_count", 0) + 1

    def _compute_cluster_percentile(self, cache_data: dict, volume: float) -> float:
        """Compute percentile rank of a cluster using normal CDF of z-score."""
        mean = cache_data.get("ema_log_mean")
        var = cache_data.get("ema_log_var", 0.0)
        if mean is None:
            return 100.0
        x = math.log(volume)
        if var <= 0:
            if x > mean:
                return 99.9
            if x < mean:
                return 0.1
            return 50.0
        z = (x - mean) / math.sqrt(var)
        return float(norm.cdf(z)) * 100.0

    def _stash_cluster_metadata(
        self, row: pd.Series, volume: Decimal, cache_data: dict
    ) -> None:
        """Stash triggering cluster metadata into cache."""
        super()._stash_cluster_metadata(row, volume, cache_data)
        cache_data["cluster_percentile"] = self._compute_cluster_percentile(
            cache_data, float(volume)
        )

    def _build_candle(
        self,
        df: DataFrame,
        cache_data: dict,
        post_open_snapshots: list[dict] | None = None,
        pre_close_snapshots: list[dict] | None = None,
    ) -> dict:
        """Build a candle from clustered trades."""
        candle = super()._build_candle(
            df, cache_data, post_open_snapshots, pre_close_snapshots
        )
        candle["significantClusterFilter"] = cache_data["significant_cluster_filter"]
        candle["clusterPercentile"] = cache_data["cluster_percentile"]
        return candle

    def compute_adaptive_threshold(self, cache_data: dict) -> Decimal | None:
        """Compute threshold from EMA z-score."""
        mean = cache_data.get("ema_log_mean")
        if mean is None:
            return None
        var = cache_data.get("ema_log_var", 0.0)
        if var <= 0:
            raw = math.exp(mean)
        else:
            raw = math.exp(mean + self.z_threshold * math.sqrt(var))
        return Decimal(str(raw)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)

    def aggregate(
        self,
        timestamp_from: datetime,
        timestamp_to: datetime,
        data_frame: DataFrame,
        cache_data: dict,
    ) -> tuple[list, dict | None]:
        """Aggregate trades into candles triggered by big clusters."""
        if "hourly_data" in cache_data:
            cache_data.pop("hourly_data", None)
            cache_data.pop("first_cluster_timestamp", None)
            cache_data.setdefault("ema_log_mean", None)
            cache_data.setdefault("ema_log_var", 0.0)
            cache_data.setdefault("ema_count", 0)
            cache_data.setdefault("ema_timestamp", None)

        clusters = self.get_clusters(data_frame, cache_data)

        if clusters.empty:
            return self.get_incomplete_candle(timestamp_to, [], cache_data)

        for idx, row in clusters.iterrows():
            volume = float(row.get("volume", 0))
            ts = row.get("timestamp", timestamp_from)
            self._update_ema(cache_data, volume, ts)

        if not self.is_warmup_complete(cache_data):
            return self._aggregate_clusters(
                clusters, cache_data, timestamp_to, Decimal("Infinity")
            )

        sig_filter = self.compute_adaptive_threshold(cache_data)
        if sig_filter is None:
            return self._aggregate_clusters(
                clusters, cache_data, timestamp_to, Decimal("Infinity")
            )

        cache_data["significant_cluster_filter"] = sig_filter
        return self._aggregate_clusters(
            clusters, cache_data, timestamp_to, sig_filter
        )

    class Meta:
        proxy = True
        verbose_name = _("adaptive significant cluster")
        verbose_name_plural = _("adaptive significant clusters")
