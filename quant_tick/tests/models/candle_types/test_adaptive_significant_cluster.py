import math
from decimal import Decimal

import pandas as pd
from django.test import TestCase

from quant_tick.constants import FileData
from quant_tick.lib import get_current_time
from quant_tick.models import AdaptiveSignificantCluster


def make_cluster(
    timestamp: pd.Timestamp,
    price: str,
    volume: str,
    notional: str,
    tick_rule: int,
) -> dict:
    """Build a cluster row."""
    buy = tick_rule == 1
    return {
        "timestamp": timestamp,
        "nanoseconds": 0,
        "price": Decimal(price),
        "volume": Decimal(volume),
        "notional": Decimal(notional),
        "totalNotional": Decimal(notional),
        "totalVolume": Decimal(volume),
        "totalBuyNotional": Decimal(notional) if buy else Decimal("0"),
        "totalBuyVolume": Decimal(volume) if buy else Decimal("0"),
        "totalTicks": 1,
        "totalBuyTicks": 1 if buy else 0,
        "ticks": 1,
        "tickRule": tick_rule,
    }


class AdaptiveSignificantClusterTest(TestCase):
    """Adaptive significant cluster tests."""

    def _make_candle(self, **overrides: object) -> AdaptiveSignificantCluster:
        """Create candle with defaults."""
        json_data = {
            "source_data": FileData.FILTERED,
            "halflife_days": 1,
            "significant_cluster_percentile": 99,
            "min_warmup_samples": 100,
            "snapshot_window": 0,
        }
        json_data.update(overrides)
        return AdaptiveSignificantCluster(json_data=json_data)

    def _warm_cache(
        self, candle: AdaptiveSignificantCluster, now: pd.Timestamp
    ) -> dict:
        """Create a warm cache with EMA data.

        Sets EMA state representing a log-normal distribution of cluster
        notionals ~100-1100. The z99 threshold lands around ~1680, so
        clusters at 500/800 don't trigger but 150000 does.
        """
        cache = candle.get_initial_cache(now - pd.Timedelta("2d"))
        # mean=5.8 ≈ log(330), var=0.49 (std=0.7)
        # threshold = exp(5.8 + 2.326 * 0.7) ≈ 1681
        cache["ema_log_mean"] = 5.8
        cache["ema_log_var"] = 0.49
        cache["ema_count"] = 1000
        cache["ema_timestamp"] = str(now - pd.Timedelta("1s"))
        return cache

    def test_big_cluster_triggers_candle(self) -> None:
        """Cluster above threshold triggers candle emission."""
        now = get_current_time()
        candle = self._make_candle()
        cache = self._warm_cache(candle, now)

        df = pd.DataFrame(
            [
                make_cluster(now, "90000", "500", "500", 1),
                make_cluster(
                    now + pd.Timedelta("1s"), "90100", "150000", "150000", 1
                ),
            ]
        )
        data, cache = candle.aggregate(
            now, now + pd.Timedelta("1min"), df, cache
        )
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["clusterTickRule"], 1)
        self.assertIn("clusterTimestamp", data[0])
        self.assertIn("clusterTotalSeconds", data[0])
        self.assertEqual(data[0]["clusterVolume"], Decimal("150500"))
        self.assertEqual(data[0]["clusterNotional"], Decimal("150500"))

    def test_small_clusters_do_not_emit(self) -> None:
        """Clusters below threshold do not trigger candle."""
        now = get_current_time()
        candle = self._make_candle()
        cache = self._warm_cache(candle, now)

        df = pd.DataFrame(
            [
                make_cluster(now, "90000", "500", "500", 1),
                make_cluster(now + pd.Timedelta("1s"), "90100", "800", "800", -1),
            ]
        )
        data, cache = candle.aggregate(
            now, now + pd.Timedelta("1min"), df, cache
        )
        self.assertEqual(len(data), 0)

    def test_candle_aggregates_since_last_emit(self) -> None:
        """Candle includes all clusters from start to triggering cluster."""
        now = get_current_time()
        candle = self._make_candle()
        cache = self._warm_cache(candle, now)

        df = pd.DataFrame(
            [
                make_cluster(now, "90000", "500", "500", 1),
                make_cluster(
                    now + pd.Timedelta("1s"), "90100", "800", "800", -1
                ),
                make_cluster(
                    now + pd.Timedelta("2s"), "90200", "150000", "150000", 1
                ),
            ]
        )
        data, cache = candle.aggregate(
            now, now + pd.Timedelta("1min"), df, cache
        )
        self.assertEqual(len(data), 1)
        # All 3 clusters aggregated (volume = 500 + 800 + 150000)
        self.assertEqual(data[0]["volume"], Decimal("151300"))

    def test_cluster_percentile_in_candle(self) -> None:
        """Candle includes clusterPercentile field."""
        now = get_current_time()
        candle = self._make_candle()
        cache = self._warm_cache(candle, now)

        df = pd.DataFrame(
            [make_cluster(now, "90000", "150000", "150000", 1)]
        )
        data, cache = candle.aggregate(
            now, now + pd.Timedelta("1min"), df, cache
        )
        self.assertEqual(len(data), 1)
        self.assertIn("clusterPercentile", data[0])
        self.assertGreater(data[0]["clusterPercentile"], 90)

    def test_cluster_tick_rule_matches_trigger(self) -> None:
        """clusterTickRule matches the triggering cluster's direction."""
        now = get_current_time()
        candle = self._make_candle()
        cache = self._warm_cache(candle, now)

        df = pd.DataFrame(
            [
                make_cluster(now, "90000", "500", "500", 1),
                make_cluster(
                    now + pd.Timedelta("1s"), "89900", "150000", "150000", -1
                ),
            ]
        )
        data, cache = candle.aggregate(
            now, now + pd.Timedelta("1min"), df, cache
        )
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["clusterTickRule"], -1)

    def test_warmup_no_emission(self) -> None:
        """During warmup period, no candles are emitted."""
        now = get_current_time()
        candle = self._make_candle()
        cache = candle.get_initial_cache(now)

        df = pd.DataFrame(
            [make_cluster(now, "90000", "150000", "150000", 1)]
        )
        data, cache = candle.aggregate(
            now, now + pd.Timedelta("1min"), df, cache
        )
        self.assertEqual(len(data), 0)

    def test_multiple_big_clusters_emit_multiple_candles(self) -> None:
        """Two big clusters in one batch emit two candles."""
        now = get_current_time()
        candle = self._make_candle()
        cache = self._warm_cache(candle, now)

        df = pd.DataFrame(
            [
                make_cluster(now, "90000", "500", "500", 1),
                make_cluster(
                    now + pd.Timedelta("1s"), "90100", "150000", "150000", 1
                ),
                make_cluster(
                    now + pd.Timedelta("2s"), "89900", "600", "600", -1
                ),
                make_cluster(
                    now + pd.Timedelta("3s"), "89800", "200000", "200000", -1
                ),
            ]
        )
        data, cache = candle.aggregate(
            now, now + pd.Timedelta("1min"), df, cache
        )
        self.assertEqual(len(data), 2)
        # First: clusters 0+1 (volume 500 + 150000)
        self.assertEqual(data[0]["volume"], Decimal("150500"))
        self.assertEqual(data[0]["clusterTickRule"], 1)
        # Second: clusters 2+3 (volume 600 + 200000)
        self.assertEqual(data[1]["volume"], Decimal("200600"))
        self.assertEqual(data[1]["clusterTickRule"], -1)

    def test_significant_cluster_filter_in_candle(self) -> None:
        """Candle includes significantClusterFilter field."""
        now = get_current_time()
        candle = self._make_candle()
        cache = self._warm_cache(candle, now)

        df = pd.DataFrame(
            [make_cluster(now, "90000", "150000", "150000", 1)]
        )
        data, cache = candle.aggregate(
            now, now + pd.Timedelta("1min"), df, cache
        )
        self.assertEqual(len(data), 1)
        self.assertIn("significantClusterFilter", data[0])
        self.assertIsInstance(data[0]["significantClusterFilter"], Decimal)

    def test_pre_close_snapshots(self) -> None:
        """preCloseSnapshots contains small clusters before trigger."""
        now = get_current_time()
        candle = self._make_candle(snapshot_window=25)
        cache = self._warm_cache(candle, now)

        df = pd.DataFrame(
            [
                make_cluster(now, "90000", "500", "500", 1),
                make_cluster(
                    now + pd.Timedelta("1s"), "90100", "800", "800", -1
                ),
                make_cluster(
                    now + pd.Timedelta("2s"), "90200", "150000", "150000", 1
                ),
            ]
        )
        data, cache = candle.aggregate(
            now, now + pd.Timedelta("1min"), df, cache
        )
        self.assertEqual(len(data), 1)
        self.assertIn("preCloseSnapshots", data[0])
        # Clusters 0 and 1 are small, appear as preCloseSnapshots
        self.assertEqual(len(data[0]["preCloseSnapshots"]), 2)

    def test_pre_close_snapshots_reset_between_candles(self) -> None:
        """preCloseSnapshots reset after each emission."""
        now = get_current_time()
        candle = self._make_candle(snapshot_window=25)
        cache = self._warm_cache(candle, now)

        # Alternating directions to prevent cluster merging
        df = pd.DataFrame(
            [
                make_cluster(now, "90000", "500", "500", 1),
                make_cluster(
                    now + pd.Timedelta("1s"), "90100", "150000", "150000", -1
                ),
                make_cluster(
                    now + pd.Timedelta("2s"), "89900", "600", "600", 1
                ),
                make_cluster(
                    now + pd.Timedelta("3s"), "89800", "200000", "200000", -1
                ),
            ]
        )
        data, cache = candle.aggregate(
            now, now + pd.Timedelta("1min"), df, cache
        )
        self.assertEqual(len(data), 2)
        # First candle: cluster 0 (500) is small before trigger (150000)
        self.assertEqual(len(data[0]["preCloseSnapshots"]), 1)
        # Second candle: cluster 2 (600) is small before trigger (200000)
        self.assertEqual(len(data[1]["preCloseSnapshots"]), 1)

    def test_ema_convergence(self) -> None:
        """EMA converges to expected mean after many identical samples."""
        candle = self._make_candle()
        cache = candle.get_initial_cache(get_current_time())
        base_ts = pd.Timestamp("2024-01-01")
        for i in range(500):
            ts = base_ts + pd.Timedelta(seconds=i)
            candle._update_ema(cache, 1000.0, ts)
        self.assertAlmostEqual(
            cache["ema_log_mean"], math.log(1000.0), places=2
        )

    def test_z_threshold_from_percentile(self) -> None:
        """z_threshold derived from p99 is approximately 2.326."""
        candle = self._make_candle(significant_cluster_percentile=99)
        self.assertAlmostEqual(candle.z_threshold, 2.326, places=2)

    def test_old_cache_migration(self) -> None:
        """Old histogram-based cache is migrated on first aggregate."""
        now = get_current_time()
        candle = self._make_candle()
        cache = candle.get_initial_cache(now)
        cache["hourly_data"] = {"key": {"notional_hist": {}}}
        cache["first_cluster_timestamp"] = now

        df = pd.DataFrame(
            [make_cluster(now, "90000", "500", "500", 1)]
        )
        data, cache = candle.aggregate(
            now, now + pd.Timedelta("1min"), df, cache
        )
        self.assertNotIn("hourly_data", cache)
        self.assertIn("ema_log_mean", cache)
        self.assertEqual(cache["ema_count"], 1)

    def test_cluster_percentile_from_zscore(self) -> None:
        """Cluster at the mean gets ~50th percentile."""
        candle = self._make_candle()
        cache = candle.get_initial_cache(get_current_time())
        cache["ema_log_mean"] = math.log(1000.0)
        cache["ema_log_var"] = 1.0
        cache["ema_count"] = 200
        pct = candle._compute_cluster_percentile(cache, 1000.0)
        self.assertAlmostEqual(pct, 50.0, places=0)

    def test_percentile_zero_raises(self) -> None:
        """Percentile of 0 or 100 raises ValueError."""
        with self.assertRaises(ValueError):
            _ = self._make_candle(significant_cluster_percentile=0).z_threshold
        with self.assertRaises(ValueError):
            _ = self._make_candle(significant_cluster_percentile=100).z_threshold

    def test_halflife_zero_raises(self) -> None:
        """Zero halflife raises ValueError."""
        with self.assertRaises(ValueError):
            _ = self._make_candle(halflife_days=0).halflife_days
