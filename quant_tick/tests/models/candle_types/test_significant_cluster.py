from decimal import Decimal

import pandas as pd
from django.test import TestCase

from quant_tick.constants import FileData, Frequency
from quant_tick.lib import get_current_time
from quant_tick.models import SignificantCluster


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


class SignificantClusterTest(TestCase):
    """Significant cluster test."""

    def _make_candle(self, **overrides: object) -> SignificantCluster:
        """Create candle with defaults."""
        json_data = {
            "source_data": FileData.FILTERED,
            "significant_cluster_filter": Decimal("50000"),
            "snapshot_window": 0,
        }
        json_data.update(overrides)
        return SignificantCluster(json_data=json_data)

    def test_daily_cache_reset(self) -> None:
        """If not same day, daily cache resets."""
        now = get_current_time()
        one_day_ago = now - pd.Timedelta("1d")
        candle = SignificantCluster(
            json_data={
                "source_data": FileData.RAW,
                "cache_reset": Frequency.DAY,
                "significant_cluster_filter": Decimal("50000"),
            }
        )
        cache = candle.get_cache_data(
            now,
            {"date": one_day_ago.date()},
        )
        self.assertEqual(cache, {"date": now.date()})

    def test_daily_cache_does_not_reset(self) -> None:
        """If same day, daily cache does not reset."""
        now = get_current_time()
        candle = SignificantCluster(
            json_data={
                "source_data": FileData.RAW,
                "cache_reset": Frequency.DAY,
                "significant_cluster_filter": Decimal("50000"),
            }
        )
        cache = candle.get_cache_data(
            now,
            {"date": now.date(), "some_key": "some_value"},
        )
        self.assertEqual(cache["date"], now.date())
        self.assertEqual(cache["some_key"], "some_value")

    def test_big_cluster_triggers_candle(self) -> None:
        """Cluster above threshold triggers candle emission."""
        now = get_current_time()
        candle = self._make_candle()
        cache = candle.get_initial_cache(now)

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
        cache = candle.get_initial_cache(now)

        df = pd.DataFrame(
            [
                make_cluster(now, "90000", "500", "500", 1),
                make_cluster(
                    now + pd.Timedelta("1s"), "90100", "800", "800", -1
                ),
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
        cache = candle.get_initial_cache(now)

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

    def test_cluster_tick_rule_matches_trigger(self) -> None:
        """clusterTickRule matches the triggering cluster's direction."""
        now = get_current_time()
        candle = self._make_candle()
        cache = candle.get_initial_cache(now)

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

    def test_multiple_big_clusters_emit_multiple_candles(self) -> None:
        """Two big clusters in one batch emit two candles."""
        now = get_current_time()
        candle = self._make_candle()
        cache = candle.get_initial_cache(now)

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

    def test_trigger_first_cluster_no_post_open_snapshots(self) -> None:
        """When trigger is the first cluster, postOpenSnapshots is empty."""
        now = get_current_time()
        candle = self._make_candle(snapshot_window=25)
        cache = candle.get_initial_cache(now)

        df = pd.DataFrame(
            [make_cluster(now, "90000", "150000", "150000", 1)]
        )
        data, cache = candle.aggregate(
            now, now + pd.Timedelta("1min"), df, cache
        )
        self.assertEqual(len(data), 1)
        self.assertEqual(len(data[0]["postOpenSnapshots"]), 0)
        self.assertEqual(len(data[0]["preCloseSnapshots"]), 0)

    def test_post_open_excludes_trigger(self) -> None:
        """postOpenSnapshots does not include the triggering cluster."""
        now = get_current_time()
        candle = self._make_candle(snapshot_window=25)
        cache = candle.get_initial_cache(now)

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
        # Only 2 small clusters, trigger excluded
        self.assertEqual(len(data[0]["postOpenSnapshots"]), 2)

    def test_pre_close_snapshots(self) -> None:
        """preCloseSnapshots contains small clusters before trigger."""
        now = get_current_time()
        candle = self._make_candle(snapshot_window=25)
        cache = candle.get_initial_cache(now)

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
        # Cluster 0: buy, volume=500, notional=500
        snap = data[0]["preCloseSnapshots"][0]
        self.assertEqual(snap["tickRule"], 1)
        self.assertEqual(snap["volume"], Decimal("500"))
        self.assertEqual(snap["notional"], Decimal("500"))
        self.assertEqual(snap["ticks"], 1)
        self.assertEqual(snap["totalVolume"], Decimal("500"))
        self.assertEqual(snap["totalBuyVolume"], Decimal("500"))
        self.assertEqual(snap["totalNotional"], Decimal("500"))
        self.assertEqual(snap["totalBuyNotional"], Decimal("500"))
        self.assertEqual(snap["totalTicks"], 1)
        self.assertEqual(snap["totalBuyTicks"], 1)
        # Cluster 1: sell, volume=800, notional=800
        snap1 = data[0]["preCloseSnapshots"][1]
        self.assertEqual(snap1["tickRule"], -1)
        self.assertEqual(snap1["volume"], Decimal("800"))
        self.assertEqual(snap1["totalBuyVolume"], Decimal("0"))
        self.assertEqual(snap1["totalBuyNotional"], Decimal("0"))
        self.assertEqual(snap1["totalBuyTicks"], 0)

    def test_pre_close_snapshots_reset_between_candles(self) -> None:
        """preCloseSnapshots reset after each emission."""
        now = get_current_time()
        candle = self._make_candle(snapshot_window=25)
        cache = candle.get_initial_cache(now)

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

