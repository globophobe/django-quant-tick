from decimal import Decimal

import pandas as pd
from django.test import TestCase

from quant_tick.constants import FileData
from quant_tick.models import TimeBasedClusterCandle


def make_trade(
    timestamp: pd.Timestamp,
    price: str,
    volume: str,
    notional: str,
    tick_rule: int,
) -> dict:
    """Build a trade row for clustering."""
    return {
        "uid": f"t{timestamp}",
        "timestamp": timestamp,
        "nanoseconds": 0,
        "price": Decimal(price),
        "volume": Decimal(volume),
        "notional": Decimal(notional),
        "ticks": 1,
        "tickRule": tick_rule,
    }


class TimeBasedClusterCandleTest(TestCase):
    """Time based cluster candle tests."""

    def _make_candle(self, **overrides: object) -> TimeBasedClusterCandle:
        """Create candle with defaults."""
        json_data = {
            "source_data": FileData.FILTERED,
            "window": "1h",
        }
        json_data.update(overrides)
        return TimeBasedClusterCandle(json_data=json_data)

    def test_emits_at_time_intervals(self) -> None:
        """Candles emitted at window boundaries."""
        now = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
        candle = self._make_candle(window="1h")

        df = pd.DataFrame([
            make_trade(now + pd.Timedelta("30min"), "100", "1000", "100000", 1),
            make_trade(now + pd.Timedelta("90min"), "101", "1000", "101000", -1),
        ])

        data, cache = candle.aggregate(
            now, now + pd.Timedelta("2h"), df, {}
        )

        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]["timestamp"], now)
        self.assertEqual(data[1]["timestamp"], now + pd.Timedelta("1h"))

    def test_percentile_buckets_present(self) -> None:
        """All percentile buckets present in output."""
        now = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
        candle = self._make_candle(window="1h")

        df = pd.DataFrame([
            make_trade(now + pd.Timedelta("10min"), "100", "1000", "100000", 1),
            make_trade(now + pd.Timedelta("20min"), "101", "2000", "202000", -1),
            make_trade(now + pd.Timedelta("30min"), "99", "3000", "297000", 1),
            make_trade(now + pd.Timedelta("40min"), "98", "4000", "392000", -1),
        ])

        data, cache = candle.aggregate(
            now, now + pd.Timedelta("1h"), df, {}
        )

        self.assertEqual(len(data), 1)
        for p in [0, 25, 50, 75, 90, 95, 99]:
            key = f"p{p}"
            self.assertIn(key, data[0])
            self.assertIn("buyClusterCount", data[0][key])
            self.assertIn("totalClusterCount", data[0][key])
            self.assertIn("buyClusterNotional", data[0][key])
            self.assertIn("totalClusterNotional", data[0][key])

    def test_total_cluster_count_from_buckets(self) -> None:
        """Total cluster count equals sum of bucket counts."""
        now = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
        candle = self._make_candle(window="1h")

        df = pd.DataFrame([
            make_trade(now + pd.Timedelta("10min"), "100", "1000", "100000", 1),
            make_trade(now + pd.Timedelta("15min"), "99", "2000", "198000", -1),
            make_trade(now + pd.Timedelta("20min"), "101", "3000", "303000", 1),
            make_trade(now + pd.Timedelta("25min"), "98", "4000", "392000", -1),
        ])

        data, cache = candle.aggregate(
            now, now + pd.Timedelta("1h"), df, {}
        )

        self.assertEqual(len(data), 1)
        total = sum(data[0][f"p{p}"]["totalClusterCount"] for p in [0, 25, 50, 75, 90, 95, 99])
        self.assertEqual(total, 4)

    def test_buy_sell_in_buckets(self) -> None:
        """Buy/sell breakdown correct in percentile buckets."""
        now = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
        candle = self._make_candle(window="1h")

        df = pd.DataFrame([
            make_trade(now + pd.Timedelta("10min"), "100", "1000", "100000", 1),
            make_trade(now + pd.Timedelta("15min"), "99", "2000", "198000", -1),
            make_trade(now + pd.Timedelta("20min"), "101", "3000", "303000", 1),
            make_trade(now + pd.Timedelta("25min"), "98", "4000", "392000", -1),
        ])

        data, cache = candle.aggregate(
            now, now + pd.Timedelta("1h"), df, {}
        )

        self.assertEqual(len(data), 1)
        total_buy = sum(data[0][f"p{p}"]["buyClusterCount"] for p in [0, 25, 50, 75, 90, 95, 99])
        total_sell = sum(
            data[0][f"p{p}"]["totalClusterCount"] - data[0][f"p{p}"]["buyClusterCount"]
            for p in [0, 25, 50, 75, 90, 95, 99]
        )
        self.assertEqual(total_buy, 2)
        self.assertEqual(total_sell, 2)

    def test_notional_in_buckets(self) -> None:
        """Notional values present in buckets."""
        now = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
        candle = self._make_candle(window="1h")

        df = pd.DataFrame([
            make_trade(now + pd.Timedelta("10min"), "100", "1000", "100000", 1),
            make_trade(now + pd.Timedelta("20min"), "99", "2000", "198000", -1),
        ])

        data, cache = candle.aggregate(
            now, now + pd.Timedelta("1h"), df, {}
        )

        self.assertEqual(len(data), 1)
        total_notional = sum(
            data[0][f"p{p}"]["totalClusterNotional"] for p in [0, 25, 50, 75, 90, 95, 99]
        )
        self.assertEqual(total_notional, 298000.0)

    def test_empty_window(self) -> None:
        """Window with no trades handled."""
        now = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
        candle = self._make_candle(window="1h")

        df = pd.DataFrame([
            make_trade(now + pd.Timedelta("90min"), "100", "1000", "100000", 1),
        ])

        data, cache = candle.aggregate(
            now, now + pd.Timedelta("2h"), df, {}
        )

        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["timestamp"], now + pd.Timedelta("1h"))

    def test_ohlcv_fields_present(self) -> None:
        """Standard OHLCV fields present."""
        now = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
        candle = self._make_candle(window="1h")

        df = pd.DataFrame([
            make_trade(now + pd.Timedelta("10min"), "100", "1000", "100000", 1),
            make_trade(now + pd.Timedelta("20min"), "105", "1000", "105000", 1),
            make_trade(now + pd.Timedelta("30min"), "95", "1000", "95000", -1),
        ])

        data, cache = candle.aggregate(
            now, now + pd.Timedelta("1h"), df, {}
        )

        self.assertEqual(len(data), 1)
        self.assertIn("open", data[0])
        self.assertIn("high", data[0])
        self.assertIn("low", data[0])
        self.assertIn("close", data[0])
        self.assertIn("volume", data[0])

    def test_daily_window(self) -> None:
        """Daily window works correctly."""
        now = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
        candle = self._make_candle(window="1d")

        df = pd.DataFrame([
            make_trade(now + pd.Timedelta("6h"), "100", "1000", "100000", 1),
            make_trade(now + pd.Timedelta("30h"), "105", "1000", "105000", -1),
        ])

        data, cache = candle.aggregate(
            now, now + pd.Timedelta("2d"), df, {}
        )

        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]["timestamp"], now)
        self.assertEqual(data[1]["timestamp"], now + pd.Timedelta("1d"))

    def test_single_cluster_all_in_p99(self) -> None:
        """Single cluster ends up in highest percentile bucket."""
        now = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
        candle = self._make_candle(window="1h")

        df = pd.DataFrame([
            make_trade(now + pd.Timedelta("10min"), "100", "1000", "100000", 1),
        ])

        data, cache = candle.aggregate(
            now, now + pd.Timedelta("1h"), df, {}
        )

        self.assertEqual(len(data), 1)
        total = sum(data[0][f"p{p}"]["totalClusterCount"] for p in [0, 25, 50, 75, 90, 95, 99])
        self.assertEqual(total, 1)

    def test_clusters_separate_across_windows(self) -> None:
        """Clusters in different windows stay separate."""
        now = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
        candle = self._make_candle(window="1h")

        # Buy and sell trades in different hours (different directions = separate clusters)
        df = pd.DataFrame([
            make_trade(now + pd.Timedelta("30min"), "100", "1000", "100000", 1),
            make_trade(now + pd.Timedelta("90min"), "101", "2000", "202000", -1),
        ])

        data, cache = candle.aggregate(now, now + pd.Timedelta("2h"), df, {})

        self.assertEqual(len(data), 2)
        # First hour has volume 1000, second hour has volume 2000
        self.assertEqual(float(data[0]["volume"]), 1000.0)
        self.assertEqual(float(data[1]["volume"]), 2000.0)

    def test_partial_cluster_merged_same_window(self) -> None:
        """Partial cluster merged with next batch if same window and direction."""
        now = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
        candle = self._make_candle(window="1d")

        # Batch 1: buy at hour 1 (incomplete day)
        df1 = pd.DataFrame([
            make_trade(now + pd.Timedelta("1h"), "100", "1000", "100000", 1),
        ])
        data1, cache = candle.aggregate(now, now + pd.Timedelta("2h"), df1, {})
        self.assertEqual(len(data1), 0)  # No complete windows
        self.assertIn("partial_cluster", cache)

        # Batch 2: buy at hour 3 (same day, same direction)
        df2 = pd.DataFrame([
            make_trade(now + pd.Timedelta("3h"), "101", "2000", "202000", 1),
        ])
        data2, cache2 = candle.aggregate(
            now + pd.Timedelta("2h"), now + pd.Timedelta("4h"), df2, cache
        )

        # Should still be partial (day not complete)
        self.assertEqual(len(data2), 0)
        self.assertIn("partial_cluster", cache2)
        # Merged cluster has combined volume
        self.assertEqual(float(cache2["partial_cluster"]["volume"]), 3000.0)

    def test_partial_cluster_flushed_at_window_boundary(self) -> None:
        """Partial cluster flushed at window boundary, not merged."""
        now = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
        candle = self._make_candle(window="1d")

        # Batch 1: buy at 23:00 day 1 (incomplete day)
        df1 = pd.DataFrame([
            make_trade(now + pd.Timedelta("23h"), "100", "1000", "100000", 1),
        ])
        data1, cache = candle.aggregate(now, now + pd.Timedelta("23h30min"), df1, {})
        self.assertEqual(len(data1), 0)  # Day not complete
        self.assertIn("partial_cluster", cache)

        # Batch 2: buy at 01:00 day 2 (different window, same direction)
        df2 = pd.DataFrame([
            make_trade(now + pd.Timedelta("25h"), "101", "2000", "202000", 1),
        ])
        data2, cache2 = candle.aggregate(
            now + pd.Timedelta("23h30min"), now + pd.Timedelta("26h"), df2, cache
        )

        # Day 1 candle emitted with volume 1000 (partial flushed, not merged)
        self.assertEqual(len(data2), 1)
        self.assertEqual(float(data2[0]["volume"]), 1000.0)
        # Day 2 cluster is now partial
        self.assertIn("partial_cluster", cache2)
        self.assertEqual(float(cache2["partial_cluster"]["volume"]), 2000.0)

    def test_cross_batch_stats_accumulate(self) -> None:
        """Stats accumulate correctly across batches in same window."""
        now = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
        candle = self._make_candle(window="1h")

        # Batch 1: 2 clusters at 00:05, 00:10 (different directions = separate clusters)
        df1 = pd.DataFrame([
            make_trade(now + pd.Timedelta("5min"), "100", "1000", "100000", 1),
            make_trade(now + pd.Timedelta("10min"), "100", "2000", "200000", -1),
        ])
        data1, cache1 = candle.aggregate(now, now + pd.Timedelta("15min"), df1, {})

        # No candle emitted yet (window incomplete)
        self.assertEqual(len(data1), 0)
        self.assertIn("next", cache1)

        # Batch 2: 2 more clusters at 00:45, 00:55 (different directions = separate)
        df2 = pd.DataFrame([
            make_trade(now + pd.Timedelta("45min"), "100", "3000", "300000", 1),
            make_trade(now + pd.Timedelta("55min"), "100", "4000", "400000", -1),
        ])
        data2, cache2 = candle.aggregate(
            now + pd.Timedelta("15min"), now + pd.Timedelta("1h5min"), df2, cache1
        )

        # Now we should have 1 candle for 00:00-01:00
        self.assertEqual(len(data2), 1)
        result = data2[0]

        # OHLCV should span all trades (1000+2000+3000+4000)
        self.assertEqual(float(result["volume"]), 10000.0)

        # Bucket stats should include all 4 clusters from both batches
        total_clusters = sum(
            result[f"p{p}"]["totalClusterCount"]
            for p in [0, 25, 50, 75, 90, 95, 99]
        )
        self.assertEqual(total_clusters, 4)
