from decimal import Decimal

import pandas as pd

from quant_tick.models import (
    CandleData,
    DailyClusterTrendStrategy,
    SignificantCluster,
)

from ..test_strategies import BaseStrategyTest


class DailyClusterTrendStrategyTest(BaseStrategyTest):
    """Daily cluster trend strategy tests."""

    def setUp(self):
        """Set up."""
        super().setUp()
        self.candle = SignificantCluster.objects.create(
            symbol=self.symbol,
            json_data={
                "source_data": "raw",
                "significant_cluster_filter": "50000",
            },
        )
        self.strategy = DailyClusterTrendStrategy.objects.create(
            candle=self.candle,
            json_data={"ma_window": 3},  # Short window for testing
        )

    def _create_candle_data(self, sequence):
        """Create CandleData from (date_offset, price) tuples.

        Each tuple creates one cluster candle on that day.
        Multiple tuples with same date_offset create multiple candles per day.
        """
        for i, item in enumerate(sequence):
            if len(item) == 2:
                day_offset, price = item
                volume = 1000
            else:
                day_offset, price, volume = item
            CandleData.objects.create(
                candle=self.candle,
                timestamp=self.timestamp_from + pd.Timedelta(days=day_offset, hours=i % 24),
                json_data={
                    "open": str(price),
                    "high": str(price + 10),
                    "low": str(price - 10),
                    "close": str(price),
                    "volume": str(volume),
                },
            )

    def test_aggregates_to_daily(self):
        """Multiple cluster candles same day aggregate to one daily bar."""
        # Day 0: 3 candles, Day 1: 2 candles, Day 2: 1 candle
        self._create_candle_data([
            (0, 100), (0, 102), (0, 101),  # Day 0: 3 candles
            (1, 105), (1, 107),             # Day 1: 2 candles
            (2, 110),                       # Day 2: 1 candle
        ])
        df = pd.DataFrame(self.candle.get_candle_data(
            self.timestamp_from,
            self.timestamp_from + pd.Timedelta(days=10),
        ))
        df = self.strategy.get_data_frame(df)
        daily = self.strategy._aggregate_to_daily(df)

        self.assertEqual(len(daily), 3)
        self.assertEqual(daily.loc[0, "candle_count"], 3)
        self.assertEqual(daily.loc[1, "candle_count"], 2)
        self.assertEqual(daily.loc[2, "candle_count"], 1)

    def test_long_when_above_ma(self):
        """Signal fires when close > MA."""
        # Create data where price rises above MA
        # Days 0-2: price 100 (MA builds up)
        # Day 3: price jumps to 120 (above MA of ~100)
        # Day 4: price stays 120 (entry day)
        # Day 5: price drops to 90 (below MA, exit signal)
        # Day 6: exit execution
        self._create_candle_data([
            (0, 100), (1, 100), (2, 100),  # MA warmup
            (3, 120),  # Signal: above MA
            (4, 120),  # Entry candle
            (5, 90),   # Exit signal: below MA
            (6, 90),   # Exit candle
        ])
        events = self.strategy.get_events(
            timestamp_from=self.timestamp_from,
            timestamp_to=self.timestamp_from + pd.Timedelta(days=10),
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(events.iloc[0]["direction"], 1)

    def test_flat_when_below_ma(self):
        """No position when close < MA."""
        # Price always below MA (declining)
        self._create_candle_data([
            (0, 100), (1, 95), (2, 90), (3, 85), (4, 80),
        ])
        events = self.strategy.get_events(
            timestamp_from=self.timestamp_from,
            timestamp_to=self.timestamp_from + pd.Timedelta(days=10),
        )
        self.assertEqual(len(events), 0)

    def test_entry_at_next_day_open(self):
        """Entry price uses next day's open."""
        self._create_candle_data([
            (0, 100), (1, 100), (2, 100),  # MA warmup
            (3, 120),  # Signal day
            (4, 150),  # Entry day - open should be 150
            (5, 80),   # Exit signal
            (6, 200),  # Exit day - open should be 200
        ])
        events = self.strategy.get_events(
            timestamp_from=self.timestamp_from,
            timestamp_to=self.timestamp_from + pd.Timedelta(days=10),
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(events.iloc[0]["entry_price"], Decimal("150"))
        self.assertEqual(events.iloc[0]["exit_price"], Decimal("200"))

    def test_never_short(self):
        """Direction is always 1 (long), never -1."""
        # Create multiple entries/exits
        self._create_candle_data([
            (0, 100), (1, 100), (2, 100),
            (3, 120), (4, 120),  # First entry
            (5, 80), (6, 80),    # First exit
            (7, 130), (8, 130),  # Second entry
            (9, 70), (10, 70),   # Second exit
        ])
        events = self.strategy.get_events(
            timestamp_from=self.timestamp_from,
            timestamp_to=self.timestamp_from + pd.Timedelta(days=15),
        )
        for _, event in events.iterrows():
            self.assertEqual(event["direction"], 1)

    def test_rate_confirmation(self):
        """With use_rate_confirmation=True, requires rate ratio."""
        strategy = DailyClusterTrendStrategy.objects.create(
            candle=self.candle,
            json_data={
                "ma_window": 3,
                "use_rate_confirmation": True,
                "rate_fast_window": 2,
                "rate_slow_window": 3,
                "rate_threshold": 1.2,
            },
        )
        # Days with varying candle counts
        # Day 0-2: 1 candle each (slow rate = 1)
        # Day 3: 3 candles, price rises (fast rate = 2, ratio = 2.0 > 1.2)
        # Day 4: entry
        self._create_candle_data([
            (0, 100),
            (1, 100),
            (2, 100),
            (3, 120), (3, 121), (3, 122),  # 3 candles, price up
            (4, 125), (4, 126), (4, 127),  # 3 candles
            (5, 80),   # Exit signal
            (6, 80),   # Exit candle
        ])
        events = strategy.get_events(
            timestamp_from=self.timestamp_from,
            timestamp_to=self.timestamp_from + pd.Timedelta(days=10),
        )
        # Should have entry since rate ratio > 1.2 and price > MA
        self.assertGreaterEqual(len(events), 1)

    def test_feature_columns(self):
        """Feature columns include daily-specific fields."""
        events = pd.DataFrame()
        cols = self.strategy.get_feature_columns(events)
        self.assertIn("daily_candle_count", cols)
        self.assertIn("ma_distance", cols)
        self.assertIn("rate_ratio", cols)

    def test_ma_window_default(self):
        """Default ma_window is 30."""
        strategy = DailyClusterTrendStrategy(candle=self.candle, json_data={})
        self.assertEqual(strategy.ma_window, 30)

    def test_ma_type_default(self):
        """Default ma_type is 'sma'."""
        strategy = DailyClusterTrendStrategy(candle=self.candle, json_data={})
        self.assertEqual(strategy.ma_type, "sma")

    def test_use_rate_confirmation_default(self):
        """Default use_rate_confirmation is False."""
        strategy = DailyClusterTrendStrategy(candle=self.candle, json_data={})
        self.assertFalse(strategy.use_rate_confirmation)

    def test_rate_threshold_default(self):
        """Default rate_threshold is 1.2."""
        strategy = DailyClusterTrendStrategy(candle=self.candle, json_data={})
        self.assertEqual(strategy.rate_threshold, 1.2)

    def test_include_incomplete(self):
        """Incomplete events included when include_incomplete=True."""
        self._create_candle_data([
            (0, 100), (1, 100), (2, 100),
            (3, 120),  # Signal
            (4, 120),  # Entry, but no exit
        ])
        events = self.strategy.get_events(
            timestamp_from=self.timestamp_from,
            timestamp_to=self.timestamp_from + pd.Timedelta(days=10),
            include_incomplete=True,
        )
        self.assertEqual(len(events), 1)
        self.assertIsNone(events.iloc[0]["exit_price"])

    def test_exclude_incomplete_by_default(self):
        """Incomplete events excluded by default."""
        self._create_candle_data([
            (0, 100), (1, 100), (2, 100),
            (3, 120),  # Signal
            (4, 120),  # Entry, no exit
        ])
        events = self.strategy.get_events(
            timestamp_from=self.timestamp_from,
            timestamp_to=self.timestamp_from + pd.Timedelta(days=10),
        )
        self.assertEqual(len(events), 0)

    def test_ema_ma_type(self):
        """EMA calculation works correctly."""
        strategy = DailyClusterTrendStrategy.objects.create(
            candle=self.candle,
            json_data={"ma_window": 3, "ma_type": "ema"},
        )
        self._create_candle_data([
            (0, 100), (1, 100), (2, 100),
            (3, 120), (4, 120),
            (5, 80), (6, 80),
        ])
        events = strategy.get_events(
            timestamp_from=self.timestamp_from,
            timestamp_to=self.timestamp_from + pd.Timedelta(days=10),
        )
        # Just verify it runs without error
        self.assertIsInstance(events, pd.DataFrame)
