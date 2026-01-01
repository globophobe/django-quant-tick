import pandas as pd

from quant_tick.models import Candle, CandleData, MACrossoverStrategy, Signal

from ..test_strategies import BaseStrategyTest


class MACrossoverStrategyTest(BaseStrategyTest):
    """MA Crossover Strategy tests."""

    def setUp(self):
        """Set up."""
        super().setUp()
        self.candle = Candle.objects.create()
        self.candle.symbols.add(self.symbol)
        self.strategy = MACrossoverStrategy.objects.create(
            candle=self.candle,
            symbol=self.symbol,
            json_data={
                "moving_average_type": "sma",
                "fast_window": 2,
                "slow_window": 3,
            },
        )

    def test_get_events(self):
        """Test get_events with incomplete event."""
        for i in range(4):
            CandleData.objects.create(
                candle=self.candle,
                timestamp=self.timestamp_from + pd.Timedelta(f"{i}min"),
                json_data={"close": 100 + i},
            )

        events = self.strategy.get_events(
            timestamp_from=self.timestamp_from,
            timestamp_to=self.timestamp_from + pd.Timedelta("10min"),
            include_incomplete=True,
        )
        self.assertEqual(len(events), 1)
        event = events.iloc[0]
        self.assertEqual(event["direction"], 1)
        self.assertIsNone(event["exit_price"])

    def test_get_events_with_direction_changes(self):
        """Test get_events with direction changes."""
        prices = [100, 101, 102, 101, 100, 99, 98]
        for i, price in enumerate(prices):
            CandleData.objects.create(
                candle=self.candle,
                timestamp=self.timestamp_from + pd.Timedelta(f"{i}min"),
                json_data={"close": price},
            )

        events = self.strategy.get_events(
            timestamp_from=self.timestamp_from,
            timestamp_to=self.timestamp_from + pd.Timedelta("20min"),
            include_incomplete=False,
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(events.iloc[0]["direction"], 1)

    def test_inference(self):
        """Test inference."""
        for i in range(3):
            candle_data = CandleData.objects.create(
                candle=self.candle,
                timestamp=self.timestamp_from + pd.Timedelta(f"{i}min"),
                json_data={"close": 100 + i},
            )
        self.strategy.inference(candle_data)
        signals = Signal.objects.filter(strategy=self.strategy)
        self.assertEqual(len(signals), 1)
        signal = signals[0]
        self.assertEqual(signal.json_data["event"]["direction"], 1)

    def test_backtest_then_inference(self):
        """Test backtest, then inference."""
        for i in range(4):
            candle_data = CandleData.objects.create(
                candle=self.candle,
                timestamp=self.timestamp_from + pd.Timedelta(f"{i}min"),
                json_data={"close": 100 + i},
            )

        self.strategy.backtest()
        live_candle_data = CandleData.objects.create(
            candle=self.candle,
            timestamp=candle_data.timestamp + pd.Timedelta("1min"),
            json_data={"close": 100},
        )
        self.strategy.inference(live_candle_data)
        signals = Signal.objects.filter(strategy=self.strategy)
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].json_data["event"]["direction"], -1)
