import tempfile

import joblib
import pandas as pd

from quant_tick.management.commands.backtest import Command as BacktestCommand
from quant_tick.models import Candle, CandleData, MACrossoverStrategy, Signal

from ..test_strategies import BaseStrategyTest


class MACrossoverStrategyTest(BaseStrategyTest):
    """MA Crossover Strategy tests."""

    def setUp(self):
        """Set up."""
        super().setUp()
        self.candle = Candle.objects.create(symbol=self.symbol)
        self.strategy = MACrossoverStrategy.objects.create(
            candle=self.candle,
            json_data={
                "moving_average_type": "sma",
                "fast_window": 2,
                "slow_window": 3,
            },
        )

    def test_get_events(self):
        """Test get_events with incomplete event."""
        prices = [100, 101, 102, 101, 100, 101, 102, 101, 100]
        for i, price in enumerate(prices):
            CandleData.objects.create(
                candle=self.candle,
                timestamp=self.timestamp_from + pd.Timedelta(f"{i}min"),
                json_data={"close": price},
            )

        events = self.strategy.get_events(
            timestamp_from=self.timestamp_from,
            timestamp_to=self.timestamp_from + pd.Timedelta("10min"),
            include_incomplete=True,
        )
        self.assertEqual(len(events), 4)
        # First event has no previous run data
        self.assertTrue(pd.isna(events.iloc[0]["run_length_prev"]))
        self.assertEqual(events.iloc[0]["direction"], 1)
        self.assertEqual(events.iloc[1]["direction"], -1)
        self.assertEqual(events.iloc[2]["direction"], 1)
        self.assertEqual(events.iloc[3]["direction"], -1)
        self.assertIsNone(events.iloc[3]["exit_price"])

    def test_get_events_with_direction_changes(self):
        """Test get_events with direction changes."""
        prices = [100, 101, 102, 101, 100, 101, 102, 101, 100]
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
        self.assertEqual(len(events), 3)
        self.assertEqual(events.iloc[0]["direction"], 1)
        self.assertEqual(events.iloc[1]["direction"], -1)

    def test_inference(self):
        """Test inference."""
        prices = [100, 101, 102, 101, 100, 101, 102, 101, 100]
        for i, price in enumerate(prices):
            candle_data = CandleData.objects.create(
                candle=self.candle,
                timestamp=self.timestamp_from + pd.Timedelta(f"{i}min"),
                json_data={"close": price},
            )
        self.strategy.inference(candle_data)
        signals = Signal.objects.filter(strategy=self.strategy)
        self.assertEqual(len(signals), 1)
        signal = signals[0]
        self.assertEqual(signal.json_data["event"]["direction"], -1)

    def test_backtest_then_inference(self):
        """Test backtest, then inference."""
        prices = [100, 101, 102, 101, 100, 101, 102, 101, 100]
        for i, price in enumerate(prices):
            candle_data = CandleData.objects.create(
                candle=self.candle,
                timestamp=self.timestamp_from + pd.Timedelta(f"{i}min"),
                json_data={"close": price},
            )

        # Create parquet file from candle data
        df = pd.DataFrame(
            [
                {"timestamp": self.timestamp_from + pd.Timedelta(f"{i}min"), "close": p}
                for i, p in enumerate(prices)
            ]
        )

        # Train model and save to temp file
        events = self.strategy.get_events(
            timestamp_from=self.timestamp_from,
            timestamp_to=candle_data.timestamp,
            include_incomplete=False,
        )
        model, metadata, calibrator = self.strategy._train_model_with_cv(
            events,
            feature_cols=self.strategy.get_feature_columns(events),
            label_col=self.strategy.get_label_column(),
        )
        bundle = {"model": model, "metadata": metadata, "calibrator": calibrator}

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            joblib.dump(bundle, f.name)
            model_path = f.name

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
            df.to_parquet(f.name)
            parquet_path = f.name

        # Run backtest with model file
        cmd = BacktestCommand()
        cmd.run_backtest(
            strategy=self.strategy,
            input_file=parquet_path,
            artifact_name=model_path,
        )

        # Test inference
        live_candle_data = CandleData.objects.create(
            candle=self.candle,
            timestamp=candle_data.timestamp + pd.Timedelta("1min"),
            json_data={"close": 80},
        )
        self.strategy.inference(live_candle_data)
        signals = Signal.objects.filter(strategy=self.strategy)
        self.assertEqual(len(signals), 1)
        self.assertEqual(signals[0].json_data["event"]["direction"], -1)

    def test_hysteresis_filters_weak_signals(self):
        """Test that hysteresis filters weak crossover signals."""
        strategy_with_hyst = MACrossoverStrategy.objects.create(
            candle=self.candle,
            json_data={
                "moving_average_type": "sma",
                "fast_window": 2,
                "slow_window": 3,
                "hysteresis_k": 10.0,  # High k to filter all signals
            },
        )

        # Create price data with realizedVariance
        # Small price moves relative to vol threshold
        prices = [100, 101, 102, 101, 100, 101, 102, 101, 100]
        for i, price in enumerate(prices):
            CandleData.objects.create(
                candle=self.candle,
                timestamp=self.timestamp_from + pd.Timedelta(f"{i}min"),
                json_data={"close": price, "realizedVariance": "0.0001"},
            )

        events = strategy_with_hyst.get_events(
            timestamp_from=self.timestamp_from,
            timestamp_to=self.timestamp_from + pd.Timedelta("10min"),
            include_incomplete=True,
        )

        # High hysteresis_k filters all signals (ma_diff never exceeds threshold)
        self.assertEqual(len(events), 0)
