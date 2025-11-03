import logging
from datetime import datetime, timedelta, timezone
from io import BytesIO

import joblib
import numpy as np
import pandas as pd
from django.core.files.base import ContentFile
from django.test import TestCase

from quant_tick.constants import Exchange
from quant_tick.lib.ml import (
    align_features_to_schema,
    apply_triple_barrier,
    compute_ewma_volatility,
    compute_features,
    compute_sample_weights,
    features_from_candles,
    generate_signals,
    train_model,
    trigger_ml_inference,
)
from quant_tick.models import (
    Candle,
    CandleData,
    GlobalSymbol,
    MLArtifact,
    MLConfig,
    MLRun,
    MLSignal,
    Symbol,
)


class ComputeFeaturesTest(TestCase):
    def test_compute_features_empty_df(self):
        """Compute features returns empty for empty dataframe."""
        df = pd.DataFrame()
        result = compute_features(df)
        self.assertTrue(result.empty)

    def test_compute_features_basic(self):
        """Compute features generates price and technical indicators."""
        now = datetime.now(timezone.utc)
        data = {
            "timestamp": [now + timedelta(minutes=i) for i in range(10)],
            "close": [100 + i for i in range(10)],
            "high": [101 + i for i in range(10)],
            "low": [99 + i for i in range(10)],
        }
        df = pd.DataFrame(data)
        result = compute_features(df)

        self.assertFalse(result.empty)
        self.assertIn("return", result.columns)
        self.assertIn("log_return", result.columns)
        self.assertIn("return_ma_5", result.columns)
        self.assertIn("volatility_5", result.columns)
        self.assertIn("atr_14", result.columns)
        self.assertIn("bar_duration", result.columns)

    def test_compute_features_with_volume_data(self):
        """Compute features generates order flow metrics when volume data present."""
        now = datetime.now(timezone.utc)
        data = {
            "timestamp": [now + timedelta(minutes=i) for i in range(10)],
            "close": [100 + i for i in range(10)],
            "buyVolume": [50 + i for i in range(10)],
            "sellVolume": [40 + i for i in range(10)],
            "buyNotional": [5000 + i * 100 for i in range(10)],
            "sellNotional": [4000 + i * 100 for i in range(10)],
        }
        df = pd.DataFrame(data)
        result = compute_features(df)

        self.assertIn("ofi", result.columns)
        self.assertIn("ofi_ewma", result.columns)
        self.assertIn("dollar_imbalance", result.columns)
        self.assertIn("buy_proportion", result.columns)


class EWMAVolatilityTest(TestCase):
    def test_ewma_volatility(self):
        """EWMA volatility computes exponentially weighted moving average."""
        now = datetime.now(timezone.utc)
        data = {
            "timestamp": [now + timedelta(minutes=i) for i in range(50)],
            "close": [100 + np.sin(i / 5) * 10 for i in range(50)],
        }
        df = pd.DataFrame(data)
        vol = compute_ewma_volatility(df, span=20)

        self.assertEqual(len(vol), 50)
        self.assertTrue(all(vol >= 0))
        self.assertFalse(vol.isna().any())


class TripleBarrierTest(TestCase):
    def test_triple_barrier_labels(self):
        """Triple barrier labeling creates labels and event metadata."""
        now = datetime.now(timezone.utc)
        data = {
            "timestamp": [now + timedelta(minutes=i) for i in range(100)],
            "close": [100 + i * 0.1 for i in range(100)],
        }
        df = pd.DataFrame(data)
        result = apply_triple_barrier(df, pt_mult=2.0, sl_mult=1.0, max_holding=10)

        self.assertIn("label", result.columns)
        self.assertIn("event_end_idx", result.columns)
        self.assertIn("event_end_time", result.columns)
        self.assertTrue(all(result["label"].isin([-1, 0, 1])))

    def test_triple_barrier_profit_take(self):
        """Triple barrier labels profit take when price reaches target."""
        now = datetime.now(timezone.utc)
        close_prices = [100.0]
        for i in range(1, 50):
            if i < 5:
                close_prices.append(100.0 + i * 0.5)
            else:
                close_prices.append(100.0)

        data = {
            "timestamp": [now + timedelta(minutes=i) for i in range(50)],
            "close": close_prices,
        }
        df = pd.DataFrame(data)
        result = apply_triple_barrier(df, pt_mult=2.0, sl_mult=1.0, max_holding=48)

        first_label = result.loc[0, "label"]
        self.assertEqual(first_label, 1)


class SampleWeightsTest(TestCase):
    def test_sample_weights_no_event_end(self):
        """Sample weights default to 1.0 without event overlap data."""
        now = datetime.now(timezone.utc)
        data = {
            "timestamp": [now + timedelta(minutes=i) for i in range(10)],
            "close": [100 + i for i in range(10)],
        }
        df = pd.DataFrame(data)
        result = compute_sample_weights(df)

        self.assertIn("sample_weight", result.columns)
        self.assertTrue(all(result["sample_weight"] == 1.0))

    def test_sample_weights_with_overlap(self):
        """Sample weights decrease based on event overlap."""
        now = datetime.now(timezone.utc)
        data = {
            "timestamp": [now + timedelta(minutes=i) for i in range(10)],
            "close": [100 + i for i in range(10)],
            "event_end_idx": [5, 6, 7, 8, 9, 9, 9, 9, 9, 9],
        }
        df = pd.DataFrame(data)
        result = compute_sample_weights(df)

        self.assertIn("sample_weight", result.columns)
        self.assertTrue(all(result["sample_weight"] > 0))
        self.assertTrue(all(result["sample_weight"] <= 1.0))


class TrainModelTest(TestCase):
    def test_train_model_basic(self):
        """Train model with purged CV returns model and metrics."""
        now = datetime.now(timezone.utc)
        data = {
            "timestamp": [now + timedelta(minutes=i) for i in range(500)],
            "close": [100 + np.sin(i / 10) * 5 for i in range(500)],
            "high": [101 + np.sin(i / 10) * 5 for i in range(500)],
            "low": [99 + np.sin(i / 10) * 5 for i in range(500)],
        }
        df = pd.DataFrame(data)
        df = compute_features(df)
        df = apply_triple_barrier(df, pt_mult=2.0, sl_mult=1.0, max_holding=10)
        df = compute_sample_weights(df)

        model, metrics, importances, metadata, _ = train_model(
            df, n_estimators=10, n_splits=3, embargo_bars=5
        )

        self.assertIsNotNone(model)
        self.assertIn("auc", metrics)
        self.assertIn("f1", metrics)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIsInstance(importances, dict)
        self.assertIn("sklearn_version", metadata)
        self.assertIn("n_features", metadata)


class GenerateSignalsTest(TestCase):
    def test_generate_signals(self):
        """Generate signals from model predictions."""
        now = datetime.now(timezone.utc)
        data = {
            "timestamp": [now + timedelta(minutes=i) for i in range(500)],
            "close": [100 + np.sin(i / 10) * 5 for i in range(500)],
            "high": [101 + np.sin(i / 10) * 5 for i in range(500)],
            "low": [99 + np.sin(i / 10) * 5 for i in range(500)],
        }
        df = pd.DataFrame(data)
        df = compute_features(df)
        df = apply_triple_barrier(df, pt_mult=2.0, sl_mult=1.0, max_holding=10)
        df = compute_sample_weights(df)

        model, _, _, _, _ = train_model(df, n_estimators=10, n_splits=3, embargo_bars=5)

        test_df = df.tail(50).copy()
        signals = generate_signals(model, test_df, prob_threshold=0.5)

        self.assertIsInstance(signals, list)
        for sig in signals:
            self.assertIn("timestamp", sig)
            self.assertIn("prediction", sig)
            self.assertIn("probability", sig)
            self.assertGreaterEqual(sig["probability"], 0.5)


class FeaturesFromCandlesTest(TestCase):
    def setUp(self):
        self.global_symbol = GlobalSymbol.objects.create(name="test-global")
        self.symbol = Symbol.objects.create(
            global_symbol=self.global_symbol,
            exchange=Exchange.COINBASE,
            api_symbol="BTC-USD",
        )
        self.candle = Candle.objects.create()
        self.candle.symbols.add(self.symbol)

    def test_features_from_candles_empty(self):
        """Features from candles returns empty when no candle data exists."""
        ts_from = datetime.now(timezone.utc)
        ts_to = ts_from + timedelta(hours=1)
        result = features_from_candles(self.candle, ts_from, ts_to, lookback_bars=10)
        self.assertTrue(result.empty)

    def test_features_from_candles_with_data(self):
        """Features from candles computes features for requested time range."""
        now = datetime.now(timezone.utc)
        for i in range(100):
            ts = now + timedelta(minutes=i)
            json_data = {
                "close": 100 + i * 0.1,
                "high": 101 + i * 0.1,
                "low": 99 + i * 0.1,
                "volume": 1000,
            }
            CandleData.objects.create(candle=self.candle, timestamp=ts, json_data=json_data)

        ts_from = now + timedelta(minutes=50)
        ts_to = now + timedelta(minutes=100)
        result = features_from_candles(self.candle, ts_from, ts_to, lookback_bars=10)

        self.assertFalse(result.empty)
        self.assertTrue(all(result["timestamp"] >= ts_from))
        self.assertIn("return", result.columns)
        self.assertIn("volatility_5", result.columns)

    def test_features_from_candles_lookback(self):
        """Features from candles includes lookback bars for indicator computation."""
        now = datetime.now(timezone.utc)
        for i in range(100):
            ts = now + timedelta(minutes=i)
            json_data = {"close": 100 + i * 0.1, "volume": 1000}
            CandleData.objects.create(candle=self.candle, timestamp=ts, json_data=json_data)

        ts_from = now + timedelta(minutes=60)
        ts_to = now + timedelta(minutes=100)
        result = features_from_candles(self.candle, ts_from, ts_to, lookback_bars=20)

        self.assertFalse(result.empty)
        first_ts = result.iloc[0]["timestamp"]
        self.assertGreaterEqual(first_ts, ts_from)


class FeaturesFromCandlesEnrichmentTest(TestCase):
    def setUp(self):
        self.global_symbol = GlobalSymbol.objects.create(name="test-global")
        self.symbol = Symbol.objects.create(
            global_symbol=self.global_symbol,
            exchange=Exchange.COINBASE,
            api_symbol="BTC-USD",
        )
        self.candle = Candle.objects.create()
        self.candle.symbols.add(self.symbol)

    def test_features_from_candles_derives_sell_volume(self):
        """Features from candles derives sellVolume correctly from volume and buyVolume."""
        now = datetime.now(timezone.utc)
        for i in range(50):
            ts = now + timedelta(minutes=i)
            json_data = {
                "close": 100 + i * 0.1,
                "volume": 1000 + i * 10,
                "buyVolume": 600 + i * 6,
                "notional": 100000 + i * 1000,
                "buyNotional": 60000 + i * 600,
            }
            CandleData.objects.create(candle=self.candle, timestamp=ts, json_data=json_data)

        ts_from = now + timedelta(minutes=20)
        ts_to = now + timedelta(minutes=50)
        result = features_from_candles(self.candle, ts_from, ts_to, lookback_bars=10)

        self.assertFalse(result.empty)
        self.assertIn("buyVolume", result.columns)
        self.assertIn("sellVolume", result.columns)
        self.assertIn("buyNotional", result.columns)
        self.assertIn("sellNotional", result.columns)

        for idx, row in result.iterrows():
            expected_sell_vol = row["volume"] - row["buyVolume"]
            expected_sell_notional = row["notional"] - row["buyNotional"]
            self.assertAlmostEqual(row["sellVolume"], expected_sell_vol, places=2)
            self.assertAlmostEqual(row["sellNotional"], expected_sell_notional, places=2)

    def test_features_from_candles_derives_sell_notional(self):
        """Features from candles derives sellNotional correctly from notional and buyNotional."""
        now = datetime.now(timezone.utc)
        test_data = [
            {"volume": 1000, "buyVolume": 700, "notional": 50000, "buyNotional": 35000},
            {"volume": 1500, "buyVolume": 600, "notional": 75000, "buyNotional": 30000},
            {"volume": 2000, "buyVolume": 1200, "notional": 100000, "buyNotional": 60000},
        ]

        for i, data in enumerate(test_data):
            ts = now + timedelta(minutes=i)
            json_data = {"close": 50.0, **data}
            CandleData.objects.create(candle=self.candle, timestamp=ts, json_data=json_data)

        ts_from = now
        ts_to = now + timedelta(minutes=10)
        result = features_from_candles(self.candle, ts_from, ts_to, lookback_bars=0)

        for i, data in enumerate(test_data):
            row = result.iloc[i]
            expected_sell_vol = data["volume"] - data["buyVolume"]
            expected_sell_notional = data["notional"] - data["buyNotional"]
            self.assertEqual(row["sellVolume"], expected_sell_vol)
            self.assertEqual(row["sellNotional"], expected_sell_notional)

    def test_features_from_candles_flow_features_present(self):
        """Features from candles includes order flow features when sell values are derived."""
        now = datetime.now(timezone.utc)
        for i in range(100):
            ts = now + timedelta(minutes=i)
            json_data = {
                "close": 100 + i * 0.1,
                "high": 101 + i * 0.1,
                "low": 99 + i * 0.1,
                "volume": 1000 + i * 10,
                "buyVolume": 550 + i * 5,
                "notional": 100000 + i * 1000,
                "buyNotional": 55000 + i * 500,
            }
            CandleData.objects.create(candle=self.candle, timestamp=ts, json_data=json_data)

        ts_from = now + timedelta(minutes=50)
        ts_to = now + timedelta(minutes=100)
        result = features_from_candles(self.candle, ts_from, ts_to, lookback_bars=20)

        self.assertFalse(result.empty)
        self.assertIn("ofi", result.columns)
        self.assertIn("ofi_ewma", result.columns)
        self.assertIn("dollar_imbalance", result.columns)
        self.assertIn("buy_proportion", result.columns)

        self.assertFalse(result["ofi"].isna().all())
        self.assertFalse(result["dollar_imbalance"].isna().all())


class AlignFeaturesToSchemaTest(TestCase):
    def test_align_features_with_extra_columns(self):
        """Align features drops extra columns not in training schema."""
        data = {
            "timestamp": [datetime.now(timezone.utc)],
            "close": [100.0],
            "extra_col_1": [1.0],
            "extra_col_2": [2.0],
            "return": [0.01],
            "volatility_5": [0.02],
        }
        df = pd.DataFrame(data)
        expected_columns = ["close", "return", "volatility_5"]

        logger = logging.getLogger("quant_tick.lib.ml")
        logger.disabled = True
        try:
            aligned_df, drift_stats = align_features_to_schema(df, expected_columns)
        finally:
            logger.disabled = False

        self.assertIn("close", aligned_df.columns)
        self.assertIn("return", aligned_df.columns)
        self.assertIn("volatility_5", aligned_df.columns)
        self.assertNotIn("extra_col_1", aligned_df.columns)
        self.assertNotIn("extra_col_2", aligned_df.columns)
        self.assertEqual(drift_stats["n_extra"], 2)
        self.assertEqual(drift_stats["n_missing"], 0)

    def test_align_features_with_missing_columns(self):
        """Align features adds missing columns filled with zeros."""
        data = {
            "timestamp": [datetime.now(timezone.utc)],
            "close": [100.0],
            "return": [0.01],
        }
        df = pd.DataFrame(data)
        expected_columns = ["close", "return", "volatility_5", "atr_14"]

        logger = logging.getLogger("quant_tick.lib.ml")
        logger.disabled = True
        try:
            aligned_df, drift_stats = align_features_to_schema(df, expected_columns)
        finally:
            logger.disabled = False

        self.assertIn("close", aligned_df.columns)
        self.assertIn("return", aligned_df.columns)
        self.assertIn("volatility_5", aligned_df.columns)
        self.assertIn("atr_14", aligned_df.columns)
        self.assertEqual(aligned_df["volatility_5"].iloc[0], 0.0)
        self.assertEqual(aligned_df["atr_14"].iloc[0], 0.0)
        self.assertEqual(drift_stats["n_extra"], 0)
        self.assertEqual(drift_stats["n_missing"], 2)

    def test_align_features_with_both_extra_and_missing(self):
        """Align features handles both extra and missing columns."""
        data = {
            "timestamp": [datetime.now(timezone.utc)],
            "close": [100.0],
            "return": [0.01],
            "extra_feature": [999.0],
        }
        df = pd.DataFrame(data)
        expected_columns = ["close", "return", "volatility_5"]

        logger = logging.getLogger("quant_tick.lib.ml")
        logger.disabled = True
        try:
            aligned_df, drift_stats = align_features_to_schema(df, expected_columns)
        finally:
            logger.disabled = False

        self.assertIn("close", aligned_df.columns)
        self.assertIn("return", aligned_df.columns)
        self.assertIn("volatility_5", aligned_df.columns)
        self.assertNotIn("extra_feature", aligned_df.columns)
        self.assertEqual(aligned_df["volatility_5"].iloc[0], 0.0)
        self.assertEqual(drift_stats["n_extra"], 1)
        self.assertEqual(drift_stats["n_missing"], 1)




class TriggerMLInferenceTest(TestCase):
    def setUp(self):
        self.global_symbol = GlobalSymbol.objects.create(name="test-global")
        self.symbol = Symbol.objects.create(
            global_symbol=self.global_symbol,
            exchange=Exchange.COINBASE,
            api_symbol="BTC-USD",
        )
        self.candle = Candle.objects.create()
        self.candle.symbols.add(self.symbol)

    def test_trigger_ml_inference_no_config(self):
        """Trigger ML inference returns empty when no active config."""
        ts_from = datetime.now(timezone.utc)
        ts_to = ts_from + timedelta(hours=1)
        signals = trigger_ml_inference(self.candle, ts_from, ts_to)
        self.assertEqual(signals, [])

    def test_trigger_ml_inference_no_run(self):
        """Trigger ML inference returns empty when no completed run exists."""
        MLConfig.objects.create(
            candle=self.candle,
            code_name="test-config",
            json_data={"prob_threshold": 0.6, "lookback_bars": 50},
            is_active=True,
        )
        ts_from = datetime.now(timezone.utc)
        ts_to = ts_from + timedelta(hours=1)
        signals = trigger_ml_inference(self.candle, ts_from, ts_to)
        self.assertEqual(signals, [])

    def test_trigger_ml_inference_with_model(self):
        """Trigger ML inference with trained model creates signals."""
        now = datetime.now(timezone.utc)
        for i in range(250):
            ts = now + timedelta(minutes=i)
            json_data = {
                "close": 100 + np.sin(i / 10) * 5,
                "high": 101 + np.sin(i / 10) * 5,
                "low": 99 + np.sin(i / 10) * 5,
                "volume": 1000,
                "buyVolume": 500,
                "notional": 50000,
                "buyNotional": 25000,
            }
            CandleData.objects.create(candle=self.candle, timestamp=ts, json_data=json_data)

        data = {
            "timestamp": [now + timedelta(minutes=i) for i in range(200)],
            "close": [100 + np.sin(i / 10) * 5 for i in range(200)],
            "high": [101 + np.sin(i / 10) * 5 for i in range(200)],
            "low": [99 + np.sin(i / 10) * 5 for i in range(200)],
            "volume": [1000 for _ in range(200)],
        }
        df = pd.DataFrame(data)
        df = compute_features(df)
        df = apply_triple_barrier(df, pt_mult=2.0, sl_mult=1.0, max_holding=10)
        df = compute_sample_weights(df)

        model, _, _, metadata, _ = train_model(df, n_estimators=10, n_splits=3, embargo_bars=5)

        cfg = MLConfig.objects.create(
            candle=self.candle,
            code_name="test-config",
            json_data={"prob_threshold": 0.5, "lookback_bars": 50},
            is_active=True,
        )

        run = MLRun.objects.create(
            ml_config=cfg,
            timestamp_from=now,
            timestamp_to=now + timedelta(hours=5),
            status="completed",
            metadata=metadata,
        )

        buf = BytesIO()
        joblib.dump(model, buf)
        buf.seek(0)
        content = ContentFile(buf.read(), "model.joblib")
        MLArtifact.objects.create(ml_run=run, artifact=content, version="1.0")

        ts_from = now + timedelta(minutes=200)
        ts_to = now + timedelta(minutes=250)

        logger = logging.getLogger("quant_tick.lib.ml")
        logger.disabled = True
        try:
            signals = trigger_ml_inference(self.candle, ts_from, ts_to)
            self.assertIsInstance(signals, list)
            signal_count = MLSignal.objects.filter(candle=self.candle).count()
            self.assertEqual(len(signals), signal_count)
        finally:
            logger.disabled = False

    def test_trigger_ml_inference_idempotent(self):
        """Trigger ML inference is idempotent on repeated calls."""
        now = datetime.now(timezone.utc)
        for i in range(250):
            ts = now + timedelta(minutes=i)
            json_data = {
                "close": 100 + np.sin(i / 10) * 5,
                "high": 101 + np.sin(i / 10) * 5,
                "low": 99 + np.sin(i / 10) * 5,
                "volume": 1000,
                "buyVolume": 500,
                "notional": 50000,
                "buyNotional": 25000,
            }
            CandleData.objects.create(candle=self.candle, timestamp=ts, json_data=json_data)

        data = {
            "timestamp": [now + timedelta(minutes=i) for i in range(200)],
            "close": [100 + np.sin(i / 10) * 5 for i in range(200)],
            "high": [101 + np.sin(i / 10) * 5 for i in range(200)],
            "low": [99 + np.sin(i / 10) * 5 for i in range(200)],
            "volume": [1000 for _ in range(200)],
        }
        df = pd.DataFrame(data)
        df = compute_features(df)
        df = apply_triple_barrier(df, pt_mult=2.0, sl_mult=1.0, max_holding=10)
        df = compute_sample_weights(df)

        model, _, _, metadata, _ = train_model(df, n_estimators=10, n_splits=3, embargo_bars=5)

        cfg = MLConfig.objects.create(
            candle=self.candle,
            code_name="test-config",
            json_data={"prob_threshold": 0.5, "lookback_bars": 50},
            is_active=True,
        )

        run = MLRun.objects.create(
            ml_config=cfg,
            timestamp_from=now,
            timestamp_to=now + timedelta(hours=2),
            status="completed",
            metadata=metadata,
        )

        buf = BytesIO()
        joblib.dump(model, buf)
        buf.seek(0)
        content = ContentFile(buf.read(), "model.joblib")
        MLArtifact.objects.create(ml_run=run, artifact=content, version="1.0")

        ts_from = now + timedelta(minutes=200)
        ts_to = now + timedelta(minutes=250)

        logger = logging.getLogger("quant_tick.lib.ml")
        logger.disabled = True
        try:
            trigger_ml_inference(self.candle, ts_from, ts_to)
            count1 = MLSignal.objects.filter(candle=self.candle).count()

            trigger_ml_inference(self.candle, ts_from, ts_to)
            count2 = MLSignal.objects.filter(candle=self.candle).count()

            self.assertEqual(count1, count2)
            self.assertGreater(count1, 0)
        finally:
            logger.disabled = False

    def test_trigger_ml_inference_tolerates_missing_columns(self):
        """Trigger ML inference tolerates missing feature columns."""
        now = datetime.now(timezone.utc)

        for i in range(200):
            ts = now + timedelta(minutes=i)
            json_data = {
                "close": 100 + np.sin(i / 10) * 5,
                "high": 101 + np.sin(i / 10) * 5,
                "low": 99 + np.sin(i / 10) * 5,
                "volume": 1000,
                "buyVolume": 500,
                "notional": 50000,
                "buyNotional": 25000,
            }
            CandleData.objects.create(candle=self.candle, timestamp=ts, json_data=json_data)

        data = {
            "timestamp": [now + timedelta(minutes=i) for i in range(150)],
            "close": [100 + np.sin(i / 10) * 5 for i in range(150)],
            "high": [101 + np.sin(i / 10) * 5 for i in range(150)],
            "low": [99 + np.sin(i / 10) * 5 for i in range(150)],
            "volume": [1000 for _ in range(150)],
        }
        df = pd.DataFrame(data)
        df = compute_features(df)
        df = apply_triple_barrier(df, pt_mult=2.0, sl_mult=1.0, max_holding=10)
        df = compute_sample_weights(df)

        model, _, _, metadata, _ = train_model(df, n_estimators=10, n_splits=3, embargo_bars=5)

        cfg = MLConfig.objects.create(
            candle=self.candle,
            code_name="test-config",
            json_data={"prob_threshold": 0.5, "lookback_bars": 50},
            is_active=True,
        )

        run = MLRun.objects.create(
            ml_config=cfg,
            timestamp_from=now,
            timestamp_to=now + timedelta(hours=3),
            status="completed",
            metadata=metadata,
        )

        buf = BytesIO()
        joblib.dump(model, buf)
        buf.seek(0)
        content = ContentFile(buf.read(), "model.joblib")
        MLArtifact.objects.create(ml_run=run, artifact=content, version="1.0")

        for cd in CandleData.objects.filter(candle=self.candle):
            cd.json_data["buyVolume"] = 500
            cd.json_data["notional"] = 50000
            cd.json_data["buyNotional"] = 25000
            cd.save()

        ts_from = now + timedelta(minutes=150)
        ts_to = now + timedelta(minutes=200)

        logger = logging.getLogger("quant_tick.lib.ml")
        logger.disabled = True
        try:
            signals = trigger_ml_inference(self.candle, ts_from, ts_to)
            self.assertIsInstance(signals, list)
        except Exception as e:
            self.fail(f"trigger_ml_inference raised exception with missing columns: {e}")
        finally:
            logger.disabled = False

    def test_trigger_ml_inference_tolerates_extra_columns(self):
        """Trigger ML inference tolerates extra feature columns in live data."""
        now = datetime.now(timezone.utc)

        for i in range(200):
            ts = now + timedelta(minutes=i)
            json_data = {
                "close": 100 + np.sin(i / 10) * 5,
                "high": 101 + np.sin(i / 10) * 5,
                "low": 99 + np.sin(i / 10) * 5,
                "volume": 1000,
            }
            CandleData.objects.create(candle=self.candle, timestamp=ts, json_data=json_data)

        data = {
            "timestamp": [now + timedelta(minutes=i) for i in range(150)],
            "close": [100 + np.sin(i / 10) * 5 for i in range(150)],
            "high": [101 + np.sin(i / 10) * 5 for i in range(150)],
            "low": [99 + np.sin(i / 10) * 5 for i in range(150)],
            "volume": [1000 for _ in range(150)],
        }
        df = pd.DataFrame(data)
        df = compute_features(df)
        df = apply_triple_barrier(df, pt_mult=2.0, sl_mult=1.0, max_holding=10)
        df = compute_sample_weights(df)

        model, _, _, metadata, _ = train_model(df, n_estimators=10, n_splits=3, embargo_bars=5)

        cfg = MLConfig.objects.create(
            candle=self.candle,
            code_name="test-config",
            json_data={"prob_threshold": 0.5, "lookback_bars": 50},
            is_active=True,
        )

        run = MLRun.objects.create(
            ml_config=cfg,
            timestamp_from=now,
            timestamp_to=now + timedelta(hours=3),
            status="completed",
            metadata=metadata,
        )

        buf = BytesIO()
        joblib.dump(model, buf)
        buf.seek(0)
        content = ContentFile(buf.read(), "model.joblib")
        MLArtifact.objects.create(ml_run=run, artifact=content, version="1.0")

        for cd in CandleData.objects.filter(candle=self.candle):
            cd.json_data["extra_feature_1"] = 999
            cd.json_data["extra_feature_2"] = 777
            cd.json_data["buyVolume"] = 500
            cd.json_data["notional"] = 50000
            cd.json_data["buyNotional"] = 25000
            cd.save()

        ts_from = now + timedelta(minutes=150)
        ts_to = now + timedelta(minutes=200)

        logger = logging.getLogger("quant_tick.lib.ml")
        logger.disabled = True
        try:
            signals = trigger_ml_inference(self.candle, ts_from, ts_to)
            self.assertIsInstance(signals, list)
        except Exception as e:
            self.fail(f"trigger_ml_inference raised exception with extra columns: {e}")
        finally:
            logger.disabled = False
