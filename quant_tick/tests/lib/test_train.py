"""Tests for competing-risks model training."""

from django.test import TestCase


class ModelBundleFormatTest(TestCase):
    """Test model bundle format compatibility."""

    def test_model_bundle_format_roundtrip(self):
        """Test model bundle format compatibility: save → load → validate."""
        import tempfile
        from pathlib import Path

        import joblib
        import lightgbm as lgb
        import numpy as np

        # 1. Create minimal mock models (don't need real training)
        models_dict = {
            "first_hit_h48": lgb.LGBMClassifier(n_estimators=10, objective="multiclass", num_class=3),
            "first_hit_h96": lgb.LGBMClassifier(n_estimators=10, objective="multiclass", num_class=3),
        }

        # Fit with dummy data so models are valid
        X_dummy = np.random.randn(50, 5)
        y_dummy = np.random.randint(0, 3, 50)
        for model in models_dict.values():
            model.fit(X_dummy, y_dummy)

        # 2. Create bundle (matching training output format)
        bundle = {
            "models": models_dict,
            "metadata": {
                "model_kind": "competing_risks",
                "feature_cols": ["close", "realizedVol", "volRatio", "width", "asymmetry"],
                "horizons": [48, 96],
                "widths": [0.04, 0.06],
                "asymmetries": [-0.2, 0.0, 0.2],
                "trained_at": "2025-01-01T00:00:00",
                "model_version": "5.0",
            }
        }

        # 3. Save to temp file (mock GCS)
        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
            joblib.dump(bundle, f)
            temp_path = f.name

        try:
            # 4. Load bundle
            loaded_bundle = joblib.load(temp_path)

            # 5. Validate structure (same as storage.py validation logic)
            assert "models" in loaded_bundle
            assert "metadata" in loaded_bundle

            model_kind = loaded_bundle["metadata"].get("model_kind")
            assert model_kind == "competing_risks"

            # This is the key validation that would fail with old storage.py
            horizons = loaded_bundle["metadata"]["horizons"]
            for H in horizons:
                model_key = f"first_hit_h{H}"
                assert model_key in loaded_bundle["models"], f"Missing {model_key}"

            # 6. Validate models can predict
            X_test = np.random.randn(1, 5)
            for H in horizons:
                model = loaded_bundle["models"][f"first_hit_h{H}"]
                probs = model.predict_proba(X_test)
                assert probs.shape == (1, 3)  # 1 sample, 3 classes
                assert np.allclose(probs.sum(), 1.0)  # Probabilities sum to 1

        finally:
            Path(temp_path).unlink()


class WarmupParityTest(TestCase):
    """Test that training and inference have matching warmup behavior."""

    def test_train_serve_parity_warmup(self):
        """Test that features properly warm up after max_warmup_bars."""
        import numpy as np
        import pandas as pd
        from quant_core.features import _compute_features, compute_max_warmup_bars

        n_bars = 300
        df = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=n_bars, freq='1h'),
            'close': 100 + np.random.randn(n_bars).cumsum(),
            'open': 100 + np.random.randn(n_bars).cumsum(),
            'high': 101 + np.random.randn(n_bars).cumsum(),
            'low': 99 + np.random.randn(n_bars).cumsum(),
            'volume': 1000 + np.random.randn(n_bars) * 100,
        })

        df_features = _compute_features(df)
        max_warmup = compute_max_warmup_bars()

        # Check: volZScore should have NaNs in early rows
        self.assertTrue(
            df_features['volZScore'].iloc[:max_warmup-1].isna().any(),
            "Early rows should have NaN in volZScore before warmup completes"
        )

        # Check: After max_warmup, volZScore should be fully warmed (no NaNs)
        warmed_rows = df_features['volZScore'].iloc[max_warmup:]
        self.assertFalse(
            warmed_rows.isna().any(),
            "Rows after max_warmup should not have NaN in volZScore"
        )

        # Check: has_full_warmup indicator matches warmup status
        self.assertEqual(
            df_features['has_full_warmup'].iloc[max_warmup-1], 0,
            "has_full_warmup should be 0 before max_warmup"
        )
        self.assertEqual(
            df_features['has_full_warmup'].iloc[max_warmup], 1,
            "has_full_warmup should be 1 at max_warmup"
        )
        self.assertTrue(
            df_features['has_full_warmup'].iloc[max_warmup:].eq(1).all(),
            "has_full_warmup should be 1 for all rows after max_warmup"
        )
