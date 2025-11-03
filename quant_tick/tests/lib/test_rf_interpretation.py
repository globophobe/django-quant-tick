from unittest.mock import MagicMock, patch

import numpy as np
from django.test import TestCase
from sklearn.ensemble import RandomForestClassifier

from quant_tick.lib.rf_interpretation import (
    generate_pdp_plots,
    get_feature_contribution_summary,
)


class RFInterpretationTestCase(TestCase):
    """Test RF interpretation functions."""

    def setUp(self):
        """Create test fixtures."""
        np.random.seed(42)
        n_samples = 100
        n_features = 5

        self.X = np.random.randn(n_samples, n_features)
        self.y = np.random.randint(0, 2, n_samples)
        self.feature_names = [f"feature_{i}" for i in range(n_features)]

        self.model = RandomForestClassifier(n_estimators=10, random_state=42)
        self.model.fit(self.X, self.y)

    def test_generate_pdp_plots(self):
        """Test PDP plot generation."""
        plots = generate_pdp_plots(
            self.model,
            self.X,
            self.feature_names,
            top_k_features=3,
            kind="both",
        )

        self.assertIsInstance(plots, dict)
        # Should generate plots for top features
        self.assertLessEqual(len(plots), 3)

    def test_get_feature_contribution_summary(self):
        """Test feature contribution summary."""
        contributions = [
            {"feature_0": 0.5, "feature_1": -0.3, "feature_2": 0.1},
            {"feature_0": 0.2, "feature_1": -0.5, "feature_2": 0.3},
            {"feature_0": 0.4, "feature_1": -0.2, "feature_2": 0.2},
        ]

        summary = get_feature_contribution_summary(contributions, top_k=3)

        self.assertIn("top_features", summary)
        self.assertIn("mean_abs_contributions", summary)

        top_features = summary["top_features"]
        self.assertEqual(len(top_features), 3)
        self.assertIn("feature", top_features[0])
        self.assertIn("mean_abs_contribution", top_features[0])

        self.assertGreater(
            top_features[0]["mean_abs_contribution"],
            0.0
        )

    def test_get_feature_contribution_summary_empty(self):
        """Test feature contribution summary with empty input."""
        summary = get_feature_contribution_summary([], top_k=10)

        self.assertEqual(summary["top_features"], [])
        self.assertEqual(summary["mean_abs_contributions"], {})

    @patch('sklearn.inspection.PartialDependenceDisplay')
    @patch('matplotlib.pyplot')
    def test_generate_pdp_plots_respects_top_k(self, mock_plt, mock_pdp_display):
        """Test that PDP generation respects top_k parameter."""
        mock_fig = MagicMock()
        mock_ax = MagicMock()
        mock_plt.subplots.return_value = (mock_fig, mock_ax)
        mock_plt.close = MagicMock()

        plots = generate_pdp_plots(
            self.model,
            self.X,
            self.feature_names,
            top_k_features=2,
            kind="average",
        )

        self.assertLessEqual(len(plots), 2)
