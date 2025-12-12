"""Tests for walk-forward simulation."""

import pandas as pd
from django.test import TestCase

from quant_tick.lib.schema import MLSchema


class MLSimulateSchemaValidationTest(TestCase):
    """Test MLSchema validation for grid structure in ml_simulate context."""

    def test_validate_bar_config_structure_rejects_incomplete_grid(self):
        """validate_bar_config_structure rejects incomplete config grids."""
        # Create incomplete grid (missing config_id=3 for bar 0)
        df = pd.DataFrame({
            "bar_idx": [0, 0, 0, 1, 1, 1, 1],
            "config_id": [0, 1, 2, 0, 1, 2, 3],
            "close": [100] * 7,
        })

        widths = [0.02, 0.04]
        asymmetries = [0.0, 0.5]

        is_valid, error_msg = MLSchema.validate_bar_config_structure(df, widths, asymmetries)

        self.assertFalse(is_valid)
        self.assertIn("Bar 0", error_msg)
        self.assertIn("3 configs", error_msg)
        self.assertIn("expected 4", error_msg)

    def test_validate_bar_config_structure_accepts_complete_grid(self):
        """validate_bar_config_structure accepts complete grids."""
        # Create complete grid
        df = pd.DataFrame({
            "bar_idx": [0, 0, 0, 0, 1, 1, 1, 1],
            "config_id": [0, 1, 2, 3, 0, 1, 2, 3],
            "close": [100] * 8,
        })

        widths = [0.02, 0.04]
        asymmetries = [0.0, 0.5]

        is_valid, error_msg = MLSchema.validate_bar_config_structure(df, widths, asymmetries)

        self.assertTrue(is_valid)
        self.assertEqual(error_msg, "")

    def test_validate_bar_config_structure_handles_timestamp_filtering(self):
        """validate_bar_config_structure catches incomplete grids after timestamp filtering."""
        # Simulate what happens after timestamp filtering removes some rows
        # Bar 5 lost config_id=0, bar 10 lost config_id=1
        df = pd.DataFrame({
            "bar_idx": [5, 5, 5, 10, 10, 10],
            "config_id": [1, 2, 3, 0, 2, 3],  # Missing 0 for bar 5, missing 1 for bar 10
            "close": [100] * 6,
        })

        widths = [0.02, 0.04]
        asymmetries = [0.0, 0.5]

        is_valid, error_msg = MLSchema.validate_bar_config_structure(df, widths, asymmetries)

        self.assertFalse(is_valid)
        self.assertIn("3 configs", error_msg)

    def test_validate_bar_config_structure_requires_all_configs_per_bar(self):
        """Each bar must have exactly n_configs rows."""
        # Bar 0 has all 4 configs, but bar 1 only has 2
        df = pd.DataFrame({
            "bar_idx": [0, 0, 0, 0, 1, 1],
            "config_id": [0, 1, 2, 3, 0, 1],
            "close": [100] * 6,
        })

        widths = [0.02, 0.04]
        asymmetries = [0.0, 0.5]

        is_valid, error_msg = MLSchema.validate_bar_config_structure(df, widths, asymmetries)

        self.assertFalse(is_valid)
        self.assertIn("Bar 1", error_msg)
        self.assertIn("2 configs", error_msg)

    def test_validate_bar_config_structure_validates_config_ids(self):
        """config_id values must be valid range."""
        # Bar 0 has wrong config_ids (should be 0,1,2,3 but got 0,1,5,6)
        df = pd.DataFrame({
            "bar_idx": [0, 0, 0, 0],
            "config_id": [0, 1, 5, 6],  # Invalid: should be 0,1,2,3
            "close": [100] * 4,
        })

        widths = [0.02, 0.04]
        asymmetries = [0.0, 0.5]

        is_valid, error_msg = MLSchema.validate_bar_config_structure(df, widths, asymmetries)

        self.assertFalse(is_valid)
        self.assertIn("Config IDs mismatch", error_msg)
