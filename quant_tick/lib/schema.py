from typing import Any

import numpy as np


class MLSchema:
    """ML Schema."""

    # Metadata columns, excluded from features
    METADATA_COLS = {
        "timestamp",
        "bar_idx",
        "config_id",
        "entry_price",
    }

    # Config columns, dynamically added for prediction
    CONFIG_COLS = {
        "width",
        "asymmetry",
        "lower_bound_pct",
        "upper_bound_pct",
        "range_width",
        "range_asymmetry",
        "dist_to_lower_pct",
        "dist_to_upper_pct",
        "k",  # Time index, added during hazard expansion
    }

    # Hazard-specific metadata (k is included but kept as feature)
    HAZARD_METADATA_COLS = {
        "timestamp",
        "bar_idx",
        "config_id",
        "entry_price",
        "k",
    }

    @staticmethod
    def get_data_features(all_cols: list[str], horizons: list[int]) -> list[str]:
        """Get data features.

        Config cols like width, asymmetry, range_width are added dynamically when
        testing multiple configs per bar during inference. This method returns
        features that must be present in the input data.

        Filters out:
        - CONFIG_COLS (k, width, asymmetry, bounds, etc.)
        - *_missing indicators (created by prepare_features)

        Args:
            all_cols: All column names expected during training
            horizons: List of decision horizons (unused, kept for backward compatibility)

        Returns:
            List of feature names that must exist in input data
        """
        training_features = MLSchema.get_training_features(all_cols)
        return [
            c for c in training_features
            if c not in MLSchema.CONFIG_COLS and not c.endswith("_missing")
        ]

    @staticmethod
    def validate_bar_config_structure(
        df: Any,
        widths: list[float],
        asymmetries: list[float],
    ) -> tuple[bool, str]:
        """Validate dataframe has complete (bar_idx, config_id) structure.

        Ensures that after timestamp filtering or other operations,
        the dataframe still has all configs for each bar.

        Args:
            df: DataFrame to validate
            widths: Expected range widths
            asymmetries: Expected asymmetries

        Returns:
            Tuple of (is_valid, error_message)
        """
        n_configs = len(widths) * len(asymmetries)

        # Check required columns exist
        if "bar_idx" not in df.columns or "config_id" not in df.columns:
            return False, "Missing bar_idx or config_id columns"

        # Check each bar has exactly n_configs rows
        bar_counts = df.groupby("bar_idx").size()
        incomplete_bars = bar_counts[bar_counts != n_configs]

        if len(incomplete_bars) > 0:
            first_bad = incomplete_bars.index[0]
            return False, f"Bar {first_bad} has {incomplete_bars.iloc[0]} configs (expected {n_configs})"

        # Check config_id values are valid
        unique_configs = sorted(df["config_id"].unique())
        expected_configs = list(range(n_configs))
        if unique_configs != expected_configs:
            return False, f"Config IDs mismatch: got {unique_configs}, expected {expected_configs}"

        return True, ""

    @staticmethod
    def validate_bar_config_invariants(df: Any) -> tuple[bool, str]:
        """Validate bar_idx and config_id invariants.

        Checks:
        - bar_idx is monotonically increasing
        - config_id is contiguous 0..(n_configs-1) per bar

        Args:
            df: DataFrame to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if "bar_idx" not in df.columns:
            return False, "Missing bar_idx column"
        if "config_id" not in df.columns:
            return False, "Missing config_id column"

        # Check bar_idx monotonicity
        bar_idx_vals = df["bar_idx"].values
        if not np.all(bar_idx_vals[1:] >= bar_idx_vals[:-1]):
            return False, "bar_idx is not monotonically increasing"

        # Check config_id per bar
        for bar_idx in df["bar_idx"].unique():
            bar_rows = df[df["bar_idx"] == bar_idx]
            config_ids = sorted(bar_rows["config_id"].values)
            expected = list(range(len(config_ids)))
            if config_ids != expected:
                return False, f"bar_idx={bar_idx} has invalid config_ids: {config_ids}, expected {expected}"

        return True, ""

    @staticmethod
    def get_hazard_label_cols() -> set[str]:
        """Get hazard label column names.

        Returns:
            Set of hazard label column names
        """
        return {
            "hazard_lower",
            "hazard_upper",
            "event_lower",
            "event_upper",
        }

    @staticmethod
    def get_training_features(all_cols: list[str]) -> list[str]:
        """Get training features for survival models.

        Excludes metadata and labels, but INCLUDES k as a feature.
        This is critical: k (time since entry) is a predictor.

        Args:
            all_cols: All column names from training DataFrame

        Returns:
            List of feature column names (includes k, excludes labels)
        """
        exclude = (MLSchema.HAZARD_METADATA_COLS - {"k"}) | MLSchema.get_hazard_label_cols()
        return [c for c in all_cols if c not in exclude]

    @staticmethod
    def validate_schema(
        df: Any,
        widths: list[float],
        asymmetries: list[float],
        max_horizon: int,
    ) -> tuple[bool, str]:
        """Validate training dataframe schema.

        Checks:
        - Required columns exist (bar_idx, config_id, k, label columns)
        - k range is 1..max_horizon
        - Row count = n_bars × n_configs × max_horizon
        - Config values match expected widths/asymmetries
        - Config structure is valid (complete grid)

        Args:
            df: Training DataFrame
            widths: Expected width values
            asymmetries: Expected asymmetry values
            max_horizon: Expected max k value

        Returns:
            (is_valid, error_message)
        """
        if "k" not in df.columns:
            return False, "Missing k column"

        k_vals = df["k"].unique()
        k_min, k_max = int(k_vals.min()), int(k_vals.max())
        if k_min != 1 or k_max != max_horizon:
            return False, f"k range {k_min}-{k_max} != expected 1-{max_horizon}"

        for col in MLSchema.get_hazard_label_cols():
            if col not in df.columns:
                return False, f"Missing hazard label column: {col}"

        if "width" not in df.columns or "asymmetry" not in df.columns:
            return False, "Missing width/asymmetry columns"

        actual_widths = sorted(df["width"].unique())
        actual_asymmetries = sorted(df["asymmetry"].unique())

        if list(sorted(widths)) != list(actual_widths):
            return False, f"Width mismatch: expected={widths}, actual={actual_widths}"

        if list(sorted(asymmetries)) != list(actual_asymmetries):
            return False, f"Asymmetry mismatch: expected={asymmetries}, actual={actual_asymmetries}"

        n_configs = len(widths) * len(asymmetries)
        unique_bars = df["bar_idx"].nunique()
        expected_rows = unique_bars * n_configs * max_horizon

        if len(df) != expected_rows:
            return False, (
                f"Row count {len(df)} != expected {expected_rows} "
                f"({unique_bars} bars × {n_configs} configs × {max_horizon} time steps)"
            )

        for bar_idx in df["bar_idx"].unique():
            for k in range(1, max_horizon + 1):
                bar_k_rows = df[(df["bar_idx"] == bar_idx) & (df["k"] == k)]

                if len(bar_k_rows) != n_configs:
                    return False, (
                        f"bar_idx={bar_idx}, k={k} has {len(bar_k_rows)} configs, "
                        f"expected {n_configs}"
                    )

                config_ids = sorted(bar_k_rows["config_id"].values)
                expected_ids = list(range(n_configs))
                if config_ids != expected_ids:
                    return False, (
                        f"bar_idx={bar_idx}, k={k} has invalid config_ids: {config_ids}, "
                        f"expected {expected_ids}"
                    )

        return True, ""
