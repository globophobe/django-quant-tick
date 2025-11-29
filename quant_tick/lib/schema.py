from typing import Any


class MLSchema:
    """ML Schema."""

    # Metadata columns, excluded from features
    METADATA_COLS = {
        "timestamp",
        "timestamp_idx",
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
    }

    @staticmethod
    def get_label_cols(horizons: list[int]) -> set[str]:
        """Get label column names for given horizons.

        Args:
            horizons: List of decision horizons (e.g., [60, 120, 180])

        Returns:
            Set of label column names
        """
        labels = set()
        for h in horizons:
            labels.add(f"hit_lower_by_{h}")
            labels.add(f"hit_upper_by_{h}")
        return labels

    @staticmethod
    def get_training_features(all_cols: list[str], horizons: list[int]) -> list[str]:
        """Get training features.

        Args:
            all_cols: All column names in DataFrame
            horizons: List of decision horizons

        Returns:
            List of feature column names for training
        """
        exclude = MLSchema.METADATA_COLS | MLSchema.get_label_cols(horizons)
        return [c for c in all_cols if c not in exclude]

    @staticmethod
    def get_data_features(all_cols: list[str], horizons: list[int]) -> list[str]:
        """Get data features.

        Config cols like width, asymmetry, range_width are added dynamically when
        testing multiple configs per bar during inference. This method returns
        features that must be present in the input candle data.

        Args:
            all_cols: All column names expected during training
            horizons: List of decision horizons

        Returns:
            List of feature names that must exist in input data
        """
        training_features = MLSchema.get_training_features(all_cols, horizons)
        return [c for c in training_features if c not in MLSchema.CONFIG_COLS]

    @staticmethod
    def validate_schema(
        df: Any,
        widths: list[float],
        asymmetries: list[float],
        horizons: list[int],
    ) -> tuple[bool, str]:
        """Validate dataframe matches schema.

        Args:
            df: DataFrame to validate
            widths: Expected range widths
            asymmetries: Expected asymmetries
            horizons: Expected decision horizons

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check config columns exist
        if "width" not in df.columns or "asymmetry" not in df.columns:
            return False, "Missing width/asymmetry columns"

        # Verify config values match
        actual_widths = sorted(df["width"].unique())
        actual_asymmetries = sorted(df["asymmetry"].unique())

        if list(sorted(widths)) != list(actual_widths):
            return False, f"Width mismatch: expected={widths}, actual={actual_widths}"

        if list(sorted(asymmetries)) != list(actual_asymmetries):
            return (
                False,
                f"Asymmetry mismatch: expected={asymmetries}, actual={actual_asymmetries}",
            )

        # Verify horizon labels exist
        for h in horizons:
            if f"hit_lower_by_{h}" not in df.columns:
                return False, f"Missing label column hit_lower_by_{h}"
            if f"hit_upper_by_{h}" not in df.columns:
                return False, f"Missing label column hit_upper_by_{h}"

        # Verify row count divisible by n_configs
        n_configs = len(widths) * len(asymmetries)
        if len(df) % n_configs != 0:
            return False, f"Row count {len(df)} not divisible by {n_configs} configs"

        return True, ""

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
