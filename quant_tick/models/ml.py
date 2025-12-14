import hashlib

from django.conf import settings
from django.db import models

# Import for schema validation
from quant_tick.lib.ml import DEFAULT_ASYMMETRIES, DEFAULT_WIDTHS
from quant_tick.lib.schema import MLSchema
from quant_tick.utils import gettext_lazy as _

from .base import AbstractCodeName, AbstractDataStorage, JSONField


def upload_artifact_to(instance: "MLArtifact", filename: str) -> str:
    """Upload artifact to."""
    prefix = "test-ml" if settings.TEST else "ml"
    code_name = instance.ml_config.code_name
    return f"{prefix}/artifacts/{code_name}/{filename}"


def upload_feature_data_to(instance: "MLFeatureData", filename: str) -> str:
    """Upload feature data to."""
    prefix = "test-ml" if settings.TEST else "ml"
    code_name = instance.candle.code_name
    return f"{prefix}/features/{code_name}/{filename}"


class MLConfig(AbstractCodeName):
    """ML Config for range breach risk prediction.

    This ML pipeline predicts the probability that price will touch (breach) the
    upper or lower bounds of a liquidity range within a given time horizon.

    How it works:
    1. Train survival models using discrete-time hazard functions for each side
       (lower bound, upper bound)
    2. Each model predicts h(k) = P(first touch at step k | survived to k-1)
    3. Reconstruct survival curves S(k) and compute P(touch by horizon H) via
       hazard_to_per_horizon_probs for decision horizons (e.g., 60/120/180 bars)
    4. Calibrate predictions using isotonic regression to fix miscalibration
    5. Enforce monotonicity: longer horizons must have equal or higher touch probability
    6. Filter ranges: reject any range where total touch risk exceeds touch_tolerance

    What it is:
    - A risk filter that screens out ranges likely to get breached
    - A dual-barrier probability estimator for range survival

    Use this to avoid placing LP positions in ranges that are too tight for current
    market conditions.

    The key tradeoff:
    - Lower touch_tolerance = wider ranges selected = lower APY but safer for LP
    - Higher touch_tolerance = tighter ranges allowed = higher APY but more breach risk
    """

    candle = models.ForeignKey(
        "quant_tick.Candle",
        on_delete=models.CASCADE,
        verbose_name=_("candle"),
        related_name="ml_config"
    )
    symbol = models.ForeignKey(
        "quant_tick.Symbol",
        on_delete=models.CASCADE,
        verbose_name=_("symbol"),
        related_name="ml_config",
    )
    inference_lookback = models.IntegerField(
        _("inference lookback"),
        default=100,
        help_text=_("Number of bars to fetch for inference feature computation"),
    )
    horizon_bars = models.IntegerField(
        _("horizon bars"),
        default=180,
        help_text=_("Number of bars for touch prediction horizon"),
    )
    touch_tolerance = models.FloatField(
        _("touch tolerance"),
        default=0.15,
        help_text=_(
            "Max acceptable P(touch) for valid config. "
            "Lower = conservative (LP), Higher = aggressive (perp)"
        ),
    )
    min_hold_bars = models.IntegerField(
        _("min hold bars"),
        default=15,
        help_text=_("Minimum bars before position change"),
    )
    json_data = JSONField(
        _("json data"),
        default=dict
    )
    last_processed_timestamp = models.DateTimeField(
        _("last processed timestamp"),
        null=True,
        blank=True,
        help_text=_("Timestamp of last processed bar for idempotency"),
    )
    is_active = models.BooleanField(_("active"), default=False)

    def get_decision_horizons(self) -> list[int]:
        """Get decision horizons with fallback to sensible default."""
        return self.json_data.get("decision_horizons", [60, 120, 180])

    def get_widths(self) -> list[float]:
        """Get widths with fallback to DEFAULT_WIDTHS."""
        return self.json_data.get("widths", DEFAULT_WIDTHS)

    def get_asymmetries(self) -> list[float]:
        """Get asymmetries with fallback to DEFAULT_ASYMMETRIES."""
        return self.json_data.get("asymmetries", DEFAULT_ASYMMETRIES)

    def get_min_bars(self) -> int:
        """Get minimum bars required for label generation."""
        return self.json_data.get("min_bars", 1000)

    def get_training_params(self) -> dict:
        """Get training parameters with defaults.

        Returns dict with keys: n_splits, embargo_bars, n_estimators, max_depth,
        min_samples_leaf, learning_rate, subsample, holdout_pct, calibration_pct,
        min_train_bars_purged.
        """
        defaults = {
            "n_splits": 5,
            "embargo_bars": 96,
            "n_estimators": 300,
            "max_depth": 6,
            "min_samples_leaf": 50,
            "learning_rate": 0.05,
            "subsample": 0.75,
            "holdout_pct": 0.2,
            "calibration_pct": 0.1,
            "min_train_bars_purged": 100,
        }
        stored = self.json_data.get("training_params", {})
        return {**defaults, **stored}

    def get_optuna_n_trials(self) -> int:
        """Get Optuna n_trials with default."""
        return self.json_data.get("optuna_n_trials", 20)

    def set_training_params(self, **params) -> None:
        """Update training parameters in json_data.

        Valid keys: n_splits, embargo_bars, n_estimators, max_depth,
        min_samples_leaf, learning_rate, subsample, holdout_pct, calibration_pct,
        min_train_bars_purged.
        """
        if "training_params" not in self.json_data:
            self.json_data["training_params"] = {}
        self.json_data["training_params"].update(params)

    def get_simulation_params(self) -> dict:
        """Get simulation parameters with defaults.

        Returns dict with keys: retrain_cadence_days, train_window_days, holdout_days.
        """
        defaults = {
            "retrain_cadence_days": 7,
            "train_window_days": 84,
            "holdout_days": None,
        }
        stored = self.json_data.get("simulation_params", {})
        return {**defaults, **stored}

    def set_simulation_params(self, **params) -> None:
        """Update simulation parameters in json_data.

        Valid keys: retrain_cadence_days, train_window_days, holdout_days.
        """
        if "simulation_params" not in self.json_data:
            self.json_data["simulation_params"] = {}
        self.json_data["simulation_params"].update(params)

    class Meta:
        db_table = "quant_tick_ml_config"
        verbose_name = verbose_name_plural = _("ml config")


class MLArtifact(models.Model):
    """ML Artifact."""

    ml_config = models.ForeignKey(
        "quant_tick.MLConfig",
        on_delete=models.CASCADE,
        verbose_name=_("ml config"),
        related_name="ml_artifacts",
    )
    model_type = models.CharField(
        _("model type"),
        max_length=20,
        help_text=_("upper or lower"),
    )
    artifact = models.FileField(_("artifact"), upload_to=upload_artifact_to)
    brier_score = models.FloatField(
        _("brier score"),
        null=True,
        blank=True,
        help_text=_("Cross-validation metric"),
    )
    feature_columns = JSONField(
        _("feature columns"),
        default=list,
        help_text=_("Ordered list of feature column names"),
    )
    sha256 = models.CharField(_("sha256"), max_length=64, blank=True)

    def save(self, *args, **kwargs) -> "MLArtifact":
        """Save with SHA256 hash."""
        if self.artifact and not self.sha256:
            hasher = hashlib.sha256()
            for chunk in self.artifact.chunks():
                hasher.update(chunk)
            self.sha256 = hasher.hexdigest()
        return super().save(*args, **kwargs)

    calibrator = models.BinaryField(
        _("calibrator"),
        null=True,
        blank=True,
        help_text=_(
            "Deprecated: calibrator is stored in joblib artifact. "
            "This field is redundant and not used by inference."
        ),
    )
    horizon = models.IntegerField(
        _("horizon"),
        null=True,
        blank=True,
        help_text=_("Horizon in bars, for per-horizon models."),
    )
    calibration_method = models.CharField(
        _("calibration method"),
        max_length=20,
        default="none",
        choices=[("none", "None"), ("isotonic", "Isotonic"), ("platt", "Platt")],
        help_text=_("Calibration method."),
    )
    json_data = JSONField(
        _("json data"),
        default=dict,
        help_text=_("Data including base rates, calibration drift, etc."),
    )

    class Meta:
        db_table = "quant_tick_ml_artifact"
        verbose_name = _("ml artifact")
        verbose_name_plural = _("ml artifacts")


class MLFeatureData(AbstractDataStorage):
    """ML Feature Data."""

    candle = models.ForeignKey(
        "quant_tick.Candle",
        on_delete=models.CASCADE,
        verbose_name=_("candle"),
        related_name="ml_feature_data",
    )
    timestamp_from = models.DateTimeField(_("timestamp from"))
    timestamp_to = models.DateTimeField(_("timestamp to"))
    file_data = models.FileField(
        _("file data"), upload_to=upload_feature_data_to, blank=True
    )
    schema_hash = models.CharField(_("schema hash"), max_length=64, blank=True)

    def validate_schema(self, config: "MLConfig") -> tuple[bool, str]:
        """Validate stored schema matches config requirements.

        Returns:
            Tuple of (is_valid, error_message)
        """
        df = self.get_data_frame("file_data")
        if df is None or df.empty:
            return False, _("No feature data.")

        widths = config.json_data.get("widths", DEFAULT_WIDTHS)
        asymmetries = config.json_data.get("asymmetries", DEFAULT_ASYMMETRIES)
        max_horizon = self.json_data.get("max_horizon", config.horizon_bars)

        return MLSchema.validate_schema(df, widths, asymmetries, max_horizon)

    class Meta:
        db_table = "quant_tick_ml_feature_data"
        verbose_name = verbose_name_plural = _("ml feature data")
        ordering = ["-timestamp_to"]


class MLSignal(models.Model):
    """ML Signal."""

    ml_config = models.ForeignKey(
        "quant_tick.MLConfig",
        on_delete=models.CASCADE,
        verbose_name=_("ml config"),
        related_name="ml_signals",
    )
    timestamp = models.DateTimeField(_("timestamp"), db_index=True)
    lower_bound = models.FloatField(
        _("lower bound"),
        help_text=_("Lower bound as fraction of entry price (e.g., -0.03)"),
    )
    upper_bound = models.FloatField(
        _("upper bound"),
        help_text=_("Upper bound as fraction of entry price (e.g., 0.05)"),
    )
    borrow_ratio = models.FloatField(
        _("borrow ratio"),
        help_text=_("Borrow ratio for DBI (0.5 = balanced)"),
    )
    p_touch_lower = models.FloatField(_("P(touch lower)"))
    p_touch_upper = models.FloatField(_("P(touch upper)"))
    json_data = JSONField(
        _("json data"),
        null=True,
        blank=True
    )

    class Meta:
        db_table = "quant_tick_ml_signal"
        verbose_name = _("ml signal")
        verbose_name_plural = _("ml signals")
        ordering = ["-timestamp"]
        indexes = [
            models.Index(fields=["ml_config", "timestamp"]),
        ]
