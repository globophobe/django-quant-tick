import hashlib

from django.conf import settings
from django.db import models
from quant_core.constants import DEFAULT_ASYMMETRIES, DEFAULT_WIDTHS

from quant_tick.utils import gettext_lazy as _

from .base import AbstractCodeName, JSONField


def upload_artifact_to(instance: "MLArtifact", filename: str) -> str:
    """Upload artifact to."""
    prefix = "test-ml" if settings.TEST else "ml"
    code_name = instance.ml_config.code_name
    return f"{prefix}/artifacts/{code_name}/{filename}"


class MLConfig(AbstractCodeName):
    """ML Config for range breach risk prediction.

    This ML pipeline predicts the probability that price will touch (breach) the
    upper or lower bounds of a liquidity range within a given time horizon.

    How it works:
    1. Train competing-risks models that predict which barrier gets hit first
       (upper bound, lower bound, or neither within horizon)
    2. Each model outputs P(UP_FIRST), P(DOWN_FIRST), P(TIMEOUT) for a horizon
    3. Multiple models trained for different horizons (e.g., 48/96/144 bars)
    4. Calibrate predictions using isotonic regression to fix miscalibration
    5. Enforce monotonicity: longer horizons must have equal or higher exit probability
    6. Filter ranges: reject any range where total exit risk exceeds touch_tolerance

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
        related_name="ml_config",
    )
    symbol = models.ForeignKey(
        "quant_tick.Symbol",
        on_delete=models.CASCADE,
        verbose_name=_("symbol"),
        related_name="ml_config",
    )
    inference_lookback = models.IntegerField(
        _("inference lookback"),
        default=150,
        help_text=_(
            "Number of bars to fetch for inference feature computation. "
            "Must be >= 120 (max warmup for volZScore). "
            "Recommended: 150 (120 warmup + 30 buffer)."
        ),
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
    json_data = JSONField(_("json data"), default=dict)
    last_processed_timestamp = models.DateTimeField(
        _("last processed timestamp"),
        null=True,
        blank=True,
        help_text=_("Timestamp of last processed bar for idempotency"),
    )
    is_active = models.BooleanField(_("active"), default=False)

    def clean(self) -> None:
        """Validate model fields."""
        from django.core.exceptions import ValidationError
        from quant_core.features import compute_max_warmup_bars

        super().clean()

        max_warmup = compute_max_warmup_bars()
        if self.inference_lookback < max_warmup:
            raise ValidationError({
                'inference_lookback': (
                    f"Must be >= {max_warmup} bars. Features like volZScore "
                    f"require {max_warmup} bars to compute without NaN/sentinel values. "
                    f"Recommended: 150 (120 warmup + 30 buffer)."
                )
            })

    def get_decision_horizons(self) -> list[int]:
        """Get decision horizons with fallback to sensible default."""
        return self.json_data.get("decision_horizons", [60, 120, 180])

    def get_widths(self) -> list[float]:
        """Get widths with fallback to DEFAULT_WIDTHS."""
        return self.json_data.get("widths", DEFAULT_WIDTHS)

    def get_asymmetries(self) -> list[float]:
        """Get asymmetries with fallback to DEFAULT_ASYMMETRIES."""
        return self.json_data.get("asymmetries", DEFAULT_ASYMMETRIES)

    def get_exposures(self) -> list[float]:
        """Get exposures with fallback to default symmetric grid.

        Exposure represents signed BTC notional per $1 equity:
        - Positive: long BTC (bullish)
        - Negative: short BTC (bearish, BTC liability)
        - Zero: neutral

        Returns:
            List of exposure values (e.g., [-1.0, -0.5, 0.0, 0.5, 1.0])
        """
        return self.json_data.get("exposures", [-1.0, -0.5, 0.0, 0.5, 1.0])

    def get_perps_params(self) -> dict:
        """Get perps parameters with defaults.

        Returns dict with:
            widths: Symmetric TP/SL distances (default [0.01, 0.02, 0.03])
            conf_threshold: Min directional confidence (default 0.60)
            move_threshold: Min move probability (default 0.40)
            trail_pct: Trailing stop percentage (default 0.015 = 1.5%)
        """
        defaults = {
            "widths": [0.01, 0.02, 0.03],
            "conf_threshold": 0.60,
            "move_threshold": 0.40,
            "trail_pct": 0.015,
        }
        stored = self.json_data.get("perps_params", {})
        return {**defaults, **stored}

    def set_perps_params(self, **params) -> None:
        """Update perps parameters in json_data."""
        if "perps_params" not in self.json_data:
            self.json_data["perps_params"] = {}
        self.json_data["perps_params"].update(params)

    def get_directional2_params(self) -> dict:
        """Get directional2 parameters with defaults.

        Returns dict with:
            horizons: Forward horizons to train (default [48, 96])
            k: Volatility multiplier for thresholds (default 0.25)
            vol_window: Rolling vol window for labels (default 48)
            p_threshold: Min P(class) for non-neutral inference (default 0.55)
            conf_threshold: Min directional confidence (default 0.40)
        """
        defaults = {
            "horizons": [48, 96],
            "k": 0.25,
            "vol_window": 48,
            "p_threshold": 0.55,
            "conf_threshold": 0.40,
        }
        stored = self.json_data.get("directional2_params", {})
        return {**defaults, **stored}

    def set_directional2_params(self, **params) -> None:
        """Update directional2 parameters in json_data.

        Valid keys: horizons, k, vol_window, p_threshold, conf_threshold.
        """
        if "directional2_params" not in self.json_data:
            self.json_data["directional2_params"] = {}
        self.json_data["directional2_params"].update(params)

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

        Returns dict with keys: retrain_cadence_days, train_window_days, holdout_days, policy_mode.
        """
        defaults = {
            "retrain_cadence_days": 7,
            "train_window_days": 84,
            "holdout_days": None,
            "policy_mode": "lp",
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

    def get_horizons(self, max_days: int = 3) -> list[int]:
        """Get prediction horizons in bars based on candle frequency.

        If json_data["horizons"] is set, uses those explicit horizons.
        Otherwise, for candles with stable frequency (TimeBasedCandle or
        AdaptiveCandle with target_candles_per_day), returns horizons
        corresponding to 1-max_days.

        For adaptive candles without explicit target, returns activity-based horizons
        (not wall-clock days).

        Args:
            max_days: Maximum number of days for horizons

        Returns:
            List of horizons in bars (e.g., [24, 48, 72, ...] for 1h candles)

        Raises:
            ValueError: If cannot derive horizons from candle configuration or
                if custom horizons are invalid
        """
        import logging

        import pandas as pd

        if "horizons" in self.json_data:
            horizons_raw = self.json_data["horizons"]

            if horizons_raw is None:
                pass
            else:
                if not isinstance(horizons_raw, (list, tuple)):
                    raise ValueError("horizons must be a list of positive integers")

                if not horizons_raw:
                    raise ValueError("horizons must contain at least one positive integer")

                if not all(
                    not isinstance(h, bool) and isinstance(h, int) and h > 0
                    for h in horizons_raw
                ):
                    raise ValueError("horizons must contain only positive integers")

                horizons = sorted(set(horizons_raw))

                return horizons

        candle = self.candle
        logger = logging.getLogger(__name__)

        if "target_candles_per_day" in candle.json_data:
            bars_per_day = candle.json_data["target_candles_per_day"]
            return [bars_per_day * d for d in range(1, max_days + 1)]

        if "window" in candle.json_data:
            delta = pd.Timedelta(candle.json_data["window"])
            bars_per_day = int(24 / (delta.total_seconds() / 3600))
            return [bars_per_day * d for d in range(1, max_days + 1)]

        logger.warning(
            f"{candle}: Cannot derive bars_per_day (adaptive), "
            f"using activity-based horizons (~50 bars/day, {max_days} days)"
        )
        bars_per_activity_day = 50
        return [bars_per_activity_day * d for d in range(1, max_days + 1)]

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
    is_production = models.BooleanField(
        _("is production"),
        default=False,
        help_text=_("Whether this model is deployed to quant_horizon"),
    )
    gcs_path = models.CharField(
        _("gcs path"),
        max_length=512,
        blank=True,
        help_text=_("GCS path: gs://bucket/path/model.joblib"),
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

    class Meta:
        db_table = "quant_tick_ml_artifact"
        verbose_name = _("ml artifact")
        verbose_name_plural = _("ml artifacts")
        constraints = [
            models.UniqueConstraint(
                fields=["ml_config"],
                condition=models.Q(is_production=True),
                name="unique_production_model_per_config",
            )
        ]


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
    json_data = JSONField(_("json data"), null=True, blank=True)

    class Meta:
        db_table = "quant_tick_ml_signal"
        verbose_name = _("ml signal")
        verbose_name_plural = _("ml signals")
        ordering = ["-timestamp"]
        indexes = [
            models.Index(fields=["ml_config", "timestamp"]),
        ]
