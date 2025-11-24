"""ML models for LP range optimization."""

import hashlib

from django.conf import settings
from django.db import models

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
    """ML Config."""

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
    lookback_bars = models.IntegerField(
        _("lookback bars"),
        default=100,
        help_text=_("Number of bars for feature lookback window"),
    )
    horizon_bars = models.IntegerField(
        _("horizon bars"),
        default=60,
        help_text=_("Number of bars for touch prediction horizon"),
    )
    touch_tolerance = models.FloatField(
        _("touch tolerance"),
        default=0.15,
        help_text=_("Max acceptable P(touch) for valid config"),
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

    class Meta:
        db_table = "quant_tick_ml_config"
        verbose_name = verbose_name_plural = _("ml config")


class MLArtifact(models.Model):
    """ML model artifact (serialized Random Forest model)."""

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
