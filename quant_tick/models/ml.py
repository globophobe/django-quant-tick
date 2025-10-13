import hashlib

from django.conf import settings
from django.db import models

from quant_tick.utils import gettext_lazy as _

from .base import AbstractCodeName, AbstractDataStorage, BigDecimalField, JSONField


def upload_artifact_to(instance: "MLArtifact", filename: str) -> str:
    """Upload artifact to."""
    prefix = "test-ml" if settings.TEST else "ml"
    run_id = instance.ml_run.id
    return f"{prefix}/artifacts/run_{run_id}/{filename}"


def upload_feature_data_to(instance: "MLFeatureData", filename: str) -> str:
    """Upload feature data to."""
    prefix = "test-ml" if settings.TEST else "ml"
    candle_code = instance.candle.code_name
    return f"{prefix}/features/{candle_code}/{filename}"


class MLConfig(AbstractCodeName):
    """ML configuration."""

    candle = models.ForeignKey(
        "quant_tick.Candle",
        on_delete=models.CASCADE,
        verbose_name=_("candle"),
        related_name="ml_configs",
    )
    symbol = models.ForeignKey(
        "quant_tick.Symbol",
        on_delete=models.CASCADE,
        verbose_name=_("symbol"),
        related_name="ml_configs",
        help_text=_("Target symbol for trade execution"),
        null=True,
        blank=True,
    )
    json_data = JSONField(
        _("json data"),
        help_text=_(
            "symbols, time_window, features, labeling, cv, model_hparams, thresholds"
        ),
        default=dict,
    )
    status = models.CharField(_("status"), max_length=50, default="active")
    created_at = models.DateTimeField(_("created at"), auto_now_add=True)

    class Meta:
        db_table = "quant_tick_ml_config"
        verbose_name = verbose_name_plural = _("ml config")


class MLRun(models.Model):
    """ML run."""

    ml_config = models.ForeignKey(
        "quant_tick.MLConfig",
        on_delete=models.CASCADE,
        verbose_name=_("ml config"),
        related_name="ml_runs",
    )
    timestamp_from = models.DateTimeField(_("timestamp from"))
    timestamp_to = models.DateTimeField(_("timestamp to"))
    metrics = JSONField(_("metrics"), null=True, blank=True)
    feature_importances = JSONField(_("feature importances"), null=True, blank=True)
    metadata = JSONField(_("metadata"), null=True, blank=True)
    status = models.CharField(_("status"), max_length=50, default="pending")
    created_at = models.DateTimeField(_("created at"), auto_now_add=True)

    class Meta:
        db_table = "quant_tick_ml_run"
        verbose_name = _("ml run")
        verbose_name_plural = _("ml runs")
        ordering = ["-created_at"]


class MLArtifact(models.Model):
    """ML artifact."""

    ml_run = models.ForeignKey(
        "quant_tick.MLRun",
        on_delete=models.CASCADE,
        verbose_name=_("ml run"),
        related_name="ml_artifacts",
    )
    artifact = models.FileField(_("artifact"), upload_to=upload_artifact_to)
    version = models.CharField(_("version"), max_length=100)
    sha256 = models.CharField(_("sha256"), max_length=64, blank=True)

    def save(self, *args, **kwargs) -> "MLArtifact":
        """Save."""
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
    """ML feature data."""

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
    schema_version = models.CharField(_("schema version"), max_length=20, default="1.0")

    class Meta:
        db_table = "quant_tick_ml_feature_data"
        verbose_name = verbose_name_plural = _("ml feature data")
        ordering = ["-timestamp_to"]


class MLSignal(models.Model):
    """ML signal."""

    candle = models.ForeignKey(
        "quant_tick.Candle",
        on_delete=models.CASCADE,
        verbose_name=_("candle"),
        related_name="ml_signals",
    )
    ml_artifact = models.ForeignKey(
        "quant_tick.MLArtifact",
        on_delete=models.SET_NULL,
        verbose_name=_("ml artifact"),
        related_name="ml_signals",
        null=True,
        blank=True,
    )
    timestamp = models.DateTimeField(_("timestamp"), db_index=True)
    probability = BigDecimalField(_("probability"))
    side = models.SmallIntegerField(_("side"))
    meta_label = models.SmallIntegerField(_("meta label"), null=True, blank=True)
    size = BigDecimalField(_("size"), null=True, blank=True)
    json_data = JSONField(_("json data"), null=True, blank=True)

    class Meta:
        db_table = "quant_tick_ml_signal"
        verbose_name = _("ml signal")
        verbose_name_plural = _("ml signals")
        ordering = ["-timestamp"]
        indexes = [
            models.Index(fields=["candle", "timestamp"]),
        ]
