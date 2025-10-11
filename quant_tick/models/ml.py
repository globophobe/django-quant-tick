import hashlib
from django.db import models
from quant_tick.utils import gettext_lazy as _
from .base import AbstractCodeName, JSONField, AbstractDataStorage, BigDecimalField


def upload_artifact_to(instance: "MLArtifact", filename: str) -> str:
    """Upload artifact to."""
    run_id = instance.ml_run.id
    return f"ml/artifacts/run_{run_id}/{filename}"


def upload_feature_data_to(instance: "MLFeatureData", filename: str) -> str:
    """Upload feature data to."""
    candle_id = instance.candle.id
    return f"ml/features/candle_{candle_id}/{filename}"


class MLConfig(AbstractCodeName):
    """ML configuration."""

    candle = models.ForeignKey(
        "quant_tick.Candle",
        on_delete=models.CASCADE,
        verbose_name=_("candle"),
        related_name="ml_configs",
    )
    config_json = JSONField(
        _("config json"),
        help_text=_("symbols, time_window, features, labeling, cv, model_hparams, thresholds"),
    )
    status = models.CharField(_("status"), max_length=50, default="active")
    created_at = models.DateTimeField(_("created at"), auto_now_add=True)

    class Meta:
        db_table = "quant_tick_ml_config"
        verbose_name = _("ml config")
        verbose_name_plural = _("ml configs")


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
    metrics_json = JSONField(_("metrics json"), null=True, blank=True)
    feature_importances_json = JSONField(_("feature importances json"), null=True, blank=True)
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

    def save(self, *args, **kwargs):
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
    data = models.FileField(_("data"), upload_to=upload_feature_data_to)
    schema_hash = models.CharField(_("schema hash"), max_length=64, blank=True)

    class Meta:
        db_table = "quant_tick_ml_feature_data"
        verbose_name = _("ml feature data")
        verbose_name_plural = _("ml feature data")
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
    position_size = BigDecimalField(_("position size"), null=True, blank=True)
    notes_json = JSONField(_("notes json"), null=True, blank=True)

    class Meta:
        db_table = "quant_tick_ml_signal"
        verbose_name = _("ml signal")
        verbose_name_plural = _("ml signals")
        ordering = ["-timestamp"]
        indexes = [
            models.Index(fields=["candle", "timestamp"]),
        ]
