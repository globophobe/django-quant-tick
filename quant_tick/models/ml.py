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
    last_candle_data = models.ForeignKey(
        "quant_tick.CandleData",
        on_delete=models.SET_NULL,
        verbose_name=_("last candle data"),
        related_name="+",
        null=True,
        blank=True,
        help_text=_("Last CandleData processed for inference"),
    )
    json_data = JSONField(
        _("json data"),
        help_text=_(
            "symbols, time_window, features, labeling, cv, model_hparams, thresholds"
        ),
        default=dict,
    )
    is_active = models.BooleanField(_("active"), default=True)
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
    artifact_type = models.CharField(
        _("artifact type"),
        max_length=50,
        default="primary_model",
        help_text=_("Type of artifact: primary_model or meta_model"),
    )
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
    probability = models.FloatField(_("probability"))
    side = models.SmallIntegerField(_("side"))
    meta_label = models.SmallIntegerField(_("meta label"), null=True, blank=True)
    meta_prob = models.FloatField(_("meta probability"), null=True, blank=True)
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


class TrendScan(models.Model):
    """Trend scan result."""

    ml_config = models.ForeignKey(
        "quant_tick.MLConfig",
        on_delete=models.CASCADE,
        verbose_name=_("ml config"),
        related_name="trend_scans",
    )
    ml_run = models.ForeignKey(
        "quant_tick.MLRun",
        on_delete=models.CASCADE,
        verbose_name=_("ml run"),
        related_name="trend_scans",
        null=True,
        blank=True,
        help_text=_("Associated backtest run if from backtest"),
    )
    timestamp = models.DateTimeField(
        _("timestamp"),
        db_index=True,
        help_text=_("Timestamp when scan was performed"),
    )
    window_start_idx = models.IntegerField(_("window start index"))
    window_end_idx = models.IntegerField(_("window end index"))
    window_size = models.IntegerField(_("window size"))
    timestamp_start = models.DateTimeField(_("timestamp start"))
    timestamp_end = models.DateTimeField(_("timestamp end"))
    score = models.FloatField(
        _("score"), help_text=_("Trend statistic (Sharpe or t-stat)")
    )
    mean_return = models.FloatField(_("mean return"))
    std_return = models.FloatField(_("std return"))
    p_value = models.FloatField(_("p value"))
    n_events = models.IntegerField(
        _("n events"), help_text=_("Number of events in window")
    )
    method = models.CharField(
        _("method"),
        max_length=20,
        default="sharpe",
        help_text=_("Statistic method: sharpe or t_stat"),
    )
    returns_type = models.CharField(
        _("returns type"),
        max_length=50,
        default="predictions",
        help_text=_("Type of returns: predictions, realized_pnl, meta_filtered"),
    )
    created_at = models.DateTimeField(_("created at"), auto_now_add=True)

    class Meta:
        db_table = "quant_tick_trend_scan"
        verbose_name = _("trend scan")
        verbose_name_plural = _("trend scans")
        ordering = ["-timestamp", "-score"]
        indexes = [
            models.Index(fields=["ml_config", "timestamp"]),
            models.Index(fields=["ml_run", "timestamp"]),
        ]


class TrendAlert(models.Model):
    """Trend alert when structural break is detected."""

    ml_config = models.ForeignKey(
        "quant_tick.MLConfig",
        on_delete=models.CASCADE,
        verbose_name=_("ml config"),
        related_name="trend_alerts",
    )
    timestamp = models.DateTimeField(
        _("timestamp"),
        db_index=True,
        help_text=_("When alert was triggered"),
    )
    current_top_score = models.FloatField(_("current top score"))
    previous_top_score = models.FloatField(
        _("previous top score"), null=True, blank=True
    )
    deterioration = models.FloatField(_("deterioration"), null=True, blank=True)
    threshold = models.FloatField(_("threshold"))
    action = models.CharField(
        _("action"),
        max_length=50,
        default="notification",
        help_text=_("Action taken: pause_trading, notification"),
    )
    window_metadata = JSONField(
        _("window metadata"),
        null=True,
        blank=True,
        help_text=_("Top window details at alert time"),
    )
    status = models.CharField(
        _("status"),
        max_length=50,
        default="active",
        help_text=_("active, acknowledged, resolved"),
    )
    resolved_at = models.DateTimeField(_("resolved at"), null=True, blank=True)
    message = models.TextField(_("message"), blank=True)
    created_at = models.DateTimeField(_("created at"), auto_now_add=True)

    class Meta:
        db_table = "quant_tick_trend_alert"
        verbose_name = _("trend alert")
        verbose_name_plural = _("trend alerts")
        ordering = ["-timestamp"]
        indexes = [
            models.Index(fields=["ml_config", "status", "timestamp"]),
        ]
