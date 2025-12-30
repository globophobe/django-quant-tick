from django.conf import settings
from django.db import models

from quant_tick.utils import gettext_lazy as _

from .base import AbstractCodeName, JSONField


def upload_meta_artifact_to(instance: "MetaArtifact", filename: str) -> str:
    """Upload meta artifact to."""
    prefix = "test-meta" if settings.TEST else "meta"
    code_name = instance.meta_model.code_name
    return f"{prefix}/artifacts/{code_name}/{filename}"


class MetaModel(AbstractCodeName):
    """Meta model."""

    candle = models.ForeignKey(
        "quant_tick.Candle",
        on_delete=models.CASCADE,
        related_name="meta_models",
        null=False,
        blank=False,
        verbose_name=_("candle"),
    )
    symbol = models.ForeignKey(
        "quant_tick.Symbol",
        on_delete=models.CASCADE,
        related_name="meta_models",
        null=False,
        blank=False,
        verbose_name=_("symbol"),
        help_text=_("Target symbol for multi-exchange feature selection and trading."),
    )
    last_processed_timestamp = models.DateTimeField(
        _("last processed timestamp"),
        null=True,
        blank=True,
        help_text=_("Last processed timestamp for idempotency"),
    )
    json_data = JSONField(_("json data"), default=dict)
    is_active = models.BooleanField(_("active"), default=True)

    class Meta:
        db_table = "quant_tick_meta_model"
        verbose_name = _("meta model")
        verbose_name_plural = _("meta models")


class MetaSignal(models.Model):
    """Meta signal."""

    meta_model = models.ForeignKey(
        "quant_tick.MetaModel",
        on_delete=models.CASCADE,
        related_name="signals",
        verbose_name=_("meta model"),
    )
    candle_entry = models.ForeignKey(
        "quant_tick.CandleData",
        on_delete=models.CASCADE,
        related_name="meta_signal_entry",
        verbose_name=_("candle entry"),
        null=True,
        blank=True,
    )
    candle_exit = models.ForeignKey(
        "quant_tick.CandleData",
        on_delete=models.SET_NULL,
        related_name="meta_signal_exit",
        verbose_name=_("candle exit"),
        null=True,
        blank=True,
    )
    timestamp = models.DateTimeField(_("timestamp"), db_index=True)
    probability = models.FloatField(
        _("probability"), help_text=_("Model probability for taking the trade.")
    )
    decision = models.CharField(
        _("decision"),
        max_length=16,
        choices=[("take", "Take"), ("skip", "Skip"), ("unknown", "Unknown")],
        default="unknown",
    )
    json_data = JSONField(_("json data"), default=dict)

    class Meta:
        db_table = "quant_tick_meta_signal"
        verbose_name = _("meta signal")
        verbose_name_plural = _("meta signals")
        indexes = [
            models.Index(fields=["timestamp"]),
        ]


class MetaArtifact(models.Model):
    """Meta artifact."""

    meta_model = models.ForeignKey(
        "quant_tick.MetaModel",
        on_delete=models.CASCADE,
        related_name="artifacts",
        verbose_name=_("meta model"),
    )
    file_data = models.FileField(
        upload_to=upload_meta_artifact_to,
        verbose_name=_("bundle file"),
    )
    model_kind = models.CharField(
        _("model kind"),
        max_length=64,
        help_text=_("Identifier for bundle format, e.g., meta_logreg."),
    )
    feature_cols = JSONField(
        _("feature columns"),
        default=list,
        help_text=_("Feature columns expected by the model."),
    )
    created_at = models.DateTimeField(_("created at"), auto_now_add=True)
    json_data = JSONField(_("json data"), default=dict)
    is_active = models.BooleanField(_("active"), default=True)

    class Meta:
        db_table = "quant_tick_meta_artifact"
        verbose_name = _("meta artifact")
        verbose_name_plural = _("meta artifacts")
