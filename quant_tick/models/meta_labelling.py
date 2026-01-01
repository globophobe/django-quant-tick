from django.conf import settings
from django.db import models

from quant_tick.utils import gettext_lazy as _

from .base import JSONField


def upload_meta_artifact_to(instance: "MLArtifact", filename: str) -> str:
    """Upload meta artifact to."""
    prefix = "test-meta" if settings.TEST else "meta"
    code_name = instance.strategy.code_name
    return f"{prefix}/artifacts/{code_name}/{filename}"


class MLArtifact(models.Model):
    """ML artifact."""

    strategy = models.ForeignKey(
        "quant_tick.Strategy",
        on_delete=models.CASCADE,
        related_name="artifacts",
        verbose_name=_("strategy"),
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
    next_retrain_at = models.DateTimeField(
        _("next retrain at"),
        null=True,
        blank=True,
        help_text=_("When the strategy should be retrained again."),
    )

    class Meta:
        db_table = "quant_tick_meta_artifact"
        verbose_name = _("meta artifact")
        verbose_name_plural = _("meta artifacts")
