from django.db import models

from cryptofeed_werks.constants import SamplingType
from cryptofeed_werks.utils import gettext_lazy as _


class Series(models.Model):
    symbol = models.ForeignKey(
        "cryptofeed_werks.Symbol",
        related_name="series",
        on_delete=models.CASCADE,
    )
    sampling_type = models.CharField(
        _("type"), choices=SamplingType.choices, max_length=256
    )
    frequency = models.PositiveIntegerField(_("frequency"))

    class Meta:
        db_table = "cryptofeed_werks_series"
        verbose_name = verbose_name_plural = _("series")


class SeriesHistory(models.Model):
    series = models.ForeignKey(
        "cryptofeed_werks.Series",
        related_name="history",
        on_delete=models.CASCADE,
    )
    date = models.DateField(_("date"), db_index=True)
    data = models.JSONField(default=dict)
    ok = models.BooleanField(_("ok"), null=True, default=False, db_index=True)

    class Meta:
        db_table = "cryptofeed_werks_series_history"
        ordering = ("date",)
        verbose_name = verbose_name_plural = _("series history")
