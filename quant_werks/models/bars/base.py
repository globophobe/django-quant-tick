import os

from django.contrib.contenttypes.models import ContentType
from django.db import models

from quant_werks.constants import Frequency
from quant_werks.utils import gettext_lazy as _

from ..aggregated_trades import AggregatedTradeData
from ..base import AbstractCodeName, AbstractDataStorage


def upload_to(instance: "BarData", filename: str, is_cache: bool = False) -> str:
    """Upload to."""
    date = instance.timestamp.date().isoformat()
    fname = instance.timestamp.time().strftime("%H%M")
    _, ext = os.path.splitext(filename)
    prefix = f"bars/{instance.code_name}/{date}"
    if is_cache:
        return f"{prefix}/cache/{fname}{ext}"
    else:
        return f"{prefix}/{fname}{ext}"


def upload_data_to(instance: "BarData", filename: str) -> str:
    """Upload data to."""
    return upload_to(instance, filename)


def upload_cache_to(instance: "BarData", filename: str) -> str:
    """Upload cache to."""
    return upload_to(instance, filename, is_cache=True)


class Bar(AbstractCodeName):
    symbols = models.ManyToManyField(
        "quant_werks.Symbol",
        db_table="quant_werks_bar_symbol",
        verbose_name=_("symbols"),
    )
    content_type = models.ForeignKey(ContentType, on_delete=models.SET_NULL, null=True)
    date_from = models.DateField(_("date from"), null=True)
    date_to = models.DateField(_("date to"), null=True)
    json_data = models.JSONField(_("json data"), null=True)
    is_active = models.BooleanField(_("active"), default=True)

    @classmethod
    def on_aggregated(self, obj: AggregatedTradeData) -> None:
        """On aggregated."""
        pass

    class Meta:
        db_table = "quant_werks_bar"
        verbose_name = _("bar")
        verbose_name_plural = _("bars")


class BarData(AbstractDataStorage):
    bar = models.ForeignKey(
        "quant_werks.Bar", related_name="data", on_delete=models.CASCADE
    )
    timestamp = models.DateTimeField(_("timestamp"), db_index=True)
    frequency = models.PositiveIntegerField(_("frequency"), choices=Frequency.choices)
    json_data = models.JSONField(_("json data"), null=True)
    file_data = models.FileField(_("file data"), blank=True, upload_to=upload_data_to)
    json_cache = models.JSONField(_("json cache"), null=True)
    file_cache = models.FileField(
        _("file cache"), blank=True, upload_to=upload_cache_to
    )

    class Meta:
        db_table = "quant_werks_bar_data"
        ordering = ("timestamp",)
        verbose_name = verbose_name_plural = _("bar data")
