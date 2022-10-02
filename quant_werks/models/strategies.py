import os
from typing import Iterable

from django.contrib.contenttypes.fields import GenericForeignKey
from django.contrib.contenttypes.models import ContentType
from django.db import models

from quant_werks.utils import gettext_lazy as _

from .bars import Bar
from .base import AbstractCodeName


def upload_to(instance: "StrategyData", filename: str) -> str:
    """Upload to."""
    _, ext = os.path.splitext(filename)
    return f"strategies/{instance.strategy.code_name}/{instance.code_name}{ext}"


class StrategyState(models.IntegerChoices):
    PAPER_TRADE = 1, _("paper trade")
    LIVE_TRADE = 2, _("live trade")
    LIVE_TRADE_LIMIT_OUT = 3, _("live trade limit out")
    LIVE_TRADE_MARKET_OUT = 4, _("live trade market out")


class Strategy(AbstractCodeName):
    bars = models.ManyToManyField(
        "quant_werks.Bar", db_table="quant_werks_strategy_bar", verbose_name=_("bars")
    )
    state = models.CharField(_("state"), choices=StrategyState.choices, max_length=255)

    @classmethod
    def on_bars(self, bars: Iterable[Bar]) -> None:
        """On bars."""
        pass

    class Meta:
        db_table = "quant_werks_strategy"
        verbose_name = _("strategies")
        verbose_name_plural = _("strategies")


class StrategyData(AbstractCodeName):
    strategy = models.ForeignKey(
        "quant_werks.Strategy", related_name="data", on_delete=models.CASCADE
    )
    file_data = models.FileField(_("file data"), blank=True, upload_to=upload_to)

    class Meta:
        db_table = "quant_werks_strategy_data"
        verbose_name = verbose_name_plural = _("strategy data")


class StrategyLog(models.Model):
    strategy = models.ForeignKey(
        "quant_werks.Strategy", related_name="logs", on_delete=models.CASCADE
    )
    content_type = models.ForeignKey(ContentType, on_delete=models.CASCADE)
    object_id = models.PositiveIntegerField()
    content_object = GenericForeignKey("content_type", "object_id")
    timestamp = models.DateTimeField(_("timestamp"), db_index=True)
    json_data = models.JSONField(_("json data"), null=True)

    def __str__(self):
        return self.code_name

    class Meta:
        db_table = "quant_werks_strategy_log"
        verbose_name = verbose_name_plural = _("log")
