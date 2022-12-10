import os

from django.contrib.contenttypes.models import ContentType
from django.db import models

from quant_candles.constants import Frequency
from quant_candles.utils import gettext_lazy as _

from ..base import AbstractCodeName, AbstractDataStorage, JSONField
from ..trades import TradeData

DAILY = "D"
WEEKLY = "W"
MONTHLY = "M"
QUARTERLY = "Q"

ERA_LENGTHS = (DAILY, WEEKLY, MONTHLY, QUARTERLY)


def get_range(time_delta):
    total_seconds = time_delta.total_seconds()
    one_minute = 60.0
    one_hour = 3600.0
    if total_seconds < one_minute:
        unit = "seconds"
        step = total_seconds
    elif total_seconds >= one_minute and total_seconds <= one_hour:
        unit = "minutes"
        step = total_seconds / one_minute
    elif total_seconds > one_hour:
        unit = "hours"
        step = total_seconds / one_hour
    else:
        raise NotImplementedError
    step = int(step)
    assert total_seconds <= one_hour, f"{step} {unit} not supported"
    assert 60 % step == 0, f"{step} not divisible by 60"
    return unit, step


def upload_to(instance: "CandleData", filename: str, is_cache: bool = False) -> str:
    """Upload to."""
    date = instance.timestamp.date().isoformat()
    fname = instance.timestamp.time().strftime("%H%M")
    _, ext = os.path.splitext(filename)
    prefix = f"candles/{instance.code_name}/{date}"
    if is_cache:
        return f"{prefix}/cache/{fname}{ext}"
    else:
        return f"{prefix}/{fname}{ext}"


def upload_data_to(instance: "CandleData", filename: str) -> str:
    """Upload data to."""
    return upload_to(instance, filename)


def upload_cache_to(instance: "CandleData", filename: str) -> str:
    """Upload cache to."""
    return upload_to(instance, filename, is_cache=True)


class Candle(AbstractCodeName):
    symbols = models.ManyToManyField(
        "quant_candles.Symbol",
        db_table="quant_candles_candle_symbol",
        verbose_name=_("symbols"),
    )
    content_type = models.ForeignKey(ContentType, on_delete=models.SET_NULL, null=True)
    date_from = models.DateField(_("date from"), null=True)
    date_to = models.DateField(_("date to"), null=True)
    json_data = JSONField(_("json data"), null=True)
    is_active = models.BooleanField(_("active"), default=True)

    @classmethod
    def on_trades(self, obj: TradeData) -> None:
        """On trades."""
        pass

    class Meta:
        db_table = "quant_candles_candle"
        verbose_name = _("candle")
        verbose_name_plural = _("candles")


class CandleData(AbstractDataStorage):
    candle = models.ForeignKey(
        "quant_candles.Candle", related_name="data", on_delete=models.CASCADE
    )
    timestamp = models.DateTimeField(_("timestamp"), db_index=True)
    frequency = models.PositiveIntegerField(_("frequency"), choices=Frequency.choices)
    json_data = JSONField(_("json data"), null=True)
    file_data = models.FileField(_("file data"), blank=True, upload_to=upload_data_to)
    json_cache = JSONField(_("json cache"), null=True)
    file_cache = models.FileField(
        _("file cache"), blank=True, upload_to=upload_cache_to
    )

    class Meta:
        db_table = "quant_candles_candle_data"
        ordering = ("timestamp",)
        verbose_name = verbose_name_plural = _("candle data")
