import os

from django.db import models

from cryptofeed_werks.constants import CandleType, Frequency
from cryptofeed_werks.utils import gettext_lazy as _

from .base import BaseDataStorage


def upload_data_to(instance: "CandleData", filename: str) -> str:
    """Upload to."""
    date = instance.timestamp.date().isoformat()
    fname = instance.timestamp.time().strftime("%H%M")
    _, ext = os.path.splitext(filename)
    return f"candles/{instance.code_name}/{date}/{fname}{ext}"


def upload_cache_to(instance: "CandleData", filename: str) -> str:
    """Upload to."""
    date = instance.timestamp.date().isoformat()
    fname = instance.timestamp.time().strftime("%H%M")
    _, ext = os.path.splitext(filename)
    return f"candles/{instance.code_name}/{date}/cache/{fname}{ext}"


class Candle(models.Model):
    code_name = models.SlugField(_("code name"), unique=True, max_length=255)
    candle_type = models.CharField(
        _("candle type"), choices=CandleType.choices, max_length=255
    )
    threshold = models.CharField(
        _("threshold"),
        help_text=_(
            'For time-based candles, "1d", "4h", "1m", etc. '
            "For dollar, volume, or tick candles, a numerical value. "
            "Must be blank for candles with an adaptive threshold."
        ),
        max_length=255,
        blank=True,
    )
    expected_number_of_candles = models.PositiveIntegerField(
        _("expected number of candles"),
        help_text=_(
            "Not applicable to time-based candles. "
            "The expected number of candles per day for adaptive thresholds."
        ),
        null=True,
        blank=True,
    )
    moving_average_number_of_days = models.PositiveIntegerField(
        _("moving average number of days"),
        help_text=_(
            "The number of days for the moving average for adaptive thresholds. "
        ),
        null=True,
        blank=True,
    )
    is_ema = models.BooleanField(
        "EMA", help_text=_("Is the moving average an EMA?"), default=False
    )
    cache_reset_frequency = models.PositiveIntegerField(
        _("cache reset frequency"),
        help_text=_(
            "Not applicable to time-based candles. "
            "If not blank, the cache will be reset to 0 at the start of each period. "
        ),
        choices=[
            c for c in Frequency.choices if c[0] in (Frequency.DAY, Frequency.WEEK)
        ],
        null=True,
        blank=True,
    )

    def __str__(self):
        return self.code_name

    class Meta:
        db_table = "cryptofeed_werks_candle"
        verbose_name = _("candle")
        verbose_name_plural = _("candles")


class CandleSymbol(models.Model):
    candle = models.ForeignKey(
        "cryptofeed_werks.Candle", on_delete=models.CASCADE, verbose_name=_("candle")
    )
    symbol = models.ForeignKey(
        "cryptofeed_werks.Symbol", on_delete=models.CASCADE, verbose_name=_("symbol")
    )
    date_from = models.DateField(_("date from"), null=True)
    date_to = models.DateField(_("date to"), null=True)

    def __str__(self):
        return str(self.symbol)

    class Meta:
        db_table = "cryptofeed_werks_candle_symbol"
        unique_together = ("candle", "symbol")
        verbose_name = _("symbol")
        verbose_name_plural = _("symbols")


class CandleData(BaseDataStorage):
    candle = models.ForeignKey(
        "cryptofeed_werks.Candle", related_name="data", on_delete=models.CASCADE
    )
    symbol = models.ForeignKey(
        "cryptofeed_werks.Symbol", related_name="data", on_delete=models.CASCADE
    )
    timestamp = models.DateTimeField(_("timestamp"), db_index=True)
    frequency = models.PositiveIntegerField(_("frequency"), choices=Frequency.choices)
    data = models.FileField(_("data"), blank=True, upload_to=upload_data_to)
    cache = models.FileField(_("data"), blank=True, upload_to=upload_cache_to)

    class Meta:
        db_table = "cryptofeed_werks_candle_data"
        ordering = ("timestamp",)
        verbose_name = verbose_name_plural = _("candle data")
