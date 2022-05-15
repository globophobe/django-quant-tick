from django.db import models

from cryptofeed_werks.utils import gettext_lazy as _

from .base import big_decimal


class Candle(models.Model):
    series = models.ForeignKey(
        "cryptofeed_werks.Series",
        related_name="candles",
        on_delete=models.CASCADE,
    )
    timestamp = models.DateTimeField(_("timestamp"), db_index=True)
    open = big_decimal("open", null=True)
    high = big_decimal("high", null=True)
    low = big_decimal("low", null=True)
    close = big_decimal("close", null=True)
    buy_volume = big_decimal("buy volume", null=True)
    volume = big_decimal("volume", null=True)
    buy_notional = big_decimal("buy notional", null=True)
    notional = big_decimal("notional", null=True)
    buy_ticks = models.PositiveIntegerField(_("buy ticks"), null=True)
    ticks = models.PositiveIntegerField(_("ticks"), null=True)
    data = models.JSONField(default=dict)

    def __str__(self) -> str:
        exchange = self.series.symbol.get_exchange_display()
        symbol = self.series.symbol.symbol_display
        timestamp = self.timestamp.replace(tzinfo=None).isoformat()
        return f"{exchange} {symbol} {timestamp}"

    class Meta:
        db_table = "cryptofeed_werks_candle"
        ordering = ("timestamp",)
        verbose_name = _("candle")
        verbose_name_plural = _("candles")
