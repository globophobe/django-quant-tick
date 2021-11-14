from django.core.exceptions import ValidationError
from django.db import models

from cryptofeed_werks.utils import gettext_lazy as _

from .base import big_decimal


def one_or_minus_one(value):
    if value not in (None, 1, -1):
        raise ValidationError(
            _("%(value)s is neither 1 nor -1"), params={"value": value}
        )


class AggregatedTrade(models.Model):
    candle = models.ForeignKey(
        "cryptofeed_werks.Candle",
        related_name="aggregated_trades",
        on_delete=models.CASCADE,
    )
    timestamp = models.DateTimeField(_("timestamp"))
    nanoseconds = models.PositiveIntegerField(_("nanoseconds"))
    price = big_decimal("price")
    volume = big_decimal("volume", null=True)
    notional = big_decimal("notional", null=True)
    tick_rule = models.IntegerField(
        _("notional"), null=True, validators=[one_or_minus_one]
    )
    ticks = models.PositiveIntegerField(_("ticks"), null=True)
    high = big_decimal("high")
    low = big_decimal("low")
    total_buy_volume = big_decimal("total buy volume")
    total_volume = big_decimal("total volume")
    total_buy_notional = big_decimal("total buy notional")
    total_notional = big_decimal("total notional")
    total_buy_ticks = models.PositiveIntegerField(_("total buy ticks"))
    total_ticks = models.PositiveIntegerField(_("total ticks"))

    @property
    def columns(self) -> list:
        SINGLE_SYMBOL = [
            "timestamp",
            "nanoseconds",
            "price",
            "volume",
            "notional",
            "tickRule",
            "ticks",
            "high",
            "low",
            "totalBuyVolume",
            "totalVolume",
            "totalBuyNotional",
            "totalNotional",
            "totalBuyTicks",
            "totalTicks",
            "index",
        ]
        if self.symbol.is_futures:
            # Multiple symbol
            return ["symbol"] + SINGLE_SYMBOL[:-1] + ["expiry", "index"]
        else:
            return SINGLE_SYMBOL

    class Meta:
        db_table = "cryptofeed_werks_aggregated_trade"
        # It would be nice to order by and index nanoseconds,
        # but it uses a lot of disk space.
        ordering = ("timestamp",)
        verbose_name = _("aggregated trade")
        verbose_name_plural = _("aggregated trades")
