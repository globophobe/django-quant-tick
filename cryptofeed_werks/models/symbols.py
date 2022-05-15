from django.db import models

from cryptofeed_werks.constants import Exchange
from cryptofeed_werks.utils import gettext_lazy as _


class GlobalSymbol(models.Model):
    name = models.CharField(_("name"), max_length=255)

    def __str__(self) -> str:
        return self.name

    class Meta:
        db_table = "cryptofeed_werks_global_symbol"
        verbose_name = _("global symbol")
        verbose_name_plural = _("global symbols")


class Symbol(models.Model):
    global_symbol = models.ForeignKey(
        "cryptofeed_werks.GlobalSymbol",
        related_name="symbols",
        on_delete=models.CASCADE,
    )
    exchange = models.CharField(_("exchange"), choices=Exchange.choices, max_length=255)
    api_symbol = models.CharField(_("API symbol"), max_length=255)
    min_volume = models.PositiveIntegerField(
        _("min volume"),
        help_text=_("The minimum volume for significant trades, e.g. 1000"),
        default=1000,
    )

    @property
    def symbol(self) -> str:
        symbol = self.api_symbol
        for char in ("-", "/", "_"):
            symbol = symbol.replace(char, "")
        if self.exchange == Exchange.BITFINEX:
            return symbol[1:]  # API symbol prepended with t
        # elif self.exchange == Exchange.UPBIT:
        #     return symbol[3:] + symbol[:3]  # Reversed
        return symbol

    def __str__(self) -> str:
        exchange = self.get_exchange_display()
        return f"{exchange} {self.symbol}"

    class Meta:
        db_table = "cryptofeed_werks_symbol"
        ordering = ("exchange", "api_symbol")
        verbose_name = _("symbol")
        verbose_name_plural = _("symbols")
