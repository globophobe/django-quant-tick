from django.db import models

from cryptofeed_werks.constants import Exchange, SymbolType
from cryptofeed_werks.utils import gettext_lazy as _

from .base import NameMixin


class GlobalSymbol(NameMixin):
    class Meta:
        db_table = "cryptofeed_werks_global_symbol"
        ordering = ("name",)
        verbose_name = _("global symbol")
        verbose_name_plural = _("global symbols")


class Symbol(models.Model):
    global_symbol = models.ForeignKey(
        "cryptofeed_werks.GlobalSymbol",
        related_name="symbols",
        on_delete=models.CASCADE,
        verbose_name=_("global symbol"),
    )
    exchange = models.CharField(_("exchange"), choices=Exchange.choices, max_length=255)
    symbol_type = models.CharField(
        _("type"), choices=SymbolType.choices, max_length=255
    )
    name = models.CharField(
        _("name"),
        help_text='API symbol. For futures this is the "root symbol", i.e. XBT',
        max_length=255,
    )
    min_volume = models.PositiveIntegerField(
        _("min volume"),
        help_text=_("For aggregated trades, the minimum volume required, e.g. 1000"),
        null=True,
        blank=True,
    )

    @property
    def is_futures(self) -> bool:
        return self.symbol_type == SymbolType.FUTURE

    @property
    def symbol_display(self) -> str:
        symbol = self.name
        for char in ("-", "/", "_"):
            symbol = symbol.replace(char, "")
        if self.exchange == Exchange.BITFINEX:
            return symbol[1:]  # API symbol prepended with t
        elif self.exchange == Exchange.UPBIT:
            return symbol[3:] + symbol[:3]  # Reversed
        return symbol

    def __str__(self) -> str:
        exchange = self.get_exchange_display()
        symbol = self.symbol_display
        prefix = f"{exchange} {symbol}"
        return f"{prefix} futures" if self.is_futures else prefix

    class Meta:
        db_table = "cryptofeed_werks_symbol"
        ordering = ("exchange", "name", "symbol_type")
        verbose_name = _("symbol")
        verbose_name_plural = _("symbols")


class Future(models.Model):
    root_symbol = models.ForeignKey(
        "cryptofeed_werks.Symbol",
        related_name="futures",
        on_delete=models.CASCADE,
        verbose_name=_("root symbol"),
    )
    name = models.CharField(_("name"), max_length=255)
    expiry = models.DateTimeField(_("expiry"), null=True)

    def __str__(self) -> str:
        exchange = self.root_symbol.get_exchange_display()
        symbol = self.root_symbol.symbol_display
        prefix = f"{exchange} {symbol}"
        suffix = self.name
        return f"{prefix} {suffix}"

    class Meta:
        db_table = "cryptofeed_werks_future"
        ordering = ("name",)
        verbose_name = _("future")
        verbose_name_plural = _("futures")
