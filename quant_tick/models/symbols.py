from django.db import models

from quant_tick.constants import Exchange, SymbolType
from quant_tick.utils import gettext_lazy as _

from .base import BigDecimalField, AbstractCodeName


class GlobalSymbol(models.Model):
    """Global symbol."""

    name = models.CharField(_("name"), unique=True, max_length=255)

    def __str__(self) -> str:
        return self.name

    class Meta:
        db_table = "quant_candles_global_symbol"
        verbose_name = _("global symbol")
        verbose_name_plural = _("global symbols")


class Symbol(AbstractCodeName):
    """Symbol."""

    global_symbol = models.ForeignKey(
        "quant_tick.GlobalSymbol",
        related_name="symbols",
        on_delete=models.CASCADE,
    )
    exchange = models.CharField(_("exchange"), choices=Exchange.choices, max_length=255)
    symbol_type = models.CharField(
        _("type"), choices=SymbolType.choices, max_length=255
    )
    api_symbol = models.CharField(_("API symbol"), max_length=255)
    aggregate_trades = models.BooleanField(
        _("aggregate trades"),
        help_text=_("Should trades be aggregated?"),
        default=False,
    )
    significant_trade_filter = models.PositiveIntegerField(
        _("significant trade filter"),
        help_text=_(
            "All trades less than the value of the significant trade filter will be "
            "aggregated, according to the the algorithm described in the documentation."
        ),
        default=0,
    )
    currency_divisor = models.PositiveIntegerField(_("currency divisor"), default=1)
    recent_error_at = models.DateTimeField(
        _("recent API error"),
        help_text=_(
            "If there was a recent API error, for example due to exchange maintenance, "
            "then backoff."
        ),
        null=True,
    )
    is_active = models.BooleanField(_("active"), default=True)

    @property
    def symbol(self) -> str:
        """Symbol.

        Example, BTCUSD
        """
        symbol = self.api_symbol
        for char in ("-", "/", "_"):
            symbol = symbol.replace(char, "")
        if self.exchange == Exchange.BITFINEX:
            return symbol[1:]  # API symbol prepended with t
        # elif self.exchange == Exchange.UPBIT:
        #     return symbol[3:] + symbol[:3]  # Reversed
        return symbol

    def __str__(self) -> str:
        """str.

        Example Coinbase BTC-USD blaring-crocodile
        """
        exchange = self.get_exchange_display()
        return f"{exchange} {self.symbol} {self.code_name}"

    class Meta:
        db_table = "quant_candles_symbol"
        ordering = ("exchange", "api_symbol")
        verbose_name = _("symbol")
        verbose_name_plural = _("symbols")
