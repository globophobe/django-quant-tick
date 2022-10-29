from django.db import models

from quant_candles.constants import Exchange, SymbolType
from quant_candles.utils import gettext_lazy as _

from .base import BigDecimalField


class GlobalSymbol(models.Model):
    name = models.CharField(_("name"), unique=True, max_length=255)
    notional = BigDecimalField(_("notional"), null=True, blank=True)

    def __str__(self) -> str:
        return self.name

    class Meta:
        db_table = "quant_candles_global_symbol"
        verbose_name = _("global symbol")
        verbose_name_plural = _("global symbols")


class Symbol(models.Model):
    global_symbol = models.ForeignKey(
        "quant_candles.GlobalSymbol",
        related_name="symbols",
        on_delete=models.CASCADE,
    )
    exchange = models.CharField(_("exchange"), choices=Exchange.choices, max_length=255)
    symbol_type = models.CharField(
        _("type"), choices=SymbolType.choices, max_length=255
    )
    api_symbol = models.CharField(_("API symbol"), max_length=255)
    should_aggregate_trades = models.BooleanField(
        _("aggregate trades"),
        help_text=_("Should trades be aggregated?"),
        default=False,
    )
    significant_trade_filter = models.PositiveIntegerField(
        _("significant trade filter"),
        help_text=_(
            "If trades are aggregated, they can be further filtered according "
            "to the algorithm described in the documentation."
        ),
        default=0,
    )

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

        Example Coinbase BTCUSD aggregated 1000
        """
        parts = [self.get_exchange_display(), self.symbol]
        if self.should_aggregate_trades:
            parts += ["aggregated", str(self.significant_trade_filter)]
        else:
            parts.append("raw")
        return " ".join(parts)

    class Meta:
        db_table = "quant_candles_symbol"
        ordering = ("exchange", "api_symbol")
        unique_together = (
            (
                "exchange",
                "api_symbol",
                "should_aggregate_trades",
                "significant_trade_filter",
            ),
        )
        verbose_name = _("symbol")
        verbose_name_plural = _("symbols")
