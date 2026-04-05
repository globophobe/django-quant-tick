from django.db import models
from django.utils.translation import gettext_lazy as _

from quant_tick.constants import Exchange

from .base import AbstractCodeName


class Symbol(AbstractCodeName):
    """Symbol."""

    exchange = models.CharField(_("exchange"), choices=Exchange.choices, max_length=255)
    api_symbol = models.CharField(_("API symbol"), max_length=255)
    save_raw = models.BooleanField(
        _("save raw"),
        help_text=_("Save raw data?"),
        default=False,
    )
    save_aggregated = models.BooleanField(
        _("aggregate trades"),
        help_text=_("Should trades be aggregated?"),
        default=False,
    )
    save_filtered = models.BooleanField(
        _("save filtered"),
        help_text=_("Save filtered data?"),
        default=True,
    )
    significant_trade_filter = models.PositiveIntegerField(
        _("significant trade filter"),
        help_text=_(
            "All trades less than the value of the significant trade filter will be "
            "aggregated, according to the the algorithm described in the documentation."
        ),
        default=0,
    )
    is_active = models.BooleanField(_("active"), default=True)

    @property
    def symbol(self) -> str:
        """Symbol.

        Example: BTCUSD
        """
        symbol = self.api_symbol
        for char in ("-", "/", "_"):
            symbol = symbol.replace(char, "")
        if self.exchange == Exchange.BITFINEX:
            return symbol[1:]  # API symbol prepended with t
        # elif self.exchange == Exchange.UPBIT:
        #     return symbol[3:] + symbol[:3]  # Reversed
        return symbol

    @property
    def upload_path(self) -> str:
        """Upload path.

        Example: coinbase / BTCUSD / blaring-crocodile
        """
        return [self.exchange, self.symbol, self.code_name]

    def __str__(self) -> str:
        """str."""
        exchange = self.get_exchange_display()
        return f"{exchange} {self.symbol} {self.code_name}"

    class Meta:
        db_table = "quant_tick_symbol"
        ordering = ("exchange", "api_symbol")
        verbose_name = _("symbol")
        verbose_name_plural = _("symbols")
