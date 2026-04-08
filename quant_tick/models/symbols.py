from datetime import UTC, datetime, time

from django.db import models
from django.utils.translation import gettext_lazy as _

from quant_tick.constants import Exchange, FileData

from .base import AbstractCodeName


class Symbol(AbstractCodeName):
    exchange = models.CharField(_("exchange"), choices=Exchange.choices, max_length=255)
    api_symbol = models.CharField(_("API symbol"), max_length=255)
    date_from = models.DateField(_("date from"), null=True)
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
        """Normalized symbol without exchange-specific separators."""
        symbol = self.api_symbol
        for char in ("-", "/", "_"):
            symbol = symbol.replace(char, "")
        if self.exchange == Exchange.BITFINEX:
            return symbol[1:]  # API symbol prepended with t
        return symbol

    @property
    def upload_path(self) -> str:
        """Storage path components for this symbol."""
        return [self.exchange, self.symbol, self.code_name]

    @property
    def trade_data_fields(self) -> tuple[FileData, ...]:
        fields = []
        if self.save_raw:
            fields.append(FileData.RAW)
        if self.save_aggregated:
            fields.append(FileData.AGGREGATED)
        if self.significant_trade_filter:
            fields.append(FileData.FILTERED)
        return tuple(fields)

    def clamp_timestamp_range(
        self, timestamp_from: datetime, timestamp_to: datetime
    ) -> tuple[datetime, datetime] | None:
        """Clamp a requested range to the symbol's available history."""
        if not self.date_from:
            return timestamp_from, timestamp_to
        date_from = datetime.combine(self.date_from, time.min).replace(tzinfo=UTC)
        if timestamp_to <= date_from:
            return None
        return max(timestamp_from, date_from), timestamp_to

    def __str__(self) -> str:
        exchange = self.get_exchange_display()
        return f"{exchange} {self.symbol} {self.code_name}"

    class Meta:
        db_table = "quant_tick_symbol"
        ordering = ("exchange", "api_symbol")
        verbose_name = _("symbol")
        verbose_name_plural = _("symbols")
