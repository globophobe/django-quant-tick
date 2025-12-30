from django.db import models

from quant_tick.utils import gettext_lazy as _


class FundingRate(models.Model):
    """Funding rate for perps."""

    symbol = models.ForeignKey(
        "quant_tick.Symbol",
        on_delete=models.CASCADE,
        related_name="funding_rates",
        verbose_name=_("symbol"),
    )
    timestamp = models.DateTimeField(_("timestamp"), db_index=True)
    rate = models.DecimalField(
        _("rate"),
        max_digits=20,
        decimal_places=10,
    )

    def __str__(self) -> str:
        """String."""
        return f"{self.symbol} {self.timestamp} {self.rate}"

    class Meta:
        db_table = "quant_tick_funding_rate"
        verbose_name = _("funding rate")
        verbose_name_plural = _("funding rates")
        unique_together = ["symbol", "timestamp"]
        ordering = ["-timestamp"]
        indexes = [
            models.Index(fields=["symbol", "timestamp"]),
        ]
