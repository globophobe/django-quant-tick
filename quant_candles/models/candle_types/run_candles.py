from quant_candles.utils import gettext_lazy as _

from .imbalance_candles import ImbalanceCandle


class RunCandle(ImbalanceCandle):
    """Run candle.

    For example, 1 candle when:
    * Ticks exceed 1 standard deviation of the 7 day moving average of tick runs.
    """

    class Meta:
        proxy = True
        verbose_name = _("run candle")
        verbose_name_plural = _("run candles")
