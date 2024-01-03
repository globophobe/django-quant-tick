from quant_tick.utils import gettext_lazy as _

from .adaptive_candles import AdaptiveCandle


class ImbalanceCandle(AdaptiveCandle):
    """Imbalance candle.

    For example, 1 candle when:
    * Ticks exceed 1 standard deviation of the 7 day average tick imbalance.
    """

    class Meta:
        proxy = True
        verbose_name = _("imbalance candle")
        verbose_name_plural = _("imbalance candles")
