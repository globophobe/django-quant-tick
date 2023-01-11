from quant_candles.utils import gettext_lazy as _

from ..candles import Candle


class ImbalanceCandle(Candle):
    """Imbalance candle.

    For example, 1 candle when:
    * Ticks exceed 1 standard deviation of the 7 day average tick imbalance.
    """

    class Meta:
        proxy = True
        verbose_name = _("imbalance candle")
        verbose_name_plural = _("imbalance candles")
