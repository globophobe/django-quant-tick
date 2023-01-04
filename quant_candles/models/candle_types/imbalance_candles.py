from quant_candles.utils import gettext_lazy as _

from ..candles import Candle


class ImbalanceCandle(Candle):
    class Meta:
        proxy = True
        verbose_name = _("imbalance candle")
        verbose_name_plural = _("imbalance candles")
