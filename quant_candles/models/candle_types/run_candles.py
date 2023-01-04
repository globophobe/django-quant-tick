from quant_candles.utils import gettext_lazy as _

from ..candles import Candle


class RunCandle(Candle):
    class Meta:
        proxy = True
        verbose_name = _("run candle")
        verbose_name_plural = _("run candles")
