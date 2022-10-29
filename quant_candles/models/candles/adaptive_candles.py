from quant_candles.utils import gettext_lazy as _

from ..trades import TradeData
from .base import Candle


class AdaptiveCandle(Candle):
    @classmethod
    def on_trades(self, obj: TradeData) -> None:
        pass

    class Meta:
        proxy = True
        verbose_name = _("adaptive candle")
        verbose_name_plural = _("adaptive candles")
