from quant_candles.utils import gettext_lazy as _

from ..trades import TradeData
from .base import Candle


class RunCandle(Candle):
    @classmethod
    def on_aggregated(self, obj: TradeData) -> None:
        pass

    class Meta:
        proxy = True
        verbose_name = _("run candle")
        verbose_name_plural = _("run candles")
