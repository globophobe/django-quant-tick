from typing import Iterable

from quant_candles.utils import gettext_lazy as _

from ..trades import TradeData
from .constant_candles import ConstantCandle


class AdaptiveCandle(ConstantCandle):
    @classmethod
    def on_trades(cls, objs: Iterable[TradeData]) -> None:
        """On trades."""
        pass

    class Meta:
        proxy = True
        verbose_name = _("adaptive candle")
        verbose_name_plural = _("adaptive candles")
