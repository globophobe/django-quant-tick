from typing import Iterable

from quant_candles.utils import gettext_lazy as _

from ..candles import Candle
from ..trades import TradeData


class RunCandle(Candle):
    @classmethod
    def on_trades(cls, objs: Iterable[TradeData]) -> None:
        """On trades."""
        pass

    class Meta:
        proxy = True
        verbose_name = _("run candle")
        verbose_name_plural = _("run candles")
