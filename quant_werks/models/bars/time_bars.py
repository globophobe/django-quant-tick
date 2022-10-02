from quant_werks.utils import gettext_lazy as _

from ..aggregated_trades import AggregatedTradeData
from .base import Bar


class TimeBar(Bar):
    @classmethod
    def on_aggregated(self, bar: Bar, aggregated: AggregatedTradeData) -> None:
        pass

    class Meta:
        proxy = True
        verbose_name = _("time bar")
        verbose_name_plural = _("time bars")
