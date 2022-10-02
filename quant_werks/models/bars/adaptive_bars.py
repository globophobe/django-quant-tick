from quant_werks.utils import gettext_lazy as _

from ..aggregated_trades import AggregatedTradeData
from .base import Bar


class AdaptiveBar(Bar):
    @classmethod
    def on_aggregated(self, obj: AggregatedTradeData) -> None:
        pass

    class Meta:
        proxy = True
        verbose_name = _("adaptive bar")
        verbose_name_plural = _("adaptive bars")
