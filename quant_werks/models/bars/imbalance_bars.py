from quant_werks.utils import gettext_lazy as _

from ..aggregated_trades import AggregatedTradeData
from .base import Bar


class ImbalanceBar(Bar):
    @classmethod
    def on_aggregated(self, obj: AggregatedTradeData) -> None:
        pass

    class Meta:
        proxy = True
        verbose_name = _("imbalance bar")
        verbose_name_plural = _("imbalance bars")
