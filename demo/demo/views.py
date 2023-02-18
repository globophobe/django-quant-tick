from quant_candles.views import AggregateCandleView, AggregateTradeDataView

from .permissions import GCPServicePermission


class GCPAggregateTradeDataView(AggregateTradeDataView):
    permission_classes = (GCPServicePermission,)


class GCPAggregateCandleView(AggregateCandleView):
    permission_classes = (GCPServicePermission,)
