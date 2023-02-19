from rest_framework.permissions import AllowAny

from quant_candles.views import AggregateCandleView, AggregateTradeDataView


class GCPAggregateTradeDataView(AggregateTradeDataView):
    permission_classes = (AllowAny,)


class GCPAggregateCandleView(AggregateCandleView):
    permission_classes = (AllowAny,)
