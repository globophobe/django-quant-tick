from rest_framework.permissions import AllowAny

from quant_tick.views import AggregateCandleView, AggregateTradeDataView, InferenceView


class GCPAggregateTradeDataView(AggregateTradeDataView):
    permission_classes = (AllowAny,)


class GCPAggregateCandleView(AggregateCandleView):
    permission_classes = (AllowAny,)


class GCPInferenceView(InferenceView):
    permission_classes = (AllowAny,)
