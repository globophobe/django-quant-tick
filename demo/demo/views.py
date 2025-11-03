from rest_framework.permissions import AllowAny

from quant_tick.views import AggregateCandleView, AggregateTradeDataView, InferenceView


class GCPAggregateTradeDataView(AggregateTradeDataView):
    """Aggregate trade data view."""

    permission_classes = (AllowAny,)


class GCPAggregateCandleView(AggregateCandleView):
    """Aggregate candle view."""

    permission_classes = (AllowAny,)


class GCPInferenceView(InferenceView):
    """Inference view."""

    permission_classes = (AllowAny,)
