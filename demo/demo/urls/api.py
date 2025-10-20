from django.urls import path

from demo.views import (
    GCPAggregateCandleView,
    GCPAggregateTradeDataView,
    GCPInferenceView,
)

urlpatterns = [
    path(
        "aggregate-trades/",
        GCPAggregateTradeDataView.as_view(),
        name="aggregate_trades",
    ),
    path(
        "aggregate-candles/", GCPAggregateCandleView.as_view(), name="aggregate_candles"
    ),
    path("inference/", GCPInferenceView.as_view(), name="inference"),
]
