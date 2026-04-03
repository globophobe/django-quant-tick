from django.urls import path

from demo.views import (
    GCPAggregateCandleView,
    GCPAggregateTradeDataView,
)

urlpatterns = [
    path(
        "aggregate-trades/<str:exchange>/",
        GCPAggregateTradeDataView.as_view(),
        name="aggregate_trades",
    ),
    path(
        "aggregate-candles/", GCPAggregateCandleView.as_view(), name="aggregate_candles"
    ),
]
