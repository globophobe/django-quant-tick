from django.urls import path

from quant_tick.views import (
    AggregateCandleView,
    AggregateTradeDataView,
)

urlpatterns = [
    path(
        "aggregate-trades/<str:exchange>/",
        AggregateTradeDataView.as_view(),
        name="aggregate_trades",
    ),
    path(
        "aggregate-candles/", AggregateCandleView.as_view(), name="aggregate_candles"
    ),
]
