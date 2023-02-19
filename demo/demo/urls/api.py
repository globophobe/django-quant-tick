from django.urls import path

from quant_candles.views import AggregateCandleView, AggregateTradeDataView

urlpatterns = [
    path(
        "aggregate-trades/",
        AggregateTradeDataView.as_view(),
        name="aggregate_trades",
    ),
    path("aggregate-candles/", AggregateCandleView.as_view(), name="aggregate_candles"),
]
