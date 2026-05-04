from django.urls import path

from quant_tick.views import (
    AggregateCandleView,
    AggregateTradeDataView,
    CompactView,
    FetchExchangeDataView,
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
    path(
        "fetch-exchange-data/",
        FetchExchangeDataView.as_view(),
        name="fetch_exchange_data",
    ),
    path("compact/", CompactView.as_view(), name="compact"),
]
