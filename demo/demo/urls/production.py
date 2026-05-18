from django.urls import path

from quant_tick.views import (
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
        "fetch-exchange-data/",
        FetchExchangeDataView.as_view(),
        name="fetch_exchange_data",
    ),
    path("compact/", CompactView.as_view(), name="compact"),
]
