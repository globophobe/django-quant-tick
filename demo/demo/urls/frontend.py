from django.contrib import admin
from django.urls import path

from quant_candles.views import CandleDataView, CandleView

urlpatterns = [
    path("admin/", admin.site.urls),
    path("candles/<str:code_name>/", CandleDataView.as_view(), name="candle_data"),
    path("candles/", CandleView.as_view(), name="candles"),
]
