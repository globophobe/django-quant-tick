"""demo URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.0/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path

from quant_candles.views import (
    AggregateCandleView,
    CandleDataView,
    CandleView,
    TradeDataView,
)

urlpatterns = [
    path("admin/", admin.site.urls),
    path("trades/<str:exchange>/", TradeDataView.as_view(), name="trades"),
    path("aggregate-candles/", AggregateCandleView.as_view(), name="aggregate_candles"),
    path("candles/<str:code_name>/", CandleDataView.as_view(), name="candle_data"),
    path("candles/", CandleView.as_view(), name="candles"),
]
