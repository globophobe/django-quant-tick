from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase

from quant_candles.constants import Exchange
from quant_candles.models import Candle, GlobalSymbol, Symbol

from .base import BaseViewTest


class AggregateCandleViewTest(BaseViewTest, APITestCase):
    def setUp(self):
        super().setUp()
        self.url = reverse("aggregate_candles")
        self.global_symbol = GlobalSymbol.objects.create(name="global-symbol")
        self.symbol = Symbol.objects.create(
            global_symbol=self.global_symbol,
            exchange=Exchange.COINBASE,
            api_symbol="test",
        )

    def test_get(self):
        """List of candles."""
        candle = Candle.objects.create()
        candle.symbols.add(self.symbol)
        response = self.client.get(self.url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertEqual(len(data), 1)

    def test_one_candle(self):
        """One candle."""
        for i in range(2):
            candle = Candle.objects.create()
            candle.symbols.add(self.symbol)
        response = self.client.get(self.url, {"code_name": candle.code_name})
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertEqual(len(data), 1)
