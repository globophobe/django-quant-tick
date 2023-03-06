from unittest.mock import patch

from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase

from quant_candles.constants import Exchange

from .base import BaseTradeViewTest


@patch("quant_candles.views.aggregate_trades.api")
class AggregateTradeViewTest(BaseTradeViewTest, APITestCase):
    def get_url(self, exchange: Exchange = Exchange.COINBASE) -> str:
        """Get URL."""
        return reverse("aggregate_trades")

    def test_get(self, mock_command):
        """All symbols."""
        params = self.get_symbols(["test-1", "test-2"])

        response = self.client.get(self.url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        mock_symbols = self.get_mock_symbols(mock_command)
        self.assertEqual(params["symbol"], mock_symbols)

    def test_one_symbol(self, mock_command):
        """One symbol."""
        self.get_symbols(["test-1", "test-2"])
        expected = ["test-2"]

        response = self.client.get(self.url, {"api_symbol": expected})
        self.assertEqual(response.status_code, status.HTTP_200_OK)

        mock_symbols = self.get_mock_symbols(mock_command)
        self.assertEqual(expected, mock_symbols)
