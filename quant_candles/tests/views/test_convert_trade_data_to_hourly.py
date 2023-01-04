from unittest.mock import patch

from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase

from quant_candles.constants import Exchange

from .base import BaseTradeViewTest


class ConvertToHourlyViewTest(BaseTradeViewTest, APITestCase):
    def get_url(self, exchange: Exchange = Exchange.COINBASE) -> str:
        """Get URL."""
        return reverse("convert_trade_data_to_hourly", kwargs={"exchange": exchange})

    @patch(
        "quant_candles.views.convert_trade_data_to_hourly.convert_trade_data_to_hourly"
    )
    def test_exchange_default_all_symbols(self, mock_command):
        """If no symbol, default all symbols."""
        params = self.get_symbols(["test-1", "test-2"])

        self.client.get(self.url)

        mock_symbols = self.get_mock_symbols(mock_command)
        self.assertEqual(params["symbol"], mock_symbols)

    @patch(
        "quant_candles.views.convert_trade_data_to_hourly.convert_trade_data_to_hourly"
    )
    def test_one_symbol(self, mock_command):
        """One symbol."""
        self.get_symbols(["test-1", "test-2"])
        expected = ["test-2"]

        self.client.get(self.url, {"symbol": expected})

        mock_symbols = self.get_mock_symbols(mock_command)
        self.assertEqual(expected, mock_symbols)

    def test_nonexistant_exchange(self):
        """Exchange does not exist."""
        url = self.get_url("test")
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

    def test_nonexistant_symbol(self):
        """Symbol does not exist."""
        response = self.client.get(self.url, {"symbol": "test"})
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
