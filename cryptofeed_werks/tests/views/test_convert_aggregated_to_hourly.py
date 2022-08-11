from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.urls import reverse
from rest_framework import status

from cryptofeed_werks.constants import Exchange

from .base import BaseViewTest


class ConvertToHourlyViewTest(BaseViewTest):
    def setUp(self):
        User = get_user_model()
        user = User.objects.create(username="test")
        self.client.force_authenticate(user)
        self.url = self.get_url()

    def get_url(self, exchange: Exchange = Exchange.FTX) -> str:
        """Get URL."""
        return reverse("convert_aggregated_to_hourly", kwargs={"exchange": exchange})

    @patch(
        "cryptofeed_werks.views.convert_aggregated_to_hourly"
        ".convert_aggregated_to_hourly"
    )
    def test_exchange_default_all_symbols(self, mock_command):
        """If no symbol, default all symbols."""
        params = self.get_symbols(["test-1", "test-2"])

        self.client.get(self.url)

        mock_symbols = self.get_mock_symbols(mock_command)
        self.assertEqual(params["symbol"], mock_symbols)

    @patch(
        "cryptofeed_werks.views.convert_aggregated_to_hourly"
        ".convert_aggregated_to_hourly"
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
