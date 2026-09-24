from decimal import Decimal
from unittest.mock import call, patch

import httpx2
from django.test import SimpleTestCase

from quant_tick.exchanges.phoenix.api import get_phoenix_response


class PhoenixAPITest(SimpleTestCase):
    def setUp(self):
        self.now = 0.0
        self.get = self.enterContext(
            patch("quant_tick.exchanges.phoenix.api.httpx2.get")
        )
        self.sleep = self.enterContext(
            patch(
                "quant_tick.exchanges.phoenix.api.time.sleep", side_effect=self.advance
            )
        )
        self.enterContext(
            patch(
                "quant_tick.exchanges.phoenix.api.time.monotonic",
                side_effect=lambda: self.now,
            )
        )

    def advance(self, seconds):
        self.now += seconds

    @staticmethod
    def response(status=200, headers=None):
        return httpx2.Response(
            status,
            request=httpx2.Request(
                "GET", "https://perp-api.phoenix.trade/v1/candles/BTC"
            ),
            headers=headers,
            text='[{"volume": 0.1234567890123456789}]',
        )

    def test_retry_after_and_decimal_precision(self):
        for header, delay in (("5", 5), ("0", 2), ("invalid", 2), (None, 2)):
            with self.subTest(header=header):
                self.get.reset_mock()
                self.sleep.reset_mock()
                self.get.side_effect = [
                    self.response(429, {"Retry-After": header} if header else {}),
                    self.response(),
                ]
                result = get_phoenix_response("/v1/candles/BTC", {"timeframe": "1m"})

                self.assertEqual(result[0]["volume"], Decimal("0.1234567890123456789"))
                self.assertEqual(self.get.call_args_list[0], self.get.call_args_list[1])
                self.assertEqual(self.sleep.call_args_list, [call(delay), call(0.5)])

    def test_retry_budget_is_shared_by_transport_and_server_failures(self):
        self.get.side_effect = [httpx2.ReadTimeout("timeout"), self.response(503)]
        with self.assertRaises(httpx2.HTTPStatusError):
            get_phoenix_response("/v1/candles/BTC", {}, retry=1)
        self.assertEqual(self.get.call_count, 2)
        self.assertEqual(self.sleep.call_args_list, [call(1), call(0.5)])

    def test_client_errors_do_not_retry(self):
        self.get.return_value = self.response(400)
        with self.assertRaises(httpx2.HTTPStatusError):
            get_phoenix_response("/v1/candles/BTC", {})
        self.get.assert_called_once()
