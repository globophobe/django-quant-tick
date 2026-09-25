import os
from decimal import Decimal
from unittest.mock import call, patch

import httpx2
from django.test import SimpleTestCase

from quant_tick.exchanges.phoenix.api import get_phoenix_response
from quant_tick.exchanges.phoenix.constants import (
    PHOENIX_MAX_REQUESTS_RESET,
    PHOENIX_TOTAL_REQUESTS,
)


class PhoenixAPITest(SimpleTestCase):
    def setUp(self):
        self.now = 1000.0
        self.enterContext(
            patch.dict(
                os.environ,
                {PHOENIX_MAX_REQUESTS_RESET: "0", PHOENIX_TOTAL_REQUESTS: "0"},
            )
        )
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
                "quant_tick.exchanges.phoenix.api.time.time",
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
                self.assertEqual(self.sleep.call_args_list, [call(delay), call(1)])

    def test_retry_budget_is_shared_by_transport_and_server_failures(self):
        self.get.side_effect = [httpx2.ReadTimeout("timeout"), self.response(503)]
        with self.assertRaises(httpx2.HTTPStatusError):
            get_phoenix_response("/v1/candles/BTC", {}, retry=1)
        self.assertEqual(self.get.call_count, 2)
        self.assertEqual(self.sleep.call_args_list, [call(1), call(1)])

    def test_client_errors_do_not_retry(self):
        self.get.return_value = self.response(400)
        with self.assertRaises(httpx2.HTTPStatusError):
            get_phoenix_response("/v1/candles/BTC", {})
        self.get.assert_called_once()

    def test_request_spacing_accounts_for_response_time_across_endpoints(self):
        started = []
        latencies = iter((0.0, 1.2, 0.1, 0.4))

        def respond(*args, **kwargs):
            started.append(self.now)
            self.advance(next(latencies))
            return self.response()

        self.get.side_effect = respond
        for path in (
            "/v1/candles/BTC",
            "/v1/trades/BTC/fills",
            "/v1/funding/BTC/rates",
            "/v1/market/BTC/stats",
        ):
            get_phoenix_response(path, {})

        self.assertEqual(started, [1000.0, 1001.0, 1002.2, 1003.2])
        self.assertEqual(
            [round(item.args[0], 3) for item in self.sleep.call_args_list],
            [1.0, 0.9, 0.6],
        )

    def test_failed_attempt_shares_request_budget_across_endpoints(self):
        started = []
        outcomes = iter((httpx2.ReadTimeout("timeout"), self.response()))

        def respond(*args, **kwargs):
            started.append(self.now)
            result = next(outcomes)
            if isinstance(result, Exception):
                raise result
            return result

        self.get.side_effect = respond
        with patch("quant_tick.exchanges.phoenix.api.MIN_ELAPSED_PER_REQUEST", 0):
            with self.assertRaises(httpx2.ReadTimeout):
                get_phoenix_response("/v1/candles/BTC", {}, retry=0)
            get_phoenix_response("/v1/trades/BTC/fills", {})

        self.assertEqual(started, [1000.0, 1001.0])
        self.sleep.assert_called_once_with(1)
