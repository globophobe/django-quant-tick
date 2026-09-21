import os
import time
from datetime import UTC, datetime
from decimal import Decimal
from unittest.mock import call, patch

import httpx2
import time_machine
from django.test import SimpleTestCase

from quant_tick.exchanges.bitfinex.api import (
    get_bitfinex_api_response,
    get_bitfinex_api_url,
)
from quant_tick.exchanges.bitfinex.constants import (
    BITFINEX_MAX_REQUESTS_RESET,
    BITFINEX_TOTAL_REQUESTS,
)
from quant_tick.exchanges.bitfinex.trades import get_trades


class BitfinexAPITest(SimpleTestCase):
    def setUp(self):
        self.clock = self.enterContext(time_machine.travel(1000.0, tick=False))
        self.enterContext(
            patch.dict(
                os.environ,
                {BITFINEX_MAX_REQUESTS_RESET: "1060", BITFINEX_TOTAL_REQUESTS: "0"},
            )
        )
        self.get = self.enterContext(
            patch("quant_tick.exchanges.bitfinex.api.httpx2.get")
        )
        self.sleep = self.enterContext(
            patch(
                "quant_tick.exchanges.bitfinex.api.time.sleep",
                side_effect=self.clock.shift,
            )
        )
        self.url = "https://api-pub.bitfinex.com/v2/trades/tBTCUSD/hist?limit=1"

    def response(self, status, headers=None, data=None):
        return httpx2.Response(
            status,
            request=httpx2.Request("GET", self.url),
            headers=headers,
            json=[] if data is None else data,
        )

    def test_rate_limit_honors_retry_after_or_default_delay(self):
        for headers, delay in (({"Retry-After": "4"}, 4), ({}, 1)):
            with self.subTest(headers=headers):
                self.get.reset_mock()
                self.sleep.reset_mock()
                self.get.side_effect = [self.response(429, headers), self.response(200)]

                result = get_bitfinex_api_response(
                    get_bitfinex_api_url, self.url, pagination_id=123, retry=1
                )

                self.assertEqual(result, [])
                self.assertEqual(
                    self.get.call_args_list, [call(self.url + "&end=123")] * 2
                )
                self.sleep.assert_called_once_with(delay)

    def test_rate_limit_exhausts_retry_budget(self):
        for kwargs, attempts in (({"retry": 0}, 1), ({"retry": 2}, 3), ({}, 31)):
            with self.subTest(kwargs=kwargs):
                self.get.reset_mock()
                self.sleep.reset_mock()
                self.get.side_effect = [self.response(429)] * attempts

                with self.assertRaises(httpx2.HTTPStatusError) as raised:
                    get_bitfinex_api_response(get_bitfinex_api_url, self.url, **kwargs)

                self.assertEqual(raised.exception.response.status_code, 429)
                self.assertEqual(self.get.call_count, attempts)
                if attempts == 1:
                    self.sleep.assert_not_called()

    def test_rate_limits_and_http_and_transport_errors_share_retry_budget(self):
        self.get.side_effect = [
            self.response(429, {"Retry-After": "4"}),
            httpx2.ReadTimeout("timeout"),
            self.response(500),
            self.response(200),
        ]

        with self.assertRaises(httpx2.HTTPStatusError) as raised:
            get_bitfinex_api_response(get_bitfinex_api_url, self.url, retry=2)

        self.assertEqual(raised.exception.response.status_code, 500)
        self.assertEqual(self.get.call_count, 3)
        self.assertEqual(os.environ[BITFINEX_TOTAL_REQUESTS], "3")
        self.assertEqual(self.sleep.call_args_list, [call(4), call(1)])

    def test_failed_attempts_share_budget_across_endpoints(self):
        os.environ[BITFINEX_TOTAL_REQUESTS] = "8"
        started = []
        outcomes = iter(
            (httpx2.ReadTimeout("timeout"), self.response(500), self.response(200))
        )

        def respond(*args, **kwargs):
            started.append(time.time())
            outcome = next(outcomes)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        self.get.side_effect = respond
        for path, error in (
            ("trades/tBTCUSD/hist", httpx2.ReadTimeout),
            ("candles/trade:1m:tBTCUSD/hist", httpx2.HTTPStatusError),
        ):
            with self.assertRaises(error):
                get_bitfinex_api_response(
                    get_bitfinex_api_url,
                    f"https://api-pub.bitfinex.com/v2/{path}",
                    retry=0,
                )
        self.assertEqual(get_bitfinex_api_response(get_bitfinex_api_url, self.url), [])

        self.assertEqual(started, [1000.0, 1000.0, 1060.0])
        self.sleep.assert_called_once_with(60)

    def test_paginated_requests_account_for_response_time(self):
        timestamp_from = datetime(2026, 9, 1, tzinfo=UTC)
        start_ms = int(timestamp_from.timestamp() * 1000)
        rows = [[minute, start_ms + minute * 60000, 0.1, 100] for minute in (2, 1, 0)]
        pages = iter(rows)
        latencies = iter((0, 8, 1))
        started = []

        def respond(*args, **kwargs):
            started.append(time.time())
            self.clock.shift(next(latencies))
            return self.response(200, data=[next(pages)])

        self.get.side_effect = respond
        with patch("quant_tick.exchanges.bitfinex.trades.TRADE_MAX_RESULTS", 1):
            result, _, _ = get_trades("tBTCUSD", timestamp_from, start_ms + 120000)

        self.assertEqual([row[1] for row in result], [row[1] for row in rows])
        self.assertEqual(result[0][2], Decimal("0.1"))
        self.assertEqual(started, [1000.0, 1006.0, 1014.0])
        self.assertEqual(self.sleep.call_args_list, [call(6), call(5)])
