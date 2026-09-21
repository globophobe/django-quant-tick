import os
from unittest.mock import call, patch

import httpx2
from django.test import SimpleTestCase

from quant_tick.exchanges.bybit.api import get_bybit_result
from quant_tick.exchanges.bybit.constants import (
    BYBIT_MAX_REQUESTS_RESET,
    BYBIT_TOTAL_REQUESTS,
)


class BybitAPITest(SimpleTestCase):
    def setUp(self):
        self.now = 1000.0
        self.enterContext(
            patch.dict(
                os.environ,
                {BYBIT_MAX_REQUESTS_RESET: "0", BYBIT_TOTAL_REQUESTS: "0"},
            )
        )
        self.get = self.enterContext(patch("quant_tick.exchanges.bybit.api.httpx2.get"))
        self.sleep = self.enterContext(
            patch("quant_tick.exchanges.bybit.api.time.sleep", side_effect=self.advance)
        )
        self.enterContext(
            patch(
                "quant_tick.exchanges.bybit.api.time.time", side_effect=lambda: self.now
            )
        )

    def advance(self, seconds):
        self.now += seconds

    def response(self, code, headers=None):
        return httpx2.Response(
            200,
            request=httpx2.Request("GET", "https://api.bybit.com/v5/market/kline"),
            headers=headers,
            json={"retCode": code, "retMsg": "test response", "result": {"list": []}},
        )

    def test_rate_limit_retries_after_reset_or_fallback(self):
        for reset, expected_delay in (
            ("1004500", 4.5),
            (None, 2),
            ("invalid", 2),
            ("999000", 2),
            ("1000000", 2),
            ("1000250", 2),
        ):
            with self.subTest(reset=reset):
                self.now = 1000.0
                os.environ[BYBIT_MAX_REQUESTS_RESET] = "0"
                os.environ[BYBIT_TOTAL_REQUESTS] = "0"
                headers = (
                    {} if reset is None else {"X-Bapi-Limit-Reset-Timestamp": reset}
                )
                self.get.reset_mock()
                self.sleep.reset_mock()
                self.get.side_effect = [self.response(10006, headers), self.response(0)]

                result = get_bybit_result("/v5/market/kline", {"symbol": "BTCUSD"})

                self.assertEqual(result, {"list": []})
                self.assertEqual(self.get.call_count, 2)
                self.assertEqual(self.get.call_args_list[0], self.get.call_args_list[1])
                self.assertEqual(
                    self.sleep.call_args_list, [call(expected_delay), call(0.5)]
                )

    def test_rate_limit_exhausts_existing_retry_budget(self):
        for kwargs, attempts in (({}, 31), ({"retry": 0}, 1), ({"retry": 2}, 3)):
            with self.subTest(kwargs=kwargs):
                self.get.reset_mock()
                self.sleep.reset_mock()
                self.get.return_value = self.response(10006)

                with self.assertRaisesRegex(
                    RuntimeError, "Bybit /v5/market/kline error 10006"
                ):
                    get_bybit_result("/v5/market/kline", {}, **kwargs)

                self.assertEqual(self.get.call_count, attempts)
                self.assertEqual(
                    self.sleep.call_args_list,
                    [call(2)] * (attempts - 1) + [call(0.5)],
                )

    def test_other_api_errors_fail_without_retry(self):
        self.get.return_value = self.response(10001)

        with self.assertRaisesRegex(RuntimeError, "Bybit /v5/market/kline error 10001"):
            get_bybit_result("/v5/market/kline", {})

        self.get.assert_called_once()
        self.sleep.assert_called_once_with(0.5)

    def test_transport_and_rate_limit_errors_share_retry_budget(self):
        self.get.side_effect = [
            httpx2.ReadTimeout("timeout"),
            self.response(10006),
            self.response(0),
        ]

        with self.assertRaisesRegex(RuntimeError, "Bybit /v5/market/kline error 10006"):
            get_bybit_result("/v5/market/kline", {}, retry=1)

        self.assertEqual(self.get.call_count, 2)
        self.assertEqual(self.sleep.call_args_list, [call(1), call(0.5)])

    def test_request_spacing_accounts_for_response_time(self):
        started = []
        latencies = iter((0.0, 0.8, 0.1))

        def respond(*args, **kwargs):
            started.append(self.now)
            self.advance(next(latencies))
            return self.response(0)

        self.get.side_effect = respond
        for category in ("spot", "linear", "inverse"):
            get_bybit_result("/v5/market/kline", {"category": category})

        self.assertEqual(started, [1000.0, 1000.5, 1001.3])
        self.assertEqual(
            [round(item.args[0], 3) for item in self.sleep.call_args_list],
            [0.5, 0.4],
        )

    def test_shared_budget_counts_failed_attempts_across_endpoints_and_categories(self):
        started = []
        outcomes = iter(
            (httpx2.ReadTimeout("timeout"), self.response(10001), self.response(0))
        )

        def respond(*args, **kwargs):
            started.append(self.now)
            result = next(outcomes)
            if isinstance(result, Exception):
                raise result
            return result

        self.get.side_effect = respond
        with patch("quant_tick.exchanges.bybit.api.MIN_ELAPSED_PER_REQUEST", 0):
            with self.assertRaises(httpx2.ReadTimeout):
                get_bybit_result("/v5/market/kline", {"category": "spot"}, retry=0)
            with self.assertRaisesRegex(RuntimeError, "error 10001"):
                get_bybit_result("/v5/market/funding/history", {"category": "linear"})
            result = get_bybit_result("/v5/market/kline", {"category": "inverse"})

        self.assertEqual(result, {"list": []})
        self.assertEqual(started, [1000.0, 1000.0, 1001.0])
        self.sleep.assert_called_once_with(1)
