import os
from decimal import Decimal
from unittest.mock import patch

import httpx2
from django.test import SimpleTestCase

from quant_tick.exchanges.public import get_public_json


class PublicMarketRequestTest(SimpleTestCase):
    def response(self, status, text, headers=None):
        return httpx2.Response(
            status,
            text=text,
            headers=headers,
            request=httpx2.Request("GET", "https://venue.test/trades"),
        )

    def test_endpoints_share_throttle_and_preserve_decimal_values(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("quant_tick.exchanges.public.time.time", return_value=100),
            patch("quant_tick.exchanges.public.time.sleep") as sleep,
            patch(
                "quant_tick.exchanges.public.httpx2.get",
                return_value=self.response(200, "[0.123456789123456789]"),
            ),
        ):
            for endpoint in ("/trades", "/candles", "/trades"):
                result = get_public_json(
                    "https://venue.test", endpoint, {}, namespace="TEST"
                )
            self.assertEqual(result, [Decimal("0.123456789123456789")])
            self.assertEqual(os.environ["TEST_TOTAL_REQUESTS"], "3")
            sleep.assert_called_once_with(1)

    def test_rate_limit_cooldown_and_attempt_budget(self):
        failures = [
            self.response(429, "{}", {"Retry-After": "0"}),
            self.response(429, "{}", {"Retry-After": "5"}),
            self.response(200, '{"code":"50011","msg":"Rate limit reached"}'),
        ]
        for failure, delay in zip(failures, (2, 5, 2), strict=True):
            with (
                self.subTest(delay=delay),
                patch.dict(os.environ, {}, clear=True),
                patch("quant_tick.exchanges.public.time.sleep") as sleep,
                patch(
                    "quant_tick.exchanges.public.httpx2.get",
                    side_effect=[failure, self.response(200, "[]")],
                ) as request,
            ):
                self.assertEqual(
                    get_public_json(
                        "https://venue.test", "/trades", {}, namespace="TEST", retry=1
                    ),
                    [],
                )
                sleep.assert_called_once_with(delay)
                self.assertEqual(request.call_count, 2)
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("quant_tick.exchanges.public.httpx2.get", return_value=failures[0]),
            self.assertRaises(httpx2.HTTPStatusError),
        ):
            get_public_json(
                "https://venue.test", "/trades", {}, namespace="TEST", retry=0
            )

    def test_invalid_requests_are_not_retried(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch(
                "quant_tick.exchanges.public.httpx2.get",
                return_value=self.response(400, "{}"),
            ) as request,
        ):
            with self.assertRaises(httpx2.HTTPStatusError):
                get_public_json("https://venue.test", "/trades", {}, namespace="TEST")
            request.assert_called_once()
