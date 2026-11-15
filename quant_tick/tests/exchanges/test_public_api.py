import json
import logging
import os
import traceback
from contextlib import nullcontext
from decimal import Decimal
from functools import partial
from unittest.mock import call, patch

import httpx2
import sentry_sdk
from django.test import SimpleTestCase
from sentry_sdk.integrations.logging import LoggingIntegration

from quant_tick.exchanges.coinapi.api import get_coinapi_response
from quant_tick.exchanges.coinapi.flat_files import get_flat_file_content
from quant_tick.exchanges.femtos.api import get_femtos_response
from quant_tick.exchanges.public import get_public_json
from quant_tick.lib.download import ArchiveDownloadError


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

    def test_transport_and_server_errors_keep_bounded_retries(self):
        for collect, module, payload, expected, delay, error in (
            (
                partial(get_coinapi_response, "/v1/trades/test/history", {}),
                "quant_tick.exchanges.public",
                "[]",
                [],
                1,
                httpx2.HTTPStatusError,
            ),
            (
                partial(get_flat_file_content, "archive.csv.gz"),
                "quant_tick.exchanges.coinapi.flat_files",
                "archive",
                b"archive",
                2,
                ArchiveDownloadError,
            ),
        ):
            for recovers in (True, False):
                responses = [
                    httpx2.ReadTimeout("timeout"),
                    self.response(503, "unavailable"),
                    self.response(503, "unavailable"),
                    self.response(200 if recovers else 503, payload),
                ]
                outcome = nullcontext() if recovers else self.assertRaises(error)
                with (
                    self.subTest(module=module, recovers=recovers),
                    patch.dict(os.environ, {"COINAPI_API_KEY": "test-key"}, clear=True),
                    patch(f"{module}.throttle_api_requests"),
                    patch(f"{module}.increment_api_total_requests"),
                    patch(f"{module}.time.sleep") as sleep,
                    patch(f"{module}.httpx2.get", side_effect=responses) as request,
                    outcome,
                ):
                    self.assertEqual(collect(), expected)
                self.assertEqual(request.call_count, 4)
                self.assertEqual(sleep.call_args_list, [call(delay)] * 3)

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

    def test_provider_limits_fail_immediately_with_safe_traceback_and_sentry_details(
        self,
    ):
        cases = (
            (
                "COINAPI",
                partial(get_coinapi_response, "/v1/trades/test/history", {}),
                "quant_tick.exchanges.public",
                httpx2.HTTPStatusError,
            ),
            (
                "FEMTOS",
                partial(get_femtos_response, "/candles", {}),
                "quant_tick.exchanges.public",
                httpx2.HTTPStatusError,
            ),
            (
                "CoinAPI Flat Files",
                partial(get_flat_file_content, "archive.csv.gz"),
                "quant_tick.exchanges.coinapi.flat_files",
                ArchiveDownloadError,
            ),
        )
        for provider, collect, module, error in cases:
            for status, retry_after in (
                (429, None),
                (429, "17"),
                (429, "Sun, 15 Nov 2026 12:00:00 GMT"),
                (402, None),
            ):
                events = []
                response_headers = {
                    "X-RateLimit-Limit": "1000",
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": "2026-11-16T12:00:00Z",
                    "X-ConcurrencyLimit-Limit": "10",
                    "X-ConcurrencyLimit-Remaining": "3",
                    "Authorization": "sample-test-key",
                    "Set-Cookie": "private-cookie",
                    "X-Internal": "private-header",
                }
                if retry_after is not None:
                    response_headers["Retry-After"] = retry_after
                failure = self.response(
                    status,
                    json.dumps(
                        {
                            "error": "Request credit limit exceeded",
                            "credential": "sample-test-key",
                            "padding": "x" * 3000 + "body-end-marker",
                        }
                    ),
                    response_headers,
                )
                with (
                    self.subTest(
                        provider=provider, status=status, retry_after=retry_after
                    ),
                    patch.dict(
                        os.environ,
                        {
                            "COINAPI_API_KEY": "sample-test-key",
                            "FEMTOS_API_KEY": "sample-test-key",
                        },
                        clear=True,
                    ),
                    patch(f"{module}.httpx2.get", return_value=failure) as request,
                    patch(f"{module}.time.sleep") as sleep,
                    self.assertLogs(module, level=logging.ERROR),
                    sentry_sdk.init(
                        dsn="https://public@example.invalid/1",
                        transport=events.append,
                        default_integrations=False,
                        integrations=[LoggingIntegration(level=None)],
                        include_local_variables=False,
                        send_default_pii=False,
                    ),
                    self.assertRaises(error) as outcome,
                ):
                    collect()
                request.assert_called_once()
                sleep.assert_not_called()
                self.assertEqual(len(events), 1)
                event = events[0]
                self.assertEqual(event["level"], "error")
                self.assertIn(provider, event["logentry"]["formatted"])
                self.assertIn(str(status), event["logentry"]["formatted"])
                rendered = "".join(traceback.format_exception(outcome.exception))
                for text in (rendered, event["logentry"]["formatted"]):
                    self.assertIn("Request credit limit exceeded", text)
                    self.assertIn(
                        '"Retry-After": "' + (retry_after or "<not provided>") + '"',
                        text,
                    )
                    self.assertIn('"X-RateLimit-Remaining": "0"', text)
                    self.assertIn('"X-RateLimit-Reset": "2026-11-16T12:00:00Z"', text)
                    self.assertIn('"X-ConcurrencyLimit-Remaining": "3"', text)
                    self.assertIn("[redacted]", text)
                    self.assertIn("...[truncated]", text)
                    for private in (
                        "sample-test-key",
                        "private-cookie",
                        "private-header",
                        "body-end-marker",
                    ):
                        self.assertNotIn(private, text)
                self.assertNotIn("sample-test-key", json.dumps(event))
                self.assertLess(len(event["logentry"]["formatted"]), 3000)
