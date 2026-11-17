import json
import logging
import os
from contextlib import nullcontext
from decimal import Decimal
from functools import partial
from unittest.mock import call, patch

import httpx2
import sentry_sdk
from django.test import SimpleTestCase
from sentry_sdk.integrations.logging import LoggingIntegration

from quant_tick.exchanges.public import get_public_json
from quant_tick.exchanges.quicknode.api import query


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
            patch("quant_tick.exchanges.public.time.time", return_value=100) as clock,
            patch("quant_tick.exchanges.public.time.sleep") as sleep,
            patch(
                "quant_tick.exchanges.public.httpx2.get",
                return_value=self.response(200, "[0.123456789123456789]"),
            ),
        ):
            sleep.side_effect = lambda seconds: setattr(
                clock, "return_value", clock.return_value + seconds
            )
            for endpoint in ("/trades", "/candles", "/trades"):
                result = get_public_json(
                    "https://venue.test", endpoint, {}, namespace="TEST"
                )
            self.assertEqual(result, [Decimal("0.123456789123456789")])
            self.assertEqual(os.environ["TEST_TOTAL_REQUESTS"], "1")
            sleep.assert_called_once_with(1)

    def test_rate_limit_cooldown_and_attempt_budget(self):
        for header, delay in (
            (None, 2),
            ("invalid", 2),
            ("-1", 2),
            ("nan", 2),
            ("inf", 2),
            ("0", 0),
            ("44", 44),
            ("Thu, 01 Jan 1970 00:17:24 GMT", 44),
        ):
            failure = self.response(
                429, "{}", {"Retry-After": header} if header else {}
            )
            with (
                self.subTest(header=header),
                patch.dict(os.environ, {}, clear=True),
                patch("quant_tick.exchanges.public.time.time", return_value=1000),
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
            patch("quant_tick.exchanges.public.httpx2.get", return_value=failure),
            self.assertRaises(httpx2.HTTPStatusError),
        ):
            get_public_json(
                "https://venue.test", "/trades", {}, namespace="TEST", retry=0
            )

    def test_transport_and_server_errors_keep_bounded_retries(self):
        module = "quant_tick.exchanges.public"
        for method in ("get", "post"):
            collect = partial(
                get_public_json,
                "https://venue.test",
                "/trades",
                {},
                namespace="TEST",
                retry=3,
                json_body={"query": "SELECT 1"} if method == "post" else None,
            )
            payload, expected, delay, error = "[]", [], 1, httpx2.HTTPStatusError
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
                    patch.dict(os.environ, {}, clear=True),
                    patch(f"{module}.throttle_api_requests"),
                    patch(f"{module}.increment_api_total_requests"),
                    patch(f"{module}.time.sleep") as sleep,
                    patch(
                        f"{module}.httpx2.{method}", side_effect=responses
                    ) as request,
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

    def test_provider_limits_reach_sentry_once_even_when_retry_recovers(self):
        cases = (
            (
                "QUICKNODE",
                partial(query, "SELECT 1"),
                "post",
                '{"data":[],"rows":0}',
                [],
            ),
        )
        module = "quant_tick.exchanges.public"
        error = httpx2.HTTPStatusError
        for provider, collect, method, payload, expected in cases:
            for statuses in ((429, 200), (429, 429, 429, 429), (402,)):
                events = []
                responses = [
                    self.response(
                        status,
                        payload if status == 200 else '{"error":"sample-test-key"}',
                        {"Retry-After": "44"},
                    )
                    for status in statuses
                ]
                outcome = (
                    nullcontext() if statuses[-1] == 200 else self.assertRaises(error)
                )
                with (
                    self.subTest(provider=provider, statuses=statuses),
                    patch.dict(
                        os.environ,
                        {
                            "QUICKNODE_API_KEY": "sample-test-key",
                        },
                        clear=True,
                    ),
                    patch(
                        f"{module}.httpx2.{method}", side_effect=responses
                    ) as request,
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
                    outcome,
                ):
                    self.assertEqual(collect(), expected)
                self.assertEqual(request.call_count, len(statuses))
                self.assertEqual(
                    sum(args.args[0] == 44 for args in sleep.call_args_list),
                    sum(status == 429 for status in statuses[:-1]),
                )
                self.assertEqual(len(events), 1)
                event = events[0]
                self.assertEqual(event["level"], "error")
                self.assertIn(provider, event["logentry"]["formatted"])
                self.assertIn(str(statuses[0]), event["logentry"]["formatted"])
                self.assertNotIn("sample-test-key", json.dumps(event))

    def test_credential_url_is_absent_from_logs_and_final_error(self):
        secret_url = "https://node.quiknode.pro/sample-secret/hypercore"
        for failure in (
            self.response(429, "{}", {"Retry-After": "2"}),
            httpx2.ConnectError("failed at " + secret_url),
        ):
            with (
                self.subTest(failure=type(failure).__name__),
                patch.dict(os.environ, {}, clear=True),
                patch("quant_tick.exchanges.public.httpx2.HTTPTransport") as transport,
                patch("quant_tick.exchanges.public.time.sleep") as sleep,
                self.assertLogs("quant_tick.exchanges.public", level=logging.INFO)
                if isinstance(failure, httpx2.Response)
                else nullcontext(),
            ):
                handler = transport.return_value.__enter__.return_value.handle_request
                handler.side_effect = (
                    [failure] * 2 if isinstance(failure, Exception) else None
                )
                handler.return_value = failure
                with self.assertRaises(httpx2.HTTPError) as caught:
                    get_public_json(
                        secret_url,
                        "",
                        {},
                        namespace="TEST",
                        json_body=[],
                        redact_url=True,
                        retry=1,
                    )
                self.assertNotIn("sample-secret", str(caught.exception))
                self.assertNotIn("sample-secret", str(caught.exception.request.url))
                self.assertEqual(handler.call_count, 2)
                self.assertTrue(sleep.called)
