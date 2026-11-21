import json
import logging
import os
from contextlib import nullcontext
from decimal import Decimal
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from unittest.mock import call, patch

import httpx2
import sentry_sdk
from django.test import SimpleTestCase
from sentry_sdk.integrations.logging import LoggingIntegration

from quant_tick.exchanges import public
from quant_tick.exchanges.public import RPCRateLimitError, get_public_json
from quant_tick.exchanges.quicknode.api import query
from quant_tick.exchanges.quicknode.blocks import get_trade_blocks
from quant_tick.exchanges.quicknode.solana import rpc


class PublicMarketRequestTest(SimpleTestCase):
    def response(self, status, text, headers=None):
        return httpx2.Response(
            status,
            text=text,
            headers=headers,
            request=httpx2.Request("GET", "https://venue.test/trades"),
        )

    def test_rpc_connections_survive_retries_and_separate_calls(self):
        requests = []
        connections = []

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def setup(self):
                super().setup()
                connections.append(self.connection)

            def do_POST(self):
                requests.append(self.rfile.read(int(self.headers["Content-Length"])))
                self.send_response(503 if len(requests) == 1 else 200)
                self.send_header("Content-Length", "2")
                self.end_headers()
                self.wfile.write(b"[]")

            def log_message(self, *args):
                pass

        public._close_rpc_transport()
        with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
            worker = Thread(target=server.serve_forever, daemon=True)
            worker.start()
            try:
                with patch("quant_tick.exchanges.public.time.sleep") as sleep:
                    for number in range(2):
                        self.assertEqual(
                            get_public_json(
                                f"http://127.0.0.1:{server.server_port}",
                                "/sample-secret",
                                {},
                                namespace="TEST",
                                max_requests=None,
                                retry=1,
                                json_body={"id": number},
                                redact_url=True,
                            ),
                            [],
                        )
                self.assertEqual(len(connections), 1)
                self.assertEqual(
                    [json.loads(body)["id"] for body in requests], [0, 0, 1]
                )
                sleep.assert_called_once_with(1)
                public._close_rpc_transport()
                self.assertIsNone(public._rpc_transport)
                get_public_json(
                    f"http://127.0.0.1:{server.server_port}",
                    "/sample-secret",
                    {},
                    namespace="TEST",
                    max_requests=None,
                    json_body={},
                    redact_url=True,
                )
                self.assertEqual(len(connections), 2)
            finally:
                public._close_rpc_transport()
                server.shutdown()
                worker.join(timeout=5)

    def test_failed_rpc_body_is_closed_before_retry(self):
        class BrokenBody(httpx2.SyncByteStream):
            def __iter__(self):
                raise httpx2.ReadError("sample-secret")
                yield

        failed = httpx2.Response(200, stream=BrokenBody())
        with (
            patch("quant_tick.exchanges.public._get_rpc_transport") as transport,
            patch("quant_tick.exchanges.public.time.sleep") as sleep,
        ):
            transport.return_value.handle_request.side_effect = [
                failed,
                self.response(200, "[]"),
            ]
            self.assertEqual(
                get_public_json(
                    "https://node.quiknode.pro/sample-secret",
                    "",
                    {},
                    namespace="TEST",
                    max_requests=None,
                    retry=1,
                    json_body={},
                    redact_url=True,
                ),
                [],
            )
        self.assertTrue(failed.is_closed)
        self.assertEqual(str(failed.request.url), "https://quiknode.pro/rpc")
        sleep.assert_called_once_with(1)

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
            (None, 1),
            ("invalid", 1),
            ("-1", 1),
            ("nan", 1),
            ("inf", 1),
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
        with (
            patch("quant_tick.exchanges.public.throttle_api_requests"),
            patch("quant_tick.exchanges.public.increment_api_total_requests"),
            patch("quant_tick.exchanges.public.time.sleep") as sleep,
            patch(
                "quant_tick.exchanges.public.httpx2.get",
                return_value=self.response(429, "{}"),
            ) as request,
            self.assertRaises(httpx2.HTTPStatusError),
        ):
            get_public_json(
                "https://venue.test", "/trades", {}, namespace="TEST", retry=6
            )
        self.assertEqual(request.call_count, 7)
        self.assertEqual(
            sleep.call_args_list, [call(delay) for delay in (1, 2, 4, 8, 16, 30)]
        )

    def test_transient_errors_share_bounded_retries(self):
        module = "quant_tick.exchanges.public"
        for method in ("get", "post", "_get_rpc_transport"):
            collect = partial(
                get_public_json,
                "https://venue.test",
                "/trades",
                {},
                namespace="TEST",
                retry=3,
                json_body={"query": "SELECT 1"} if method != "get" else None,
                redact_url=method == "_get_rpc_transport",
            )
            for final_status in (200, 408, 503):
                outcome = (
                    nullcontext()
                    if final_status == 200
                    else self.assertRaises(httpx2.HTTPStatusError)
                )
                with (
                    self.subTest(method=method, final_status=final_status),
                    patch.dict(os.environ, {}, clear=True),
                    patch(f"{module}.throttle_api_requests"),
                    patch(f"{module}.increment_api_total_requests"),
                    patch(f"{module}.time.sleep") as sleep,
                    patch(
                        f"{module}.{method}"
                        if method == "_get_rpc_transport"
                        else f"{module}.httpx2.{method}"
                    ) as client,
                ):
                    request = (
                        client.return_value.handle_request
                        if method == "_get_rpc_transport"
                        else client
                    )
                    request.side_effect = [
                        httpx2.ReadTimeout("timeout"),
                        self.response(408, "request timeout"),
                        self.response(503, "unavailable"),
                        self.response(final_status, "[]"),
                    ]
                    with outcome:
                        self.assertEqual(collect(), [])
                    self.assertEqual(request.call_count, 4)
                    self.assertEqual(sleep.call_args_list, [call(1)] * 3)
                    if final_status != 200:
                        self.assertEqual(
                            outcome.exception.response.status_code, final_status
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

    def test_provider_limits_reach_sentry_once_even_when_retry_recovers(self):
        cases = (
            (
                "QUICKNODE",
                partial(query, "SELECT 1"),
                "post",
                '{"data":[],"rows":0}',
                [],
            ),
            (
                "QUICKNODE_SOLANA",
                partial(rpc, "getBlockTime", [123]),
                "_get_rpc_transport",
                '{"jsonrpc":"2.0","id":1,"result":123}',
                123,
            ),
            (
                "QUICKNODE_BLOCKS",
                partial(get_trade_blocks, [123]),
                "_get_rpc_transport",
                (
                    '[{"jsonrpc":"2.0","id":123,"result":{"blocks":'
                    '[{"block_number":123,"events":[]}]}}]'
                ),
                {123: []},
            ),
        )
        module = "quant_tick.exchanges.public"
        for provider, collect, method, payload, expected in cases:
            scenarios = [((429, 200), [44]), ((429,) * 31, [44] * 30), ((402,), [])]
            if method == "_get_rpc_transport":
                scenarios += [
                    ((-32029, None, 503, 429, 200), [1, 1, 1, 44]),
                    (
                        (-32029, None, 503, 429) + (-32029,) * 27,
                        [1, 1, 1, 44, 16] + [30] * 25,
                    ),
                ]
            for statuses, delays in scenarios:
                events = []
                responses = []
                for status in statuses:
                    if status is None:
                        responses.append(httpx2.ReadTimeout("timeout"))
                        continue
                    body = payload if status == 200 else '{"error":"sample-test-key"}'
                    if status < 0:
                        body = json.dumps(
                            {
                                "jsonrpc": "2.0",
                                "id": 123 if provider == "QUICKNODE_BLOCKS" else 1,
                                "error": {
                                    "code": status,
                                    "message": "Rate limit reached",
                                },
                            }
                        )
                    headers = {
                        "X-RateLimit-Limit": "20",
                        "X-RateLimit-Remaining": "59" if status == 200 else "0",
                    }
                    if status == 429:
                        headers["Retry-After"] = "44"
                    responses.append(
                        self.response(200 if status < 0 else status, body, headers)
                    )
                error = (
                    RPCRateLimitError if statuses[-1] < 0 else httpx2.HTTPStatusError
                )
                outcome = (
                    nullcontext() if statuses[-1] == 200 else self.assertRaises(error)
                )
                with (
                    self.subTest(provider=provider, statuses=statuses),
                    patch.dict(
                        os.environ,
                        {
                            "QUICKNODE_API_KEY": "sample-test-key",
                            "QUICKNODE_SOLANA_URL": "https://node.quiknode.pro/sample-test-key/",
                            "QUICKNODE_HYPERCORE_URL": "https://node.quiknode.pro/sample-test-key/",
                        },
                        clear=True,
                    ),
                    patch(
                        f"{module}.{method}"
                        if method == "_get_rpc_transport"
                        else f"{module}.httpx2.{method}"
                    ) as client,
                    patch(f"{module}.throttle_api_requests"),
                    patch(f"{module}.increment_api_total_requests"),
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
                    request = (
                        client.return_value.handle_request
                        if method == "_get_rpc_transport"
                        else client
                    )
                    request.side_effect = responses
                    self.assertEqual(collect(), expected)
                self.assertEqual(request.call_count, len(statuses))
                self.assertEqual(
                    sleep.call_args_list,
                    [call(delay) for delay in delays],
                )
                self.assertEqual(len(events), 1)
                event = events[0]
                self.assertEqual(event["level"], "error")
                self.assertIn(provider, event["logentry"]["formatted"])
                self.assertIn(str(statuses[0]), event["logentry"]["formatted"])
                self.assertNotIn("sample-test-key", json.dumps(event))

    def test_credential_url_is_absent_from_logs_and_final_error(self):
        secret_url = "https://node.quiknode.pro/sample-secret/hypercore"
        for result in (
            self.response(200, "[]"),
            self.response(408, "{}"),
            self.response(429, "{}", {"Retry-After": "2"}),
            httpx2.ConnectError("failed at " + secret_url),
        ):
            events = []
            recovers = isinstance(result, httpx2.Response) and result.status_code == 200
            with (
                self.subTest(result=type(result).__name__),
                patch.dict(os.environ, {}, clear=True),
                patch("quant_tick.exchanges.public._get_rpc_transport") as transport,
                patch("quant_tick.exchanges.public.time.sleep") as sleep,
                self.assertLogs("quant_tick.exchanges.public", level=logging.INFO)
                if isinstance(result, httpx2.Response)
                else self.assertNoLogs("quant_tick.exchanges.public") as logs,
                sentry_sdk.init(
                    dsn="https://public@example.invalid/1",
                    transport=events.append,
                    default_integrations=False,
                    integrations=[
                        LoggingIntegration(level=logging.INFO, event_level=None)
                    ],
                    include_local_variables=False,
                    send_default_pii=False,
                ),
            ):
                handler = transport.return_value.handle_request
                handler.side_effect = (
                    [result] * 2 if isinstance(result, Exception) else None
                )
                handler.return_value = result
                outcome = (
                    nullcontext() if recovers else self.assertRaises(httpx2.HTTPError)
                )
                with outcome:
                    self.assertEqual(
                        get_public_json(
                            secret_url,
                            "",
                            {},
                            namespace="TEST",
                            json_body=[],
                            redact_url=True,
                            retry=1,
                        ),
                        [],
                    )
                if not recovers:
                    self.assertNotIn("sample-secret", str(outcome.exception))
                    self.assertNotIn(
                        "sample-secret", str(outcome.exception.request.url)
                    )
                    sentry_sdk.capture_exception(outcome.exception)
                    self.assertNotIn("sample-secret", json.dumps(events[0]))
                self.assertEqual(len(events), 0 if recovers else 1)
                self.assertEqual(handler.call_count, 1 if recovers else 2)
                self.assertEqual(sleep.called, not recovers)
                if isinstance(result, httpx2.Response):
                    messages = [record.getMessage() for record in logs.records]
                    self.assertNotIn("sample-secret", "\n".join(messages))
                    self.assertEqual(
                        [
                            message
                            for message in messages
                            if message.startswith("HTTP Request:")
                        ],
                        [
                            (
                                'HTTP Request: POST https://quiknode.pro/rpc "HTTP/1.1 '
                                f'{result.status_code} {result.reason_phrase}"'
                            )
                        ]
                        * handler.call_count,
                    )
