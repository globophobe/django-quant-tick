import json
import os
import re
from copy import deepcopy
from decimal import Decimal
from functools import partial
from itertools import pairwise
from pathlib import Path
from unittest.mock import call, patch

import httpx2
import pandas as pd
from django.core.exceptions import ImproperlyConfigured
from django.test import SimpleTestCase

from quant_tick.exchanges.quicknode import api, solana
from quant_tick.exchanges.quicknode.blocks import get_trade_blocks
from quant_tick.exchanges.quicknode.candles import quicknode_candles, validation_candles
from quant_tick.exchanges.quicknode.trades import iter_trades, trade_frame
from quant_tick.lib import aggregate_trades, volume_filter_with_time_window


class QuickNodeTradesTest(SimpleTestCase):
    def setUp(self):
        self.start = pd.Timestamp("2026-11-17T05:49Z")
        self.end = self.start + pd.Timedelta(minutes=1)
        path = Path(__file__).parent.parent / "fixtures/quicknode/ordered_block.json"
        self.fixture = json.loads(path.read_text())
        self.rows = self.fixture["rows"]

    def collect(self, coin="BTC"):
        return list(iter_trades(coin, self.start, self.end, bounds=(1, 2000000000)))

    def test_native_order_and_filter_survive_pages_splitting_one_block(self):
        pages = [self.rows[index : index + 4] for index in range(0, len(self.rows), 4)]
        with (
            patch("quant_tick.exchanges.quicknode.trades.PAGE_SIZE", 4),
            patch(
                "quant_tick.exchanges.quicknode.trades.query", side_effect=pages
            ) as request,
            patch("quant_tick.exchanges.quicknode.blocks.get_trade_blocks") as raw,
        ):
            frame = trade_frame(self.collect())
        raw.assert_not_called()
        self.assertEqual(
            frame.id_trade.tolist(), list(map(str, self.fixture["expected_order"]))
        )
        self.assertEqual(frame.notional.sum(), Decimal(5))
        self.assertEqual(frame.price.iloc[0], Decimal(64892))
        self.assertEqual(frame.price.iloc[-1], Decimal(64887))
        self.assertIn("(block_number, trade_id) >", request.call_args_list[1].args[0])
        filtered = volume_filter_with_time_window(
            aggregate_trades(frame), min_volume=1000, window=None
        )
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered.iloc[0].price, Decimal(64887))
        self.assertEqual(filtered.iloc[0].notional, Decimal(5))

    def independent_rows(self):
        rows = deepcopy(self.rows[:2])
        for index, row in enumerate(rows):
            row["buyer_address"] = "buyer-" + str(index)
            row["seller_address"] = "seller-" + str(index)
        return rows

    def raw_events(self, rows, coin="BTC"):
        return [
            [
                "taker",
                {
                    "coin": coin,
                    "crossed": True,
                    "tid": row["trade_id"],
                    "px": row["price"],
                    "sz": row["size"],
                    "side": row["side"],
                    "time": int(pd.Timestamp(row["timestamp"]).timestamp() * 1000),
                },
            ]
            for row in rows
        ]

    def test_ambiguous_sql_uses_raw_sequence_and_checks_the_complete_row_set(self):
        rows = self.independent_rows()
        for coin in ("BTC", "@142"):
            for failure in (
                None,
                "missing",
                "extra",
                "value",
                "duplicate",
                "zero_id",
                "dust_trade_id",
            ):
                events = self.raw_events(rows[::-1], coin)
                if coin == "@142":
                    dust = deepcopy(events[0])
                    dust[1].update(
                        tid=0, dir="Spot Dust Conversion", sz="0.00000492"
                    )
                    events[1:1] = [dust, deepcopy(dust)]
                    for direction in (
                        "Partial Borrow Liquidation",
                        "Backstop Borrow Liquidation",
                    ):
                        for size in ("0", "2"):
                            transfer = deepcopy(events[0])
                            transfer[1].update(
                                tid=900000 + len(events),
                                dir=direction,
                                sz=size,
                                px="1",
                            )
                            events.insert(1, transfer)
                if failure == "missing":
                    events.pop()
                if failure in ("extra", "duplicate", "zero_id", "dust_trade_id"):
                    events.append(deepcopy(events[0]))
                if failure in ("extra", "dust_trade_id"):
                    events[-1][1]["tid"] = 999999
                if failure == "zero_id":
                    events[-1][1].update(tid=0, dir="Sell")
                if failure == "dust_trade_id":
                    events[-1][1]["dir"] = "Spot Dust Conversion"
                if failure == "value":
                    events[0][1]["sz"] = "99"
                with (
                    self.subTest(coin=coin, failure=failure),
                    patch("quant_tick.exchanges.quicknode.trades.PAGE_SIZE", 1),
                    patch(
                        "quant_tick.exchanges.quicknode.trades.query",
                        side_effect=[[rows[0]], [rows[1]], []],
                    ) as query,
                    patch(
                        "quant_tick.exchanges.quicknode.blocks.get_trade_blocks",
                        return_value={rows[0]["block_number"]: events},
                    ) as raw,
                ):
                    if failure:
                        with self.assertRaisesRegex(
                            ValueError, "QuickNode.*(disagree|omitted)"
                        ):
                            self.collect(coin)
                    else:
                        frame = trade_frame(self.collect(coin))
                        self.assertEqual(
                            frame.id_trade.tolist(),
                            [str(r["trade_id"]) for r in rows[::-1]],
                        )
                        self.assertEqual(
                            frame.notional.sum(),
                            sum(Decimal(r["size"]) for r in rows),
                        )
                    raw.assert_called_once_with([rows[0]["block_number"]])
                    sql = query.call_args_list[0].args[0]
                    self.assertIn(
                        "AND NOT (buyer_dir IN ('Partial Borrow Liquidation', "
                        "'Backstop Borrow Liquidation') AND seller_dir=buyer_dir)",
                        sql,
                    )

    def test_invalid_sql_pages_fail_before_persistence(self):
        for change in (
            {"size": "0"},
            {"size": "-1"},
            {"size": "NaN"},
            {"size": "Infinity"},
            {"price": "-1"},
            {"side": "X"},
            {"buyer_crossed": 1},
            {"timestamp": self.end.isoformat()},
        ):
            with (
                self.subTest(change=change),
                patch(
                    "quant_tick.exchanges.quicknode.trades.query",
                    return_value=[{**self.rows[0], **change}],
                ),
                self.assertRaises(ValueError),
            ):
                self.collect()
        with (
            patch("quant_tick.exchanges.quicknode.trades.PAGE_SIZE", 1),
            patch(
                "quant_tick.exchanges.quicknode.trades.query",
                return_value=[self.rows[0]],
            ),
            self.assertRaisesRegex(ValueError, "duplicated"),
        ):
            self.collect()

    def test_absent_block_index_does_not_discard_trade_history(self):
        with (
            patch(
                "quant_tick.exchanges.quicknode.trades.block_range", return_value=None
            ),
            patch(
                "quant_tick.exchanges.quicknode.trades.query",
                return_value=[self.rows[0]],
            ) as query,
        ):
            self.assertEqual(len(list(iter_trades("BTC", self.start, self.end))), 1)
        self.assertNotIn("BETWEEN", query.call_args.args[0])

    def test_historical_candles_use_verified_order_and_skip_absent_minutes(self):
        with (
            patch(
                "quant_tick.exchanges.quicknode.trades.query", return_value=self.rows
            ),
            patch(
                "quant_tick.exchanges.quicknode.trades.block_range", return_value=None
            ),
        ):
            result = quicknode_candles(
                "BTC", self.start, self.end + pd.Timedelta(minutes=1)
            )
        self.assertEqual(result.index.tolist(), [self.start])
        row = result.iloc[0]
        self.assertEqual(
            (row.open, row.close, row.high, row.low),
            tuple(map(Decimal, (64892, 64887, 64892, 64887))),
        )
        self.assertEqual(row.notional, Decimal(5))
        self.assertEqual(row.trades, len(self.rows))
        self.assertEqual(
            row.volume, sum(Decimal(r["size"]) * Decimal(r["price"]) for r in self.rows)
        )

    def test_validation_batches_preserve_complete_minutes_and_exact_quantity(self):
        end = self.start + pd.Timedelta(hours=25, minutes=2)
        moments = pd.date_range(self.start, end, freq="min", inclusive="left")
        # Empty observations stay absent; they must not become zero-volume bars.
        moments = moments[(moments.minute % 3 != 0) & (moments.hour != 12)]
        quantity = Decimal("5.000000000000000001")
        completed = []

        def request(sql):
            values = re.search(
                r"AND timestamp >= '([^']+)' AND timestamp < '([^']+)'", sql
            ).groups()
            lower, upper = map(api.timestamp, values)
            self.assertIn(
                "AND NOT (buyer_dir IN ('Partial Borrow Liquidation', "
                "'Backstop Borrow Liquidation') AND seller_dir=buyer_dir)",
                sql,
            )
            if upper - lower > pd.Timedelta(hours=6):
                httpx2.Response(
                    400,
                    json={"message": error},
                    request=httpx2.Request("POST", "https://api.quicknode.com"),
                ).raise_for_status()
            completed.append((lower, upper))
            return [
                {"minute": moment.isoformat(), "notional": str(quantity), "trades": 18}
                for moment in moments
                if lower <= moment < upper
            ]

        for coin in ("BTC", "@142"):
            for error in ("TOO_MANY_BYTES", "TOO_MANY_ROWS"):
                for bounds in (None, (1, 2000000000)):
                    completed.clear()
                    with (
                        self.subTest(coin=coin, error=error, bounds=bounds),
                        patch(
                            "quant_tick.exchanges.quicknode.candles.block_range",
                            return_value=bounds,
                        ),
                        patch(
                            "quant_tick.exchanges.quicknode.candles.query",
                            side_effect=request,
                        ),
                    ):
                        result = validation_candles(coin, self.start, end)
                        self.assertEqual(result.index.tolist(), moments.tolist())
                        self.assertEqual(
                            result.notional.tolist(), [quantity] * len(moments)
                        )
                        self.assertEqual(result.trades.tolist(), [18] * len(moments))
                        self.assertEqual(completed[0][0], self.start)
                        self.assertEqual(completed[-1][1], end)
                        for (_, left_end), (right_start, _) in pairwise(completed):
                            self.assertEqual(left_end, right_start)

    def test_validation_does_not_hide_other_errors_or_split_within_a_minute(self):
        for status, error, duration in (
            (400, "TOO_MANY_BYTES", pd.Timedelta(minutes=1)),
            (400, "TOO_MANY_ROWS", pd.Timedelta(minutes=1)),
            (400, "SYNTAX_ERROR", pd.Timedelta(days=1)),
            (401, "TOO_MANY_BYTES", pd.Timedelta(days=1)),
        ):
            response = httpx2.Response(
                status,
                json={"message": error},
                request=httpx2.Request("POST", "https://api.quicknode.com"),
            )
            with (
                self.subTest(status=status, error=error, duration=duration),
                patch(
                    "quant_tick.exchanges.quicknode.candles.block_range",
                    return_value=None,
                ),
                patch(
                    "quant_tick.exchanges.quicknode.candles.query",
                    side_effect=lambda sql, response=response: (
                        response.raise_for_status()
                    ),
                ) as request,
                self.assertRaises(httpx2.HTTPStatusError) as raised,
            ):
                validation_candles("@142", self.start, self.start + duration)
            self.assertIs(raised.exception.response, response)
            request.assert_called_once()


class QuickNodeRequestTest(SimpleTestCase):
    def test_rpc_throttling_honors_timing_without_retrying_node_errors(self):
        routes = (
            (
                partial(solana.rpc, "getBlockTime", [123]),
                {"jsonrpc": "2.0", "id": 1, "result": 123},
                123,
            ),
            (
                partial(get_trade_blocks, [123]),
                [
                    {
                        "jsonrpc": "2.0",
                        "id": 123,
                        "result": {"blocks": [{"block_number": 123, "events": []}]},
                    }
                ],
                {123: []},
            ),
        )
        cases = (
            (429, "Requests per second exceed the plan", {}, 1),
            (-32007, "Per-second request limit reached", {}, 1),
            (-32008, "Per-minute request limit reached", {"Retry-After": "44"}, 44),
            (-32011, "Method rate limit reached", {}, 1),
            (-32029, "Rate limit reached; try again in 1500 ms", {}, 1.5),
            (-32029, "Rate limit reached; retry after 5 seconds", {}, 5),
            (-32029, "Rate limit reached; wait 5 seconds", {"Retry-After": "44"}, 44),
            (
                -32029,
                "Rate limit reached; wait 5 seconds",
                {"Retry-After": "invalid"},
                5,
            ),
            (-32029, "Rate limit reached", {"Retry-After": "0"}, 0),
            (-32007, "Slot 123 was skipped or missing due to ledger jump", {}, None),
            (-32011, "Transaction history is not available from this node", {}, None),
            (-32602, "Invalid param", {}, None),
        )
        module = "quant_tick.exchanges.public"
        for collect, success, expected in routes:
            for code, message, headers, delay in cases:
                failure = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "error": {"code": code, "message": message},
                }
                if isinstance(success, list):
                    failure["id"] = 123
                    failure = [failure]
                with (
                    self.subTest(
                        route=collect.func.__name__,
                        code=code,
                        message=message,
                        headers=headers,
                    ),
                    patch.dict(
                        os.environ,
                        {
                            "QUICKNODE_SOLANA_URL": "https://node.quiknode.pro/test-token/",
                            "QUICKNODE_HYPERCORE_URL": "https://node.quiknode.pro/test-token/",
                        },
                        clear=True,
                    ),
                    patch(f"{module}.httpx2.HTTPTransport") as transport,
                    patch(f"{module}.throttle_api_requests"),
                    patch(f"{module}.increment_api_total_requests"),
                    patch(f"{module}.time.sleep") as sleep,
                ):
                    request = (
                        transport.return_value.__enter__.return_value.handle_request
                    )
                    request.side_effect = [
                        httpx2.Response(200, json=failure, headers=headers),
                        httpx2.Response(200, json=success),
                    ]
                    if delay is None:
                        with self.assertRaises(ValueError):
                            collect()
                    else:
                        self.assertEqual(collect(), expected)
                    self.assertEqual(request.call_count, 1 if delay is None else 2)
                    self.assertEqual(
                        sleep.call_args_list, [] if delay is None else [call(delay)]
                    )

    def test_sql_uses_response_quota_and_checks_schema(self):
        for headers, delay in (
            ({"X-RateLimit-Limit": "20", "X-RateLimit-Remaining": "59"}, None),
            ({"X-RateLimit-Limit": "20", "X-RateLimit-Remaining": "0"}, 0.05),
            ({"X-RateLimit-Limit": "4", "X-RateLimit-Remaining": "0"}, 0.25),
            ({}, None),
            ({"X-RateLimit-Limit": "20"}, None),
            ({"X-RateLimit-Remaining": "0"}, None),
        ):
            with (
                self.subTest(headers=headers),
                patch.dict(os.environ, {"QUICKNODE_API_KEY": "test"}, clear=True),
                patch("quant_tick.exchanges.public.time.time", return_value=100),
                patch("quant_tick.exchanges.public.time.sleep") as sleep,
                patch(
                    "quant_tick.exchanges.public.httpx2.post",
                    return_value=httpx2.Response(
                        200,
                        json={"data": [], "rows": 0},
                        headers=headers,
                        request=httpx2.Request(
                            "POST", "https://api.quicknode.com/sql/rest/v1/query"
                        ),
                    ),
                ) as request,
            ):
                for _ in range(2):
                    self.assertEqual(api.query("SELECT 1"), [])
            self.assertEqual(request.call_count, 2)
            self.assertEqual(
                sleep.call_args_list, [call(delay)] * 2 if delay is not None else []
            )
            kwargs = request.call_args.kwargs
            self.assertIn("max_rows_to_read=", kwargs["json"]["query"])
            self.assertIn("overflow_mode='throw'", kwargs["json"]["query"])
        for payload in (
            {},
            {"data": [], "rows": 1},
            {"data": [], "rows": 0, "error": "partial"},
        ):
            with (
                patch.dict(os.environ, {"QUICKNODE_API_KEY": "test"}),
                patch(
                    "quant_tick.exchanges.quicknode.api.get_public_json",
                    return_value=payload,
                ),
                self.assertRaises(ValueError),
            ):
                api.query("SELECT 1")

    def test_raw_history_boundary_is_cached_and_not_inferred_from_summaries(self):
        api._history_start.cache_clear()
        self.addCleanup(api._history_start.cache_clear)
        with patch(
            "quant_tick.exchanges.quicknode.api.query",
            return_value=[{"block_time": "2025-07-27 08:50:10.273720"}],
        ) as query:
            self.assertEqual(
                api.history_start(), pd.Timestamp("2025-07-27T08:50:10.273720Z")
            )
            api.history_start()
        query.assert_called_once()
        self.assertIn("FROM hyperliquid_trades", query.call_args.args[0])

    def test_coin_literal_rejects_sql_and_missing_index_coverage_is_only_a_hint(self):
        for coin in ("BTC';DROP", "BTC USD", "", "BTC/USDC"):
            with self.assertRaises(ValueError):
                api.coin_literal(coin)
        start = pd.Timestamp("2026-11-01T00:00Z")
        end = start + pd.Timedelta(minutes=1)
        for rows in ([], [{"blocks": 0}], [{"blocks": 1, "start": end, "end": end}]):
            with patch("quant_tick.exchanges.quicknode.api.query", return_value=rows):
                self.assertIsNone(api.block_range(start, end))

    def test_block_index_queries_stay_within_months_and_keep_partial_edges(self):
        cases = (
            ("2026-04-02T00:00Z", "2026-04-02T06:01Z", None),
            ("2026-03-31T23:00Z", "2026-04-01T06:01Z", "2026-04-01T00:00Z"),
            ("2024-02-29T23:00Z", "2024-03-01T01:00Z", "2024-03-01T00:00Z"),
            ("2025-12-31T23:00Z", "2026-01-01T01:00Z", "2026-01-01T00:00Z"),
            ("2026-04-01T00:00Z", "2026-04-01T01:00Z", "2026-04-01T00:00Z"),
            ("2026-03-31T23:58Z", "2026-03-31T23:59:30Z", "2026-04-01T00:00Z"),
            ("2026-03-31T23:58Z", "2026-03-31T23:59Z", None),
        )
        observed = []

        def request(sql):
            values = re.findall(r"block_time [<>]=? '([^']+)'", sql)
            left, right = (api.timestamp(value) for value in values)
            self.assertLess(left, right)
            self.assertEqual(
                left.strftime("%Y-%m"),
                (right - pd.Timedelta(nanoseconds=1)).strftime("%Y-%m"),
                "The block lookup would scan multiple monthly partitions",
            )
            observed.append((left, right))
            return [
                {
                    "first": int(left.timestamp()),
                    "last": int(right.timestamp()) - 1,
                    "blocks": 1,
                    "start": left,
                    "end": right - pd.Timedelta(seconds=1),
                }
            ]

        for first, last, boundary in cases:
            start, end = pd.Timestamp(first), pd.Timestamp(last)
            lower, upper = (
                start - pd.Timedelta(minutes=1),
                end + pd.Timedelta(minutes=1),
            )
            observed.clear()

            expected = (
                [(lower, upper)]
                if boundary is None
                else [(lower, pd.Timestamp(boundary)), (pd.Timestamp(boundary), upper)]
            )
            with (
                self.subTest(start=start, end=end),
                patch.object(api, "query", side_effect=request),
            ):
                self.assertEqual(
                    api.block_range(start, end),
                    (int(lower.timestamp()), int(upper.timestamp()) - 1),
                )
                self.assertEqual(observed, expected)

    def test_monthly_block_bounds_require_coverage_only_for_the_requested_window(self):
        march = {
            "first": 100,
            "last": 199,
            "blocks": 100,
            "start": "2026-03-31T22:59Z",
            "end": "2026-03-31T23:59:59Z",
        }
        april = {
            "first": 200,
            "last": 299,
            "blocks": 100,
            "start": "2026-04-01T00:00Z",
            "end": "2026-04-01T01:00:59Z",
        }
        cases = (
            ("2026-03-31T23:00Z", "2026-04-01T01:00Z", [[march], [april]], (100, 299)),
            ("2026-03-31T23:00Z", "2026-04-01T01:00Z", [[], [april]], None),
            (
                "2026-03-31T23:00Z",
                "2026-04-01T01:00Z",
                [[march], [{"blocks": 0}]],
                None,
            ),
            ("2026-04-01T00:00Z", "2026-04-01T01:00Z", [[], [april]], (200, 299)),
            ("2026-03-31T23:00Z", "2026-03-31T23:59:30Z", [[march], []], (100, 199)),
        )
        for first, last, responses, expected in cases:
            with (
                self.subTest(start=first, end=last, responses=responses),
                patch.object(api, "query", side_effect=responses),
            ):
                self.assertEqual(
                    api.block_range(pd.Timestamp(first), pd.Timestamp(last)), expected
                )

    def test_block_endpoint_is_validated_and_does_not_log_credentials(self):
        for endpoint in (
            "",
            "http://example.test/secret",
            "https://attacker.test/secret",
            "https://a.quiknode.pro/key?bad=1",
        ):
            with (
                patch.dict(os.environ, {"QUICKNODE_HYPERCORE_URL": endpoint}),
                self.assertRaises(ImproperlyConfigured),
            ):
                get_trade_blocks([123])
        with (
            patch.dict(
                os.environ,
                {"QUICKNODE_HYPERCORE_URL": "https://node.quiknode.pro/test-token/evm"},
            ),
            patch(
                "quant_tick.exchanges.quicknode.blocks.get_public_json",
                return_value=[
                    {
                        "id": 123,
                        "result": {"blocks": [{"block_number": 123, "events": []}]},
                    }
                ],
            ) as request,
        ):
            self.assertEqual(get_trade_blocks([123]), {123: []})
        self.assertTrue(request.call_args.kwargs["redact_url"])
        self.assertEqual(
            request.call_args.kwargs["json_body"][0]["params"],
            {"stream": "trades", "from": 123, "to": 123},
        )

    def test_block_batches_match_ids_and_reject_partial_or_duplicate_responses(self):
        numbers = list(range(100, 112))

        def response(*args, **kwargs):
            return [
                {
                    "id": item["id"],
                    "result": {"blocks": [{"block_number": item["id"], "events": []}]},
                }
                for item in reversed(kwargs["json_body"])
            ]

        with (
            patch.dict(
                os.environ,
                {"QUICKNODE_HYPERCORE_URL": "https://node.quiknode.pro/test-token/"},
            ),
            patch(
                "quant_tick.exchanges.quicknode.blocks.get_public_json",
                side_effect=response,
            ) as request,
        ):
            self.assertEqual(get_trade_blocks(numbers), {n: [] for n in numbers})
            self.assertEqual(
                [len(c.kwargs["json_body"]) for c in request.call_args_list], [10, 2]
            )
        good = {"id": 100, "result": {"blocks": [{"block_number": 100, "events": []}]}}
        for replies in ([], [good], [good, good], [{"id": 100, "error": {}}, good]):
            with (
                self.subTest(replies=replies),
                patch.dict(
                    os.environ,
                    {
                        "QUICKNODE_HYPERCORE_URL": "https://node.quiknode.pro/test-token/"
                    },
                ),
                patch(
                    "quant_tick.exchanges.quicknode.blocks.get_public_json",
                    return_value=replies,
                ),
                self.assertRaises(ValueError),
            ):
                get_trade_blocks([100, 101])
