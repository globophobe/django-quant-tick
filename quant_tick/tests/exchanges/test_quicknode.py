import json
import os
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
from unittest.mock import call, patch

import httpx2
import pandas as pd
from django.core.exceptions import ImproperlyConfigured
from django.test import SimpleTestCase

from quant_tick.exchanges.quicknode import api
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

    def collect(self):
        return list(iter_trades("BTC", self.start, self.end, bounds=(1, 2000000000)))

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

    def raw_events(self, rows):
        return [
            [
                "taker",
                {
                    "coin": "BTC",
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
        for failure in (None, "missing", "extra", "value", "duplicate"):
            events = self.raw_events(rows[::-1])
            if failure == "missing":
                events.pop()
            if failure in ("extra", "duplicate"):
                events.append(deepcopy(events[0]))
            if failure == "extra":
                events[-1][1]["tid"] = 999999
            if failure == "value":
                events[0][1]["sz"] = "99"
            with (
                self.subTest(failure=failure),
                patch("quant_tick.exchanges.quicknode.trades.PAGE_SIZE", 1),
                patch(
                    "quant_tick.exchanges.quicknode.trades.query",
                    side_effect=[[rows[0]], [rows[1]], []],
                ),
                patch(
                    "quant_tick.exchanges.quicknode.blocks.get_trade_blocks",
                    return_value={rows[0]["block_number"]: events},
                ) as raw,
            ):
                if failure:
                    with self.assertRaisesRegex(
                        ValueError, "QuickNode.*(disagree|omitted)"
                    ):
                        self.collect()
                else:
                    self.assertEqual(
                        [r["trade_id"] for r in self.collect()],
                        [r["trade_id"] for r in rows[::-1]],
                    )
                raw.assert_called_once_with([rows[0]["block_number"]])

    def test_invalid_sql_pages_fail_before_persistence(self):
        for change in (
            {"size": "NaN"},
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

    def test_separate_validation_query_preserves_exact_quantity(self):
        with (
            patch(
                "quant_tick.exchanges.quicknode.candles.block_range", return_value=None
            ),
            patch(
                "quant_tick.exchanges.quicknode.candles.query",
                return_value=[
                    {
                        "minute": self.start.isoformat(),
                        "notional": "5.000000000000000001",
                        "trades": 18,
                    }
                ],
            ),
        ):
            result = validation_candles("BTC", self.start, self.end)
        self.assertEqual(result.iloc[0].notional, Decimal("5.000000000000000001"))


class QuickNodeRequestTest(SimpleTestCase):
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
