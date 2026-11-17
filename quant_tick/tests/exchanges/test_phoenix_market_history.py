import json
import os
from copy import deepcopy
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from django.core.management import call_command
from django.test import SimpleTestCase, TestCase

from quant_tick.constants import Exchange, SymbolType
from quant_tick.exchanges.perpetual_stats import perpetual_stats
from quant_tick.exchanges.phoenix import events, positioning
from quant_tick.exchanges.phoenix.market_history import phoenix_market_history
from quant_tick.exchanges.quicknode import solana
from quant_tick.models import PerpetualStatsData, Symbol


def observation(timestamp, **overrides):
    return {
        "timestamp": timestamp.isoformat(),
        "open_interest": Decimal("24.4222"),
        "mark_price": Decimal("84587.0"),
        "spot_price": Decimal("84563.0"),
        "total_maker_fees": Decimal("15009.333919"),
        "total_taker_fees": Decimal("224907.999017"),
        "slot": 449845940,
    } | overrides


def response(points, **overrides):
    return {
        "market_id": 2,
        "symbol": "BTC",
        "timeframe": "1h",
        "stats": points,
    } | overrides


class PhoenixMarketHistoryTest(SimpleTestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, {"QUICKNODE_SOLANA_URL": ""}))
        self.start = datetime(2026, 9, 23, 23, tzinfo=UTC)
        self.api = self.enterContext(
            patch("quant_tick.exchanges.phoenix.market_history.get_phoenix_response")
        )

    def test_base_open_interest_and_mark_value_preserve_source_metadata(self):
        self.api.return_value = response([observation(self.start)])
        result = phoenix_market_history(
            "btc", self.start, self.start + timedelta(hours=1)
        )
        self.assertEqual(list(result.index), [self.start])
        row = result.iloc[0]
        self.assertEqual(row.open_interest, Decimal("24.4222"))
        self.assertEqual(row.open_interest_value, Decimal("2065800.6314"))
        self.assertEqual(row.open_interest_unit, "base_asset")
        self.assertEqual(row.mark_price, Decimal(84587))
        self.assertEqual(row.spot_price, Decimal(84563))
        self.assertEqual(row.slot, 449845940)
        self.assertEqual(row.market_history_timestamp_convention, "interval_start")
        self.assertEqual(row.single_open_interest, row.open_interest)
        self.assertEqual(row.cumulative_taker_fees, Decimal("224907.999017"))
        self.assertEqual(row.cumulative_maker_fees, Decimal("15009.333919"))
        self.assertGreater(row.mark_spot_basis_bps, 0)
        self.assertNotIn("long_short_account_ratio", result.columns)
        self.api.assert_called_once_with(
            "/v1/market/BTC/stats",
            {
                "start_time": self.start.isoformat(),
                "end_time": (
                    self.start + timedelta(hours=1, microseconds=-1)
                ).isoformat(),
                "timeframe": "1h",
                "limit": 1000,
            },
        )

    def test_aligned_chunks_preserve_missing_hours_without_filling(self):
        timestamps = [self.start + timedelta(hours=hour) for hour in (1, 2, 4)]
        self.api.side_effect = [
            response([observation(timestamp) for timestamp in timestamps[:2]]),
            response([observation(timestamps[2], open_interest=0)]),
        ]
        with patch(
            "quant_tick.exchanges.phoenix.market_history.MARKET_HISTORY_MAX_RESULTS", 3
        ):
            result = phoenix_market_history(
                "BTC",
                self.start + timedelta(minutes=1),
                self.start + timedelta(hours=5, minutes=20),
            )
        self.assertEqual(list(result.index), timestamps)
        self.assertEqual(result.iloc[-1].open_interest, Decimal(0))
        self.assertEqual(result.iloc[-1].open_interest_value, Decimal(0))
        params = [item.args[1] for item in self.api.call_args_list]
        self.assertEqual(params[0]["start_time"], timestamps[0].isoformat())
        self.assertEqual(
            datetime.fromisoformat(params[0]["end_time"]) + timedelta(microseconds=1),
            datetime.fromisoformat(params[1]["start_time"]),
        )
        self.assertEqual(
            params[1]["end_time"],
            (self.start + timedelta(hours=5, microseconds=-1)).isoformat(),
        )

    def test_unfinished_hour_does_not_fetch_or_create_an_observation(self):
        result = phoenix_market_history(
            "BTC", self.start, self.start + timedelta(minutes=59)
        )
        self.assertTrue(result.empty)
        self.assertIn("open_interest", result.columns)
        self.api.assert_not_called()

    def test_rejects_response_mismatches_and_invalid_points(self):
        end = self.start + timedelta(hours=2)
        cases = [
            (response([], symbol="SOL"), "market or timeframe"),
            (response([], timeframe="1d"), "market or timeframe"),
            (response([observation(end)]), "outside"),
            (
                response([observation(self.start + timedelta(seconds=1))]),
                "hourly buckets",
            ),
            (
                response([observation(self.start), observation(self.start)]),
                "hourly buckets",
            ),
            (
                response(
                    [
                        observation(self.start + timedelta(hours=1)),
                        observation(self.start),
                    ]
                ),
                "hourly buckets",
            ),
            (
                response([observation(self.start, open_interest=-1)]),
                "invalid quantities",
            ),
            (
                response([observation(self.start, mark_price="NaN")]),
                "invalid quantities",
            ),
        ]
        for payload, message in cases:
            with self.subTest(payload=payload):
                self.api.return_value = payload
                with self.assertRaisesRegex(ValueError, message):
                    phoenix_market_history("BTC", self.start, end)

    def test_response_limit_cannot_silently_truncate_history(self):
        self.api.return_value = response([observation(self.start)] * 2)
        with (
            patch(
                "quant_tick.exchanges.phoenix.market_history.MARKET_HISTORY_MAX_RESULTS",
                2,
            ),
            self.assertRaisesRegex(ValueError, "response limit"),
        ):
            phoenix_market_history("BTC", self.start, self.start + timedelta(hours=1))


class PhoenixMarketHistoryCollectionTest(TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ, {"QUICKNODE_SOLANA_URL": ""}))

    def test_empty_paid_hour_is_a_gap_until_older_history_is_exhausted(self):
        symbol = Symbol.objects.create(
            exchange=Exchange.PHOENIX,
            api_symbol="BTC",
            symbol_type=SymbolType.PERPETUAL,
        )
        end = datetime(2026, 11, 17, 12, tzinfo=UTC)
        point = observation(end - timedelta(hours=2))
        with (
            patch.dict(os.environ, {"QUICKNODE_SOLANA_URL": "configured"}),
            patch(
                "quant_tick.exchanges.phoenix.market_history.get_phoenix_response",
                side_effect=[response([]), response([point]), response([])],
            ) as request,
            patch(
                "quant_tick.exchanges.phoenix.market_history.has_trades",
                side_effect=[True, False],
            ),
            patch(
                "quant_tick.exchanges.phoenix.market_history.phoenix_positioning",
                return_value={"trade_position_flow_validated": True},
            ),
        ):
            perpetual_stats(symbol, end - timedelta(days=365), end)
        self.assertEqual(request.call_count, 3)
        self.assertEqual(
            PerpetualStatsData.objects.get(symbol=symbol).timestamp,
            end - timedelta(hours=2),
        )

    def test_paid_hours_checkpoint_before_a_later_provider_failure(self):
        symbol = Symbol.objects.create(
            exchange=Exchange.PHOENIX,
            api_symbol="BTC",
            symbol_type=SymbolType.PERPETUAL,
        )
        start = datetime(2026, 11, 17, 9, tzinfo=UTC)
        end = start + timedelta(hours=2)
        point = observation(start + timedelta(hours=1))
        with (
            patch.dict(os.environ, {"QUICKNODE_SOLANA_URL": "configured"}),
            patch(
                "quant_tick.exchanges.phoenix.market_history.get_phoenix_response",
                side_effect=[response([point]), RuntimeError("provider unavailable")],
            ),
            patch(
                "quant_tick.exchanges.phoenix.market_history.phoenix_positioning",
                return_value={
                    "trade_position_flow_validated": True,
                    "long_open_size": Decimal("2.5"),
                },
            ),
            self.assertRaisesRegex(RuntimeError, "provider unavailable"),
        ):
            perpetual_stats(symbol, start, end)
        saved = PerpetualStatsData.objects.get(symbol=symbol).to_row()
        self.assertEqual(saved["timestamp"], start + timedelta(hours=1))
        self.assertEqual(Decimal(saved["long_open_size"]), Decimal("2.5"))
        self.assertTrue(saved["trade_position_flow_validated"])

    def test_management_backfill_stops_on_empty_history_and_repairs_gaps(self):
        symbol = Symbol.objects.create(
            exchange=Exchange.PHOENIX,
            api_symbol="BTC",
            symbol_type=SymbolType.PERPETUAL,
        )
        start = datetime(2026, 9, 23, tzinfo=UTC)
        end = start + timedelta(hours=3, minutes=15)
        points = [observation(start + timedelta(hours=hour)) for hour in (0, 2)]

        def fetch(path, params):
            left = datetime.fromisoformat(params["start_time"])
            right = datetime.fromisoformat(params["end_time"])
            return response(
                [
                    point
                    for point in points
                    if left <= datetime.fromisoformat(point["timestamp"]) <= right
                ]
            )

        with patch(
            "quant_tick.exchanges.phoenix.market_history.get_phoenix_response",
            side_effect=fetch,
        ) as api:
            call_command(
                "perpetual_stats",
                "--exchange",
                "phoenix",
                "--api-symbol",
                "BTC",
                "--date-from",
                (start - timedelta(days=270)).date().isoformat(),
                "--date-to",
                "2026-09-23",
                "--time-to",
                "03:15",
                stdout=StringIO(),
            )

        self.assertEqual(api.call_count, 6)
        self.assertEqual(
            min(
                datetime.fromisoformat(item.args[1]["start_time"])
                for item in api.call_args_list
            ),
            start + timedelta(hours=3) - timedelta(days=180),
        )
        rows = PerpetualStatsData.objects.filter(symbol=symbol)
        self.assertEqual(rows.count(), 2)
        self.assertEqual(rows.complete_for_exchange(Exchange.PHOENIX).count(), 2)
        first = rows.first()
        self.assertEqual(first.frequency, 60)
        self.assertEqual(first.open_interest, Decimal("24.4222"))
        self.assertEqual(first.open_interest_value, Decimal("2065800.6314"))
        self.assertEqual(first.json_data["slot"], 449845940)
        self.assertEqual(
            first.json_data["market_history_timestamp_convention"], "interval_start"
        )
        self.assertIsNone(first.long_short_account_ratio)

        missing = start + timedelta(hours=1)
        with patch(
            "quant_tick.exchanges.phoenix.market_history.get_phoenix_response",
            return_value=response([observation(missing)]),
        ) as api:
            perpetual_stats(symbol, start, end)
        self.assertEqual(api.call_args.args[1]["start_time"], missing.isoformat())
        self.assertEqual(
            api.call_args.args[1]["end_time"],
            (missing + timedelta(hours=1, microseconds=-1)).isoformat(),
        )
        self.assertEqual(rows.count(), 3)

        with patch(
            "quant_tick.exchanges.phoenix.market_history.get_phoenix_response"
        ) as api:
            perpetual_stats(symbol, start, end)
        api.assert_not_called()


class PhoenixPositioningTest(SimpleTestCase):
    def setUp(self):
        path = Path(__file__).parent.parent / "fixtures/phoenix/positioning.json"
        self.fixture = json.loads(path.read_text())
        self.start = datetime(2026, 11, 17, 9, tzinfo=UTC)
        self.end = self.start + timedelta(hours=1)

    def collect(self, signatures, data=None):
        data = data or self.fixture["transactions"]
        rows = [
            row for sig in reversed(signatures) for row in data[sig]["native_fills"]
        ]
        candle = pd.DataFrame(
            [
                {
                    "timestamp": self.start,
                    "notional": sum(abs(Decimal(row["baseQty"])) for row in rows),
                    "volume": sum(abs(Decimal(row["quoteQty"])) for row in rows),
                    "trades": len(rows),
                }
            ]
        ).set_index("timestamp")
        with (
            patch.object(positioning, "get_trades", return_value=(rows, True, None)),
            patch.object(
                positioning,
                "transaction",
                side_effect=lambda sig: data[sig]["transaction"],
            ),
            patch.object(positioning, "phoenix_candles", return_value=candle),
        ):
            return positioning.phoenix_positioning("BTC", self.start, self.end)

    def test_real_transactions_match_the_official_decoder_and_native_fills(self):
        for kind, signature in self.fixture["cases"].items():
            case = self.fixture["transactions"][signature]
            with self.subTest(kind=kind):
                decoded = events.transaction_events(case["transaction"], signature)
                normalized = [
                    [
                        {
                            key: list(value) if isinstance(value, bytes) else value
                            for key, value in item.items()
                        }
                        for item in group
                    ]
                    for group in decoded
                ]
                self.assertEqual(normalized, case["events"])
                result = self.collect([signature])
                for field, value in case["stats"].items():
                    self.assertEqual(Decimal(result[field]), Decimal(value), field)
                self.assertTrue(result["trade_position_flow_validated"])
                self.assertNotIn("long_short_account_ratio", result)
                self.assertNotIn("open_interest_high", result)

    def test_missing_or_duplicate_executions_cannot_become_valid_flows(self):
        signatures = self.fixture["sequence"]
        self.assertGreater(self.collect(signatures)["trades"], 0)
        for broken in ([signatures[0], signatures[2]], [*signatures, signatures[-1]]):
            with (
                self.subTest(signatures=broken),
                self.assertRaisesRegex(ValueError, "gap|duplicate|match"),
            ):
                self.collect(broken)
        data = deepcopy(self.fixture["transactions"])
        data[signatures[0]]["native_fills"][0]["timestamp"] = self.end.isoformat()
        with self.assertRaisesRegex(ValueError, "match"):
            self.collect(signatures, data)

        native = [
            row
            for sig in signatures
            for row in self.fixture["transactions"][sig]["native_fills"]
        ]
        candle = pd.DataFrame(
            [{"timestamp": self.start, "trades": len(native)}]
        ).set_index("timestamp")
        with (
            patch.object(
                positioning, "get_trades", return_value=(native[1:], True, None)
            ),
            patch.object(positioning, "phoenix_candles", return_value=candle),
            patch.object(positioning, "transaction") as request,
            self.assertRaisesRegex(ValueError, "count"),
        ):
            positioning.phoenix_positioning("BTC", self.start, self.end)
        request.assert_not_called()

    def test_bad_transaction_metadata_or_log_framing_is_rejected(self):
        signature = next(iter(self.fixture["cases"].values()))
        original = self.fixture["transactions"][signature]["transaction"]
        variants = []
        tx = deepcopy(original)
        tx["meta"]["err"] = {"InstructionError": [0, "failure"]}
        variants.append(tx)
        tx = deepcopy(original)
        tx["meta"]["innerInstructions"] = None
        variants.append(tx)
        tx = deepcopy(original)
        tx["transaction"]["signatures"][0] = "wrong"
        variants.append(tx)
        tx = deepcopy(original)
        tx["meta"].pop("loadedAddresses", None)
        tx["transaction"]["message"]["addressTableLookups"] = [{"missing": True}]
        variants.append(tx)
        tx = deepcopy(original)
        tx["meta"]["innerInstructions"][0]["instructions"][0]["stackHeight"] = None
        variants.append(tx)
        for tx in variants:
            with self.subTest(tx=variants.index(tx)), self.assertRaises(ValueError):
                events.transaction_events(tx, signature)
        for data in (b"", b"\xff", b"\0" + b"\0" * 15):
            with self.subTest(data=data), self.assertRaises(ValueError):
                events.event(data)
        with self.assertRaisesRegex(ValueError, "framing"):
            events.decode_logs([(events.LOG, b"\0\0\0\0\x01\0\0\0\0" + b"\0" * 16)])

    def test_rpc_errors_propagate_and_endpoint_is_redacted(self):
        signature = next(iter(self.fixture["cases"].values()))
        tx = self.fixture["transactions"][signature]["transaction"]
        solana.transaction.cache_clear()
        self.addCleanup(solana.transaction.cache_clear)
        with (
            patch.dict(
                os.environ,
                {
                    "QUICKNODE_SOLANA_URL": "https://sample.solana-mainnet.quiknode.pro/token/",
                    "QUICKNODE_SOLANA_RATE_LIMIT": "2",
                },
            ),
            patch.object(
                solana, "get_public_json", return_value={"id": signature, "result": tx}
            ) as request,
        ):
            self.assertEqual(solana.transaction(signature), tx)
            self.assertTrue(request.call_args.kwargs["redact_url"])
            self.assertEqual(request.call_args.kwargs["reset_seconds"], 0.5)
            solana.transaction.cache_clear()
            request.return_value = {"id": signature, "result": None}
            with self.assertRaisesRegex(ValueError, "omitted"):
                solana.transaction(signature)
