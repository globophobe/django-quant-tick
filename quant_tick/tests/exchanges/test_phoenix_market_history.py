import base64
import gzip
import json
import os
import struct
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
from quant_tick.exchanges.phoenix import (
    accounts,
    events,
    history,
    positioning,
    snapshots,
)
from quant_tick.exchanges.phoenix.market_history import phoenix_market_history
from quant_tick.exchanges.quicknode import solana
from quant_tick.models import Checkpoint, PerpetualStatsData, Symbol


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


def snapshot_fixture():
    path = Path(__file__).parents[1] / "fixtures/phoenix/accounts.json.gz"
    return json.loads(gzip.decompress(path.read_bytes()))


def replay_fixture():
    path = Path(__file__).parents[1] / "fixtures/phoenix/reconstruction.json.gz"
    return json.loads(gzip.decompress(path.read_bytes()))


def encode58(data):
    number, result = int.from_bytes(data, "big"), ""
    while number:
        number, digit = divmod(number, 58)
        result = events.ALPHABET[digit] + result
    return "1" * (len(data) - len(data.lstrip(b"\0"))) + result


def history_transaction(slot, timestamp, changes=(), transfer=None, adl=None):
    items = [struct.pack("<BQQ", 0, int(timestamp.timestamp()), slot)]
    for account, before, after in changes:
        items.append(
            struct.pack(
                "<B32sI16sqqqqqq", 37, account, 2, b"BTC", 0, 0, before, after, 0, 0
            )
        )
    if transfer:
        sender, receiver, quantity = transfer
        items.append(
            struct.pack("<B32s32sQqqHqq", 31, sender, receiver, 2, quantity, 0, 0, 0, 0)
        )
    if adl:
        long, short, profitable, quantity = adl
        items.append(
            struct.pack(
                "<B32s32s32s32sQqqqqq",
                33,
                b"a" * 32,
                short,
                long,
                profitable,
                2,
                quantity,
                0,
                0,
                0,
                0,
            )
        )
    logs = [
        (
            events.LENGTHS,
            struct.pack("<II", 0, len(items))
            + struct.pack("<" + "H" * len(items), *map(len, items)),
        ),
        (events.LOG, struct.pack("<II", 0, len(items)) + b"".join(items)),
    ]
    return {
        "slot": slot,
        "transactionIndex": 0,
        "blockTime": int(timestamp.timestamp()) + 2,
        "transaction": {
            "signatures": [f"signature-{slot}"],
            "message": {
                "accountKeys": [events.PROGRAM, "buffer"],
                "instructions": [
                    {
                        "programIdIndex": 0,
                        "accounts": [1],
                        "data": encode58(b"\x01" * 8),
                    }
                ],
            },
        },
        "meta": {
            "err": None,
            "innerInstructions": [
                {
                    "index": 0,
                    "instructions": [
                        {
                            "programIdIndex": 0,
                            "accounts": [],
                            "stackHeight": 2,
                            "data": encode58(struct.pack("<Q", tag) + payload),
                        }
                        for tag, payload in logs
                    ],
                }
            ],
        },
    }


class PhoenixSnapshotTest(SimpleTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.fixture = snapshot_fixture()

    def test_real_accounts_match_official_decoder_and_native_open_interest(self):
        result = accounts.decode_snapshot(self.fixture["result"], "BTC")
        expected = self.fixture["expected"]
        self.assertEqual(
            result["positions"],
            {
                events.decode58(key): value["base_lots"]
                for key, value in expected["positions"].items()
            },
        )
        self.assertEqual(result["open_interest_lots"], expected["native_oi_lots"])
        self.assertEqual(result["price"], Decimal(64328))
        values = accounts.cohort_values(result)
        for side in ("long", "short"):
            self.assertEqual(values[side + "_accounts"], expected[side + "_accounts"])
            self.assertEqual(
                sum(tier[side + "Sz"] for tier in values["cohorts"]),
                values["cohort_single_open_interest"],
            )
        self.assertEqual(
            values["long_account_ratio"] + values["short_account_ratio"], 1
        )
        self.assertEqual(values["cohort_open_interest"], Decimal("61.6772"))

    def test_missing_corrupt_or_unhandled_accounts_cannot_produce_partial_cohorts(self):
        original = self.fixture["result"]
        cold = next(
            key
            for key, value in self.fixture["expected"]["positions"].items()
            if value["source"] == "cold"
        )
        variants = []
        missing = deepcopy(original)
        missing["value"] = [r for r in missing["value"] if r["pubkey"] != cold]
        variants.append(missing)
        duplicate = deepcopy(original)
        duplicate["value"].append(duplicate["value"][0])
        variants.append(duplicate)
        wrong_owner = deepcopy(original)
        wrong_owner["value"][0]["account"]["owner"] = "unexpected"
        variants.append(wrong_owner)
        for kind, offset, fmt, value in (
            ("global_trader_index", 52, "H", 2),
            ("global_trader_index", 80, "I", 0),
            ("trader", 96, "I", 1),
            ("perp_asset_map", 40, "Q", 512),
        ):
            changed = deepcopy(original)
            for row in changed["value"]:
                data = bytearray(accounts.account_bytes(row["account"]))
                if accounts.TAGS.get(bytes(data[:8])) == kind and (
                    kind != "trader" or row["pubkey"] == cold
                ):
                    struct.pack_into("<" + fmt, data, offset, value)
                    row["account"]["data"] = [base64.b64encode(data).decode(), "base64"]
                    break
            variants.append(changed)
        for index, invalid in enumerate(variants):
            with self.subTest(case=index), self.assertRaises(ValueError):
                accounts.decode_snapshot(invalid, "BTC")

    def test_empty_market_keeps_zero_counts_and_undefined_ratios(self):
        values = accounts.cohort_values(
            {
                "positions": {},
                "base_lot_decimals": 4,
                "price": Decimal(60000),
                "open_interest_lots": 0,
                "slot": 100,
            }
        )
        self.assertEqual(values["long_accounts"], 0)
        self.assertEqual(values["short_accounts"], 0)
        self.assertEqual(values["cohort_single_open_interest"], 0)
        self.assertIsNone(values["long_short_account_ratio"])

    def test_fetch_requires_fresh_finalized_state_and_retains_observation_time(self):
        now = datetime(2026, 11, 17, 11, 20, tzinfo=UTC)
        for age in (30, 301, -1):
            block_time = now - timedelta(seconds=age)
            with (
                self.subTest(age=age),
                patch.object(snapshots, "get_current_time", return_value=now),
                patch.object(
                    snapshots,
                    "rpc",
                    side_effect=[
                        self.fixture["result"],
                        int(block_time.timestamp()),
                    ],
                ) as request,
            ):
                if age != 30:
                    with self.assertRaisesRegex(ValueError, "stale|future"):
                        snapshots.fetch_snapshot("BTC")
                    continue
                values = snapshots.fetch_snapshot("BTC")
                self.assertEqual(values["cohort_timestamp"], block_time)
                self.assertEqual(values["cohort_available_at"], now)
                self.assertEqual(
                    request.call_args_list[0].args[1][1],
                    {
                        "encoding": "base64+zstd",
                        "commitment": "finalized",
                        "withContext": True,
                    },
                )
                self.assertEqual(
                    request.call_args.args, ("getBlockTime", [values["cohort_slot"]])
                )


class PhoenixSnapshotCollectionTest(TestCase):
    def setUp(self):
        self.symbol = Symbol.objects.create(
            exchange=Exchange.PHOENIX,
            api_symbol="BTC",
            symbol_type=SymbolType.PERPETUAL,
        )
        self.hour = datetime(2026, 11, 17, 11, tzinfo=UTC)
        self.enterContext(
            patch.dict(os.environ, {"QUICKNODE_SOLANA_URL": "configured"})
        )
        self.clock = self.enterContext(
            patch.object(snapshots, "get_current_time", return_value=self.hour)
        )
        self.fetch = self.enterContext(
            patch.object(
                snapshots,
                "fetch_snapshot",
                side_effect=lambda symbol: {
                    "cohort_available_at": self.hour + timedelta(minutes=2),
                    "cohort_timestamp": self.hour + timedelta(minutes=1),
                    "cohort_snapshot_validated": True,
                    "long_account_ratio": Decimal("0.7"),
                    "short_account_ratio": Decimal("0.3"),
                    "long_short_account_ratio": Decimal(7) / 3,
                    "long_accounts": 7,
                    "short_accounts": 3,
                },
            )
        )

    def test_hourly_capture_is_idempotent_and_native_history_still_fills_the_row(self):
        snapshots.collect_snapshot(self.symbol)
        snapshots.collect_snapshot(self.symbol)
        self.fetch.assert_called_once_with("BTC")
        saved = PerpetualStatsData.objects.get(symbol=self.symbol)
        self.assertEqual(saved.timestamp, self.hour)
        self.assertIsNone(saved.open_interest)
        self.assertFalse(
            PerpetualStatsData.objects.complete_for_exchange(Exchange.PHOENIX).exists()
        )
        with (
            patch(
                "quant_tick.exchanges.phoenix.market_history.get_phoenix_response",
                return_value=response([observation(self.hour)]),
            ) as request,
            patch(
                "quant_tick.exchanges.phoenix.history.PhoenixPositionHistory.enrich",
                return_value={"trade_position_flow_validated": True},
            ),
        ):
            perpetual_stats(self.symbol, self.hour, self.hour + timedelta(hours=1))
        request.assert_called_once()
        saved = PerpetualStatsData.objects.get(symbol=self.symbol)
        self.assertEqual(saved.open_interest, Decimal("24.4222"))
        self.assertEqual(saved.long_account_ratio, Decimal("0.7"))
        self.assertTrue(saved.json_data["cohort_snapshot_validated"])
        self.assertTrue(saved.json_data["trade_position_flow_validated"])

    def test_disabled_provider_failure_or_lost_lease_does_not_write(self):
        with patch.dict(os.environ, {"QUICKNODE_SOLANA_URL": ""}):
            snapshots.collect_snapshot(self.symbol)
        self.fetch.assert_not_called()
        self.fetch.side_effect = ValueError("invalid snapshot")
        with self.assertRaisesRegex(ValueError, "invalid snapshot"):
            snapshots.collect_snapshot(self.symbol)
        self.assertFalse(PerpetualStatsData.objects.exists())
        self.fetch.side_effect = None
        self.fetch.return_value = {"cohort_available_at": self.hour}

        def lost():
            raise RuntimeError("lease lost")

        with self.assertRaisesRegex(RuntimeError, "lease lost"):
            snapshots.collect_snapshot(self.symbol, assert_lease_owned=lost)
        self.assertFalse(PerpetualStatsData.objects.exists())

    def test_hour_boundary_uses_receipt_hour_and_preserves_existing_native_values(self):
        self.clock.return_value = self.hour - timedelta(seconds=1)
        PerpetualStatsData.objects.create(
            symbol=self.symbol,
            timestamp=self.hour,
            frequency=60,
            open_interest=Decimal(25),
            open_interest_value=Decimal(1600000),
            json_data={"trade_position_flow_validated": True},
        )
        snapshots.collect_snapshot(self.symbol)
        saved = PerpetualStatsData.objects.get()
        self.assertEqual(saved.timestamp, self.hour)
        self.assertEqual(saved.open_interest, Decimal(25))
        self.assertTrue(saved.json_data["trade_position_flow_validated"])
        self.assertTrue(saved.json_data["cohort_snapshot_validated"])


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
                "quant_tick.exchanges.phoenix.history.PhoenixPositionHistory.enrich",
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
                "quant_tick.exchanges.phoenix.history.PhoenixPositionHistory.enrich",
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


class PhoenixReversePositionsTest(SimpleTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.fixture = replay_fixture()

    def replay(self, transactions, state=None):
        replay = history.ReversePositions(state or self.fixture["end_state"])
        for tx in transactions:
            groups, clock = history.decode_transaction(
                tx, replay.state["history_address"]
            )
            replay.undo(tx, groups, clock)
        return replay

    def test_real_reverse_replay_recovers_every_independent_starting_position(self):
        transactions = self.fixture["transactions"]
        replay = self.replay(transactions)
        self.assertEqual(replay.dump()["positions"], self.fixture["start_positions"])
        # Resume in the middle of a slot using a JSON round-trip, as the database does.
        split = next(
            i
            for i in range(1, len(transactions))
            if transactions[i]["slot"] == transactions[i - 1]["slot"]
        )
        state = json.loads(json.dumps(self.replay(transactions[:split]).dump()))
        resumed = self.replay(transactions[split:], state)
        self.assertEqual(resumed.dump(), replay.dump())

    def test_missing_duplicate_reordered_or_corrupt_position_evidence_is_rejected(self):
        original = self.fixture["transactions"]
        index = next(
            i
            for i, tx in enumerate(original)
            if any(
                item["kind"] == "TradeSummary"
                for group in history.decode_transaction(
                    tx, self.fixture["end_state"]["history_address"]
                )[0]
                for item in group
            )
        )
        variants = [
            original[:index] + original[index + 1 :],
            original[:index] + original[index : index + 1] + original[index:],
            list(reversed(original)),
        ]
        for broken in variants:
            with (
                self.subTest(case=variants.index(broken)),
                self.assertRaises(ValueError),
            ):
                self.replay(broken)
        state = deepcopy(self.fixture["end_state"])
        accounts_ = list(state["positions"])
        state["positions"][accounts_[0]] += 1
        state["positions"][accounts_[1]] -= 1
        with self.assertRaises(ValueError):
            self.replay(original, state)

    def test_real_liquidation_transfer_and_forced_closes_restore_both_accounts(self):
        tx = self.fixture["transfer_transaction"]
        groups = events.transaction_events(tx, tx["transaction"]["signatures"][0])
        transfer = next(
            item
            for group in groups
            for item in group
            if item["kind"] == "LiquidationTransfer"
        )
        sender, receiver = transfer["liquidatee"], transfer["liquidator"]
        for group in groups:
            history.validate_transfers(group)
        state = {
            **self.fixture["end_state"],
            "asset_id": transfer["asset_id"],
            "positions": {
                sender.hex(): 8,
                receiver.hex(): 40,
                (b"s" * 32).hex(): -48,
            },
        }
        replay = history.ReversePositions(state)
        replay.undo({**tx, "transactionIndex": 0}, groups, 1)
        self.assertEqual(replay.positions, {sender: 48, b"s" * 32: -48})
        forced = positioning.nontrade_flows(groups, transfer["asset_id"], 1)
        self.assertEqual(forced["backstop_events"], 1)
        self.assertEqual(forced["backstop_long_size"], Decimal(4))
        self.assertEqual(forced["adl_size"], 0)
        self.assertEqual(forced["long_account_delta"], 1)
        self.assertNotIn("backstop_value", forced)
        self.assertEqual(positioning.nontrade_flows(groups, 2, 4)["backstop_events"], 0)
        for kind in ("CloseMatchedPositions", "ShutdownClosePositions"):
            short = b"s" * 32
            group = [
                {
                    "kind": "PnL",
                    "asset_id": 0,
                    "trader": sender,
                    "base_lots_before": 10,
                    "base_lots_after": 6,
                },
                {
                    "kind": "PnL",
                    "asset_id": 0,
                    "trader": short,
                    "base_lots_before": -10,
                    "base_lots_after": -6,
                },
                {
                    "kind": kind,
                    "asset_id": 0,
                    "base_lots_closed": 4,
                    "closed_long": sender,
                    "closed_short": short,
                    "in_profit_account": short,
                    "long_trader": sender,
                    "short_trader": short,
                    "trade_sequence_number": 9,
                },
            ]
            with self.subTest(kind=kind):
                replay = history.ReversePositions(
                    {**state, "positions": {sender.hex(): 6, short.hex(): -6}}
                )
                replay.undo({**tx, "transactionIndex": 0}, [group], 1)
                self.assertEqual(replay.positions, {sender: 10, short: -10})
                flows = positioning.nontrade_flows([group], state["asset_id"], 1)
                field = "adl" if kind == "CloseMatchedPositions" else "shutdown"
                self.assertEqual(flows[field + "_events"], 1)
                self.assertEqual(flows[field + "_size"], Decimal("0.4"))
                self.assertEqual(flows["long_close_size"], Decimal("0.4"))
                self.assertEqual(flows["short_close_size"], Decimal("0.4"))
                self.assertEqual(flows["backstop_size"], 0)
                self.assertNotIn("taker_buy_size", flows)
                group[-1]["base_lots_closed"] = 5
                with self.assertRaisesRegex(ValueError, "transfer/close"):
                    history.validate_transfers(group)

        tx = self.fixture["adl_transaction"]
        groups = events.transaction_events(tx, tx["transaction"]["signatures"][0])
        adl = next(
            item
            for group in groups
            for item in group
            if item["kind"] == "CloseMatchedPositions"
        )
        for group in groups:
            history.validate_transfers(group)
        flows = positioning.nontrade_flows(
            groups, adl["asset_id"], self.fixture["adl_base_lot_decimals"]
        )
        self.assertEqual(flows["adl_events"], 1)
        self.assertEqual(flows["adl_short_size"], Decimal("0.110"))
        self.assertEqual(flows["adl_long_size"], 0)
        self.assertEqual(flows["long_account_delta"], -1)
        self.assertEqual(flows["short_account_delta"], -1)
        self.assertEqual(flows["shutdown_size"], 0)
        oi = next(
            item["open_interest"]
            for group in groups
            for item in group
            if item["kind"] == "MarketSummary"
        )
        replay = history.ReversePositions(
            {
                **state,
                "asset_id": adl["asset_id"],
                "base_lot_decimals": self.fixture["adl_base_lot_decimals"],
                "positions": {(b"x" * 32).hex(): oi, (b"y" * 32).hex(): -oi},
            }
        )
        replay.undo(tx, groups, None)
        self.assertEqual(replay.positions[adl["closed_long"]], 110)
        self.assertEqual(replay.positions[adl["closed_short"]], -110)

    def test_bulk_pages_preserve_order_and_resume_with_an_exclusive_signature(self):
        transactions = self.fixture["transactions"]
        selected = transactions[:3]
        token = f"{selected[0]['slot']}:{selected[0]['transactionIndex']}"
        with patch.object(
            solana,
            "rpc",
            side_effect=[
                {"data": selected[:1], "paginationToken": token},
                {"data": selected[1:], "paginationToken": None},
            ],
        ) as request:
            self.assertEqual(
                list(solana.address_transactions("buffer", selected[0]["slot"])),
                selected,
            )
            self.assertEqual(request.call_args.args[1][1]["paginationToken"], token)
        with patch.object(
            solana, "rpc", return_value={"data": selected[1:], "paginationToken": None}
        ) as request:
            list(
                solana.address_transactions(
                    "buffer",
                    selected[0]["slot"],
                    before="saved-signature",
                    transaction_index=selected[0]["transactionIndex"],
                )
            )
            self.assertEqual(
                request.call_args.args[1][1]["filters"]["signature"],
                {"lt": "saved-signature"},
            )
        for page in (
            {"data": list(reversed(selected))},
            {"data": [selected[0]] * 2},
            {"data": [], "paginationToken": token},
            {"data": selected[:1], "paginationToken": "bad"},
        ):
            with (
                patch.object(solana, "rpc", return_value=page),
                self.assertRaises(ValueError),
            ):
                list(solana.address_transactions("buffer", selected[0]["slot"]))

    def test_native_ledger_boundary_skips_newer_slots_and_failed_transactions(self):
        newer = [
            {
                "signature": f"signature-{slot}",
                "slot": slot,
                "err": None,
                "confirmationStatus": "finalized",
            }
            for slot in range(2140, 1140, -1)
        ]
        failed = {**newer[-1], "slot": 140, "signature": "failed", "err": {"code": 1}}
        head = {**failed, "signature": "head", "err": None}
        with patch.object(
            solana, "rpc", side_effect=[newer, [failed, head]]
        ) as request:
            self.assertEqual(
                solana.latest_address_signature("buffer", 140),
                {"signature": "head", "slot": 140},
            )
            self.assertEqual(request.call_count, 2)
            self.assertEqual(
                request.call_args.args,
                (
                    "getSignaturesForAddress",
                    [
                        "buffer",
                        {
                            "commitment": "finalized",
                            "minContextSlot": 140,
                            "limit": 1000,
                            "before": newer[-1]["signature"],
                        },
                    ],
                ),
            )
        for page in (
            [],
            [failed],
            [failed, failed],
            [failed, newer[0]],
            [{**head, "confirmationStatus": "confirmed"}],
            [{key: value for key, value in head.items() if key != "err"}],
        ):
            with (
                self.subTest(page=page),
                patch.object(solana, "rpc", return_value=page),
                self.assertRaises(ValueError),
            ):
                solana.latest_address_signature("buffer", 140)


class PhoenixHistoryCollectionTest(TestCase):
    def setUp(self):
        self.symbol = Symbol.objects.create(
            exchange=Exchange.PHOENIX,
            api_symbol="BTC",
            symbol_type=SymbolType.PERPETUAL,
        )
        self.start = datetime(2026, 11, 17, 9, tzinfo=UTC)
        self.end = self.start + timedelta(hours=2)
        self.accounts = [bytes([i]) * 32 for i in range(1, 5)]
        a, b, c, d = self.accounts
        self.txs = [
            history_transaction(130, self.end + timedelta(minutes=5)),
            history_transaction(
                120,
                self.start + timedelta(hours=1, minutes=30),
                [(a, 10, 6), (b, 0, 4)],
                (a, b, 4),
            ),
            history_transaction(
                110,
                self.start + timedelta(minutes=30),
                [(d, 10, 0), (a, 0, 10)],
                (d, a, 10),
            ),
            history_transaction(100, self.start - timedelta(seconds=1)),
        ]
        self.enterContext(
            patch.dict(os.environ, {"QUICKNODE_SOLANA_URL": "configured"})
        )
        self.snapshot = self.enterContext(
            patch.object(
                history,
                "fetch_position_snapshot",
                return_value=(
                    {
                        "slot": 140,
                        "asset_id": 2,
                        "base_lot_decimals": 4,
                        "positions": {a: 6, b: 4, c: -10},
                        "history_address": "buffer",
                    },
                    self.end + timedelta(minutes=10),
                    self.end + timedelta(minutes=10),
                ),
            )
        )
        self.rpc = self.enterContext(
            patch.object(
                solana, "rpc", return_value={"data": self.txs, "paginationToken": None}
            )
        )
        self.head = self.enterContext(
            patch.object(
                history,
                "latest_address_signature",
                return_value={"signature": "signature-130", "slot": 130},
            )
        )
        self.flows = self.enterContext(
            patch.object(
                history,
                "phoenix_positioning",
                side_effect=positioning.phoenix_positioning,
            )
        )
        self.enterContext(
            patch.object(positioning, "get_trades", return_value=([], True, None))
        )
        self.enterContext(
            patch.object(
                positioning,
                "phoenix_candles",
                side_effect=lambda symbol, start, end, **kwargs: pd.DataFrame(
                    [{"trades": 0, "notional": Decimal(0), "volume": Decimal(0)}],
                    index=[start],
                ),
            )
        )
        self.native = self.enterContext(
            patch(
                "quant_tick.exchanges.phoenix.market_history.get_phoenix_response",
                side_effect=lambda path, params: response(
                    [observation(datetime.fromisoformat(params["start_time"]))]
                ),
            )
        )

    def test_newest_first_collection_backfills_ratios_and_resumes_after_a_failure(self):
        self.rpc.side_effect = [
            {"data": self.txs[:3], "paginationToken": "110:0"},
            RuntimeError("provider failed"),
        ]
        with self.assertRaisesRegex(RuntimeError, "provider failed"):
            perpetual_stats(self.symbol, self.start, self.end)
        saved = PerpetualStatsData.objects.get()
        self.assertEqual(saved.timestamp, self.start + timedelta(hours=1))
        self.assertEqual(saved.json_data["long_accounts"], 2)
        self.assertEqual(saved.json_data["short_accounts"], 1)
        self.assertAlmostEqual(saved.long_account_ratio, Decimal(2) / 3, places=12)
        self.assertEqual(saved.json_data["cohort_available_at"], self.end)
        self.assertEqual(saved.json_data["cohort_price_source"], "native_hourly_mark")
        self.assertEqual(
            Decimal(saved.json_data["backstop_long_size"]), Decimal("0.0004")
        )
        self.assertEqual(saved.json_data["backstop_events"], 1)
        self.assertEqual(saved.json_data["trades"], 0)
        self.assertTrue(saved.json_data["liquidation_flow_validated"])
        checkpoint = Checkpoint.objects.get(timestamp=self.start + timedelta(hours=1))
        self.assertEqual(checkpoint.state["before"], "signature-120")
        self.assertFalse(Checkpoint.objects.filter(timestamp=self.start).exists())
        self.rpc.side_effect = None
        self.rpc.return_value = {"data": self.txs[2:], "paginationToken": None}
        self.snapshot.reset_mock()
        perpetual_stats(self.symbol, self.start, self.end)
        self.snapshot.assert_not_called()
        self.assertEqual(
            self.rpc.call_args.args[1][1]["filters"]["signature"],
            {"lt": "signature-120"},
        )
        older = PerpetualStatsData.objects.get(timestamp=self.start)
        self.assertEqual(older.long_account_ratio, Decimal("0.5"))
        self.assertEqual(older.json_data["long_accounts"], 1)
        self.assertTrue(older.json_data["cohort_replay_validated"])
        self.assertEqual(self.flows.call_args.kwargs["replayed_groups"], {})
        self.native.reset_mock()
        perpetual_stats(self.symbol, self.start, self.end)
        self.native.assert_not_called()

    def test_transfers_and_adl_reconcile_without_market_trades_and_retry_idempotently(
        self,
    ):
        end = self.start + timedelta(hours=1)
        a, b, c, d = self.accounts
        for sign in (1, -1):
            with self.subTest(distressed_side=sign):
                Checkpoint.objects.all().delete()
                PerpetualStatsData.objects.all().delete()
                self.snapshot.reset_mock()
                snapshot, _, _ = self.snapshot.return_value
                snapshot["positions"] = {a: sign * 4, b: sign * 4, c: -sign * 8}
                self.snapshot.return_value = (
                    snapshot,
                    end + timedelta(minutes=10),
                    end + timedelta(minutes=10),
                )
                transactions = [
                    history_transaction(130, end + timedelta(minutes=5)),
                    history_transaction(
                        120,
                        self.start + timedelta(minutes=30),
                        [(a, sign * 6, sign * 4), (c, -sign * 10, -sign * 8)],
                        adl=(a, c, c, 2) if sign == 1 else (c, a, c, 2),
                    ),
                    history_transaction(
                        110,
                        self.start,
                        [(a, sign * 10, sign * 6), (b, 0, sign * 4)],
                        (a, b, sign * 4),
                    ),
                    history_transaction(100, self.start - timedelta(seconds=1)),
                ]

                def fetch(method, params, transactions=transactions):
                    before = params[1]["filters"].get("signature", {}).get("lt")
                    offset = next(
                        (
                            i + 1
                            for i, tx in enumerate(transactions)
                            if tx["transaction"]["signatures"][0] == before
                        ),
                        0,
                    )
                    return {"data": transactions[offset:], "paginationToken": None}

                self.rpc.side_effect = fetch
                PerpetualStatsData.objects.create(
                    symbol=self.symbol,
                    timestamp=self.start,
                    frequency=60,
                    open_interest=1,
                    open_interest_value=1,
                    taker_long_short_volume_ratio=Decimal(7),
                    json_data={"liquidated_size": "99"},
                )
                for retry in range(2):
                    perpetual_stats(self.symbol, self.start, end, retry=True)
                    saved = PerpetualStatsData.objects.get()
                    row = saved.json_data
                    side = "long" if sign == 1 else "short"
                    self.assertEqual(row["backstop_events"], 1)
                    self.assertEqual(row["adl_events"], 1)
                    self.assertEqual(row["minute_open_interest_open"], Decimal("0.001"))
                    self.assertEqual(
                        row["minute_open_interest_close"], Decimal("0.0008")
                    )
                    self.assertEqual(row["minute_open_interest_low"], Decimal("0.0008"))
                    self.assertEqual(row["minute_open_interest_high"], Decimal("0.001"))
                    self.assertEqual(
                        row["minute_open_interest_max_drawdown"], Decimal("0.0002")
                    )
                    self.assertEqual(row["minute_open_interest_increase"], 0)
                    self.assertEqual(
                        row["minute_open_interest_decrease"], Decimal("0.0002")
                    )
                    self.assertTrue(
                        all(
                            not (r["taker_buy_size"] or r["taker_sell_size"])
                            for r in row["position_size_flows"]
                        )
                    )
                    self.assertEqual(
                        Decimal(row["backstop_" + side + "_size"]), Decimal("0.0004")
                    )
                    self.assertEqual(
                        Decimal(row["adl_" + side + "_size"]), Decimal("0.0002")
                    )
                    self.assertEqual(
                        Decimal(row[side + "_open_size"]), Decimal("0.0004")
                    )
                    self.assertEqual(
                        Decimal(row[side + "_close_size"]), Decimal("0.0006")
                    )
                    self.assertEqual(row[side + "_account_delta"], 1)
                    for field in (
                        "trades",
                        "liquidations",
                        "liquidated_size",
                        "taker_buy_size",
                        "taker_sell_size",
                    ):
                        self.assertEqual(Decimal(row[field]), 0)
                    self.assertTrue(row["liquidation_flow_validated"])
                    self.assertIsNone(saved.taker_long_short_volume_ratio)
                self.snapshot.assert_called_once()
                self.assertEqual(Checkpoint.objects.count(), 2)

                bad = history_transaction(
                    120,
                    self.start + timedelta(minutes=30),
                    [(a, sign * 6, sign * 4), (c, -sign * 10, -sign * 8)],
                    adl=(a, c, d, 2) if sign == 1 else (c, a, d, 2),
                )
                transactions[1] = bad
                with self.assertRaisesRegex(ValueError, "profitable counterparty"):
                    perpetual_stats(self.symbol, self.start, end, retry=True)
                self.assertEqual(PerpetualStatsData.objects.get().json_data, row)

    def test_lost_lease_rolls_back_the_hour_and_its_lower_checkpoint(self):
        def lost_after_statistics_write():
            if PerpetualStatsData.objects.exists():
                raise RuntimeError("lease lost")

        with self.assertRaisesRegex(RuntimeError, "lease lost"):
            perpetual_stats(
                self.symbol,
                self.start,
                self.end,
                assert_lease_owned=lost_after_statistics_write,
            )
        self.assertFalse(PerpetualStatsData.objects.exists())
        self.assertFalse(Checkpoint.objects.filter(timestamp__lt=self.end).exists())
        self.assertTrue(Checkpoint.objects.filter(timestamp=self.end).exists())

    def test_native_failure_retains_existing_data_and_does_not_advance_replay(self):
        PerpetualStatsData.objects.create(
            symbol=self.symbol,
            timestamp=self.start + timedelta(hours=1),
            frequency=60,
            open_interest=Decimal(4),
            open_interest_value=Decimal(400),
        )
        self.flows.side_effect = ValueError("native mismatch")
        with self.assertRaisesRegex(ValueError, "native mismatch"):
            perpetual_stats(self.symbol, self.start, self.end, retry=True)
        self.assertEqual(PerpetualStatsData.objects.get().open_interest, 4)
        self.assertFalse(Checkpoint.objects.filter(timestamp__lt=self.end).exists())

    def test_missing_or_lagging_history_cannot_publish_counts(self):
        wrong_head = deepcopy(self.txs[0])
        wrong_head["transaction"]["signatures"] = ["earlier-in-the-same-slot"]
        for transactions in (
            [],
            self.txs[:1],
            self.txs[1:],
            [wrong_head, *self.txs[1:]],
        ):
            with self.subTest(slots=[tx["slot"] for tx in transactions]):
                Checkpoint.objects.all().delete()
                self.rpc.return_value = {"data": transactions, "paginationToken": None}
                with self.assertRaisesRegex(ValueError, "ended|ledger boundary"):
                    perpetual_stats(self.symbol, self.start, self.end)
                self.assertFalse(PerpetualStatsData.objects.exists())

    def test_index_lag_keeps_the_snapshot_until_its_transfer_can_be_replayed(self):
        end = self.start + timedelta(hours=1)
        snapshot, _, _ = self.snapshot.return_value
        snapshot_time = end + timedelta(minutes=10)
        self.snapshot.return_value = snapshot, snapshot_time, snapshot_time
        a, b, _, _ = self.accounts
        transfer = history_transaction(
            130,
            end + timedelta(minutes=7),
            [(a, 10, 6), (b, 0, 4)],
            (a, b, 4),
        )
        older = [
            history_transaction(120, end + timedelta(minutes=5)),
            history_transaction(110, self.start + timedelta(minutes=30)),
            history_transaction(100, self.start - timedelta(seconds=1)),
        ]
        self.head.side_effect = solana.latest_address_signature
        self.flows.side_effect = positioning.phoenix_positioning
        self.native.side_effect = None
        self.native.return_value = response(
            [observation(self.start, open_interest=Decimal("0.001"))]
        )
        self.enterContext(
            patch.object(positioning, "get_trades", return_value=([], None, None))
        )
        self.enterContext(
            patch.object(
                positioning,
                "phoenix_candles",
                return_value=pd.DataFrame(
                    [{"trades": 0, "notional": Decimal(0), "volume": Decimal(0)}],
                    index=[self.start],
                ),
            )
        )

        for lagging in (False, True):
            with self.subTest(lagging=lagging):
                Checkpoint.objects.all().delete()
                PerpetualStatsData.objects.all().delete()
                self.snapshot.reset_mock()
                self.head.reset_mock()
                transactions = older.copy() if lagging else [transfer, *older]

                def fetch(method, params, rows=transactions):
                    if method == "getSignaturesForAddress":
                        self.assertEqual(params[1]["minContextSlot"], 140)
                        return [
                            {
                                "slot": 130,
                                "signature": "signature-130",
                                "err": None,
                                "confirmationStatus": "finalized",
                            }
                        ]
                    self.assertEqual(method, "getTransactionsForAddress")
                    self.assertEqual(params[1]["filters"]["slot"], {"lte": 140})
                    return {"data": rows, "paginationToken": None}

                self.rpc.side_effect = fetch
                if lagging:
                    for retry in (False, True):
                        with self.assertRaisesRegex(ValueError, "ledger boundary"):
                            perpetual_stats(self.symbol, self.start, end, retry=retry)
                        self.assertFalse(PerpetualStatsData.objects.exists())
                        pending = Checkpoint.objects.get()
                        self.assertEqual(pending.kind, "phoenix_snapshot")
                        self.assertEqual(pending.timestamp, snapshot_time)
                        self.assertEqual(pending.state["slot"], 140)
                        self.assertEqual(
                            pending.state["positions"],
                            {
                                key.hex(): value
                                for key, value in snapshot["positions"].items()
                            },
                        )
                    transactions.insert(0, transfer)
                perpetual_stats(self.symbol, self.start, end, retry=True)
                self.snapshot.assert_called_once()
                self.head.assert_called_once()
                saved = PerpetualStatsData.objects.get()
                self.assertEqual(saved.json_data["long_accounts"], 1)
                self.assertEqual(saved.json_data["short_accounts"], 1)
                self.assertEqual(saved.long_account_ratio, Decimal("0.5"))
                self.assertTrue(saved.json_data["cohort_replay_validated"])
                self.assertTrue(saved.json_data["trade_position_flow_validated"])
                self.assertFalse(
                    Checkpoint.objects.filter(kind="phoenix_snapshot").exists()
                )
                self.assertEqual(
                    Checkpoint.objects.filter(kind="phoenix_positions").count(), 2
                )

    def test_verified_idle_snapshot_does_not_require_a_transaction_after_hour_end(self):
        self.head.return_value = {"signature": "signature-120", "slot": 120}
        self.rpc.return_value = {"data": self.txs[1:], "paginationToken": None}
        perpetual_stats(self.symbol, self.start, self.end)
        saved = PerpetualStatsData.objects.get(
            timestamp=self.start + timedelta(hours=1)
        )
        self.assertEqual(saved.json_data["long_accounts"], 2)
        self.assertTrue(saved.json_data["cohort_replay_validated"])

    def test_existing_native_rows_receive_reconstruction_without_retry(self):
        unrelated = Checkpoint.objects.create(
            symbol=self.symbol,
            kind="other_collector",
            timestamp=self.end,
            state={"cursor": "unchanged"},
        )
        PerpetualStatsData.objects.create(
            symbol=self.symbol,
            timestamp=self.start + timedelta(hours=1),
            frequency=60,
            open_interest=Decimal(4),
            open_interest_value=Decimal(400),
        )
        perpetual_stats(self.symbol, self.start, self.end)
        unrelated.refresh_from_db()
        self.assertEqual(unrelated.state, {"cursor": "unchanged"})
        self.assertEqual(
            Checkpoint.objects.filter(
                symbol=self.symbol, kind="phoenix_positions"
            ).count(),
            3,
        )
        self.assertEqual(PerpetualStatsData.objects.count(), 2)
        self.assertEqual(
            PerpetualStatsData.objects.filter(
                json_data__cohort_replay_validated=True
            ).count(),
            2,
        )

    def test_empty_reconstruction_clears_old_account_ratios(self):
        PerpetualStatsData.objects.create(
            symbol=self.symbol,
            timestamp=self.start,
            frequency=60,
            open_interest=1,
            open_interest_value=1,
            long_account_ratio=Decimal("0.7"),
            short_account_ratio=Decimal("0.3"),
            long_short_account_ratio=Decimal(7) / 3,
        )
        PerpetualStatsData.write(
            self.symbol,
            60,
            self.start,
            self.end,
            pd.DataFrame(
                [
                    {
                        "timestamp": self.start,
                        "open_interest": 0,
                        "open_interest_value": 0,
                        "long_account_ratio": None,
                        "short_account_ratio": None,
                        "long_short_account_ratio": None,
                        "long_accounts": 0,
                        "short_accounts": 0,
                        "cohort_replay_validated": True,
                    }
                ]
            ),
        )
        saved = PerpetualStatsData.objects.get()
        self.assertIsNone(saved.long_account_ratio)
        self.assertIsNone(saved.short_account_ratio)
        self.assertIsNone(saved.long_short_account_ratio)


class PhoenixPositioningTest(SimpleTestCase):
    def setUp(self):
        path = Path(__file__).parent.parent / "fixtures/phoenix/positioning.json"
        self.fixture = json.loads(path.read_text())
        self.start = datetime(2026, 11, 17, 9, tzinfo=UTC)
        self.end = self.start + timedelta(hours=1)

    def collect(self, signatures, data=None, *, replayed=False):
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
            ) as request,
            patch.object(positioning, "phoenix_candles", return_value=candle),
        ):
            groups = (
                {
                    sig: events.transaction_events(data[sig]["transaction"], sig)
                    for sig in signatures
                }
                if replayed
                else None
            )
            result = positioning.phoenix_positioning(
                "BTC", self.start, self.end, replayed_groups=groups
            )
            if replayed:
                request.assert_not_called()
            return result

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
                self.assertEqual(self.collect([signature], replayed=True), result)
                for field, value in case["stats"].items():
                    self.assertEqual(Decimal(result[field]), Decimal(value), field)
                self.assertTrue(result["trade_position_flow_validated"])
                self.assertNotIn("long_short_account_ratio", result)
                self.assertNotIn("open_interest_high", result)
                self.assertNotIn("minute_open_interest_high", result)
                tiers = result["position_size_flows"]
                nonempty = [
                    r for r in tiers if r["taker_buy_size"] or r["taker_sell_size"]
                ]
                self.assertEqual(
                    [r["tier"] for r in nonempty],
                    [2 if kind == "PlaceMultiLimitOrder" else 1],
                )
                for field in (
                    "taker_buy_size",
                    "taker_sell_size",
                    "taker_long_open_size",
                    "taker_long_close_size",
                    "taker_short_open_size",
                    "taker_short_close_size",
                    "taker_buy_value",
                    "taker_sell_value",
                ):
                    self.assertEqual(sum(r[field] for r in tiers), result[field])
                self.assertNotIn("liquidation_flow_validated", result)
                if kind == "LiquidateViaMarketOrder":
                    self.assertEqual(result["liquidations"], 1)
                    self.assertEqual(result["liquidated_long_size"], Decimal("0.0107"))
                    enhanced = deepcopy(decoded)
                    for group in enhanced:
                        group.extend(
                            {**item, "kind": "LiquidationWithFee"}
                            for item in group.copy()
                            if item["kind"] == "Liquidation"
                        )
                    for keep_original in (True, False):
                        groups = [
                            [
                                item
                                for item in group
                                if keep_original or item["kind"] != "Liquidation"
                            ]
                            for group in enhanced
                        ]
                        totals = positioning.aggregate(
                            positioning.parsed_trades(groups, "BTC")
                        )
                        self.assertEqual(totals["liquidations"], 1)
                        self.assertEqual(
                            totals["liquidated_size"], result["liquidated_size"]
                        )

    def test_one_order_can_cross_position_brackets_between_fills(self):
        for sign in (1, -1):
            with self.subTest(taker_direction=sign):
                sizes = (150, 100, 100)
                trade = {
                    "header": {
                        "base_lot_decimals": 0,
                        "quote_lot_decimals": 0,
                        "tick_size": 1,
                    },
                    "summary": {
                        "trader": "taker",
                        "side": "bid" if sign == 1 else "ask",
                        "trade_sequence_number": 1,
                        "base_lots_filled": 350,
                        "quote_lots_filled": 35000,
                        "base_lot_position": sign * 150,
                        "fee_in_quote_lots": 0,
                    },
                    "fills": [
                        {
                            "maker": i,
                            "side": "ask" if sign == 1 else "bid",
                            "base_lots_filled": size,
                            "quote_lots_filled": size * 100,
                            "maker_base_lot_position": 0,
                            "price": 100,
                        }
                        for i, size in enumerate(sizes)
                    ],
                    "pnls": [
                        {
                            "trader": "taker",
                            "base_lots_before": -sign * 200,
                            "base_lots_after": sign * 150,
                            "realized_pnl": 0,
                        },
                        *(
                            {
                                "trader": i,
                                "base_lots_before": sign * size,
                                "base_lots_after": 0,
                                "realized_pnl": 0,
                            }
                            for i, size in enumerate(sizes)
                        ),
                    ],
                    "liquidated": False,
                }
                positioning.validate_trade(trade)
                row = positioning.aggregate([trade])
                small, medium = row["position_size_flows"][:2]
                side = "buy" if sign == 1 else "sell"
                closed = "short" if sign == 1 else "long"
                opened = "long" if sign == 1 else "short"
                self.assertEqual(medium["taker_" + side + "_size"], 150)
                self.assertEqual(medium["taker_" + closed + "_close_size"], 150)
                self.assertEqual(small["taker_" + side + "_size"], 200)
                self.assertEqual(small["taker_" + closed + "_close_size"], 50)
                self.assertEqual(small["taker_" + opened + "_open_size"], 150)

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
                },
            ),
            patch.object(
                solana, "get_public_json", return_value={"id": signature, "result": tx}
            ) as request,
        ):
            self.assertEqual(solana.transaction(signature), tx)
            self.assertTrue(request.call_args.kwargs["redact_url"])
            self.assertEqual(request.call_args.kwargs["reset_seconds"], 1 / 50)
            request.return_value = {
                "id": 1,
                "result": {"context": {"slot": 100}, "value": []},
            }
            solana.rpc("getProgramAccounts", [events.PROGRAM])
            self.assertEqual(request.call_args.kwargs["reset_seconds"], 1 / 3)
            self.assertTrue(request.call_args.kwargs["redact_url"])
            solana.transaction.cache_clear()
            request.return_value = {"id": signature, "result": None}
            with self.assertRaisesRegex(ValueError, "omitted"):
                solana.transaction(signature)
