import json
import os
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from django.test import SimpleTestCase, TestCase

from quant_tick.constants import Exchange, SymbolType
from quant_tick.exchanges.perpetual_stats import perpetual_stats
from quant_tick.exchanges.quicknode import flows, market_history, positions
from quant_tick.models import PerpetualStatsData

from ..base import BaseSymbolTest

START = pd.Timestamp("2025-10-10T19:00Z")
END = START + pd.Timedelta(hours=4)


def fixture():
    path = Path(__file__).parent.parent / "fixtures/quicknode/positioning.json"
    return json.loads(path.read_text())


def collect(data=None):
    data = deepcopy(data or fixture())
    with (
        patch.object(
            market_history, "block_range", return_value=(758656732, 758877568)
        ),
        patch.object(
            positions,
            "query",
            side_effect=[
                data[k] for k in ("funding", "last_blocks", "last_trades", "tiers")
            ],
        ),
        patch("quant_tick.exchanges.quicknode.flows.query", return_value=data["flows"]),
    ):
        return market_history.market_window("BTC", START, END)


class QuickNodeStatsTest(SimpleTestCase):
    def test_crash_hour_positions_flows_and_cohorts_reconcile(self):
        rows = collect()
        self.assertEqual(len(rows), 4)
        for row, expected in zip(rows, fixture()["expected"], strict=True):
            for field, value in expected.items():
                self.assertEqual(Decimal(str(row[field])), Decimal(str(value)), field)
            self.assertIsNone(row["open_interest_high"])
            self.assertIsNone(row["open_interest_low"])
            self.assertEqual(row["market_history_source"], "quicknode")
            self.assertEqual(row["cohort_price_source"], "last_trade_before_funding")
            self.assertTrue(row["position_flow_validated"])
            self.assertTrue(row["liquidation_flow_validated"])
            self.assertLess(row["cohort_timestamp"], row["oi_timestamp"])
        self.assertEqual(rows[2]["liquidated_size"], Decimal("5472.31819"))
        self.assertEqual(rows[2]["backstop_size"], Decimal("6077.05804"))
        self.assertEqual(rows[2]["adl_size"], Decimal("5832.39030"))
        self.assertEqual(rows[2]["netting_size"], Decimal("30.40946"))
        self.assertEqual(rows[2]["adl_short_size"], Decimal("0.39549"))
        self.assertEqual(rows[2]["backstop_short_size"], Decimal("367.16769"))
        self.assertEqual(rows[2]["trades"], 229248)
        self.assertEqual(rows[2]["position_events"], 233527)
        self.assertEqual(rows[2]["adl_events"], 2443)
        self.assertEqual(rows[2]["cohort_price"], Decimal(114126))
        self.assertEqual(rows[2]["large_long_size"], Decimal("15650.21991"))
        self.assertEqual(rows[2]["taker_buy_value"], Decimal("1218081260.02816"))
        self.assertEqual(rows[2]["long_position_flips"], 2007)
        self.assertEqual(rows[0]["maker_fees"], Decimal("-8959.142465"))
        self.assertEqual(rows[0]["fee_currency"], "USDC")

        data = fixture()
        data["flows"][0]["fee_tokens"] = ["USDC", "USDH"]
        with self.assertRaisesRegex(ValueError, "currencies"):
            collect(data)

    def test_complete_fill_pairs_reconcile_missing_derived_trades(self):
        path = (
            Path(__file__).parent.parent / "fixtures/quicknode/missing_trade_pairs.json"
        )
        data = json.loads(path.read_text())
        start = pd.Timestamp("2026-04-05T22:00Z")
        end = start + pd.Timedelta(hours=1, minutes=1)
        with patch.object(positions, "query", return_value=deepcopy(data["funding"])):
            points = positions.funding_points("BTC", start, end, None)
        before, after = points.values()

        def request(sql, **kwargs):
            if "FROM hyperliquid_fills" in sql:
                return deepcopy(data["fill_flows"])
            if "FROM hyperliquid_trades" in sql:
                return deepcopy(data["derived_flows"])
            self.fail("Unexpected position-flow source")

        with patch.object(flows, "query", side_effect=request):
            flow = flows.position_flows("BTC", points, start, end)[0]
        flows.validate_flow(before, after, flow)
        self.assertEqual(flow["trades"], 52940)
        self.assertEqual(flow["long_open_size"], Decimal("1147.87021"))
        self.assertEqual(flow["short_open_size"], Decimal("1692.68045"))
        self.assertEqual(flow["taker_buy_value"], Decimal("138606178.54091"))
        self.assertEqual(flow["taker_sell_value"], Decimal("78165996.16129"))

    def test_incomplete_or_inconsistent_evidence_cannot_become_valid_statistics(self):
        for section, field in (
            ("funding", "users"),
            ("funding", "short_size"),
            ("funding", "last_rate"),
            ("tiers", "long_size"),
            ("flows", "long_open_size"),
            ("flows", "long_account_delta"),
            ("flows", "liquidated_short_size"),
            ("flows", "liquidated_short_value"),
            ("flows", "adl_size"),
            ("flows", "backstop_short_value"),
            ("flows", "netting_size"),
            ("flows", "trades"),
        ):
            data = fixture()
            data[section][0][field] = str(Decimal(str(data[section][0][field])) + 1)
            with (
                self.subTest(section=section, field=field),
                self.assertRaisesRegex(ValueError, "reconcile|event counts"),
            ):
                collect(data)

        data = fixture()
        data["flows"][0]["invalid_events"] = 1
        with self.assertRaisesRegex(ValueError, "unclassified"):
            collect(data)

    def test_absent_flow_is_left_retryable_instead_of_zero_filled(self):
        data = fixture()
        data["flows"].pop(1)
        rows = collect(data)
        self.assertEqual(
            [r["timestamp"] for r in rows],
            [START, START + pd.Timedelta(hours=2), START + pd.Timedelta(hours=3)],
        )

    def test_forced_only_intervals_reconcile_without_inventing_market_trades(self):
        points = {START: {"block_number": 1}, END: {"block_number": 2}}
        for kind in ("backstop", "adl"):
            for side in ("long", "short"):
                with self.subTest(kind=kind, side=side):
                    raw = {
                        key: 0
                        for key in (
                            *flows.FLOW_FIELDS,
                            *flows.SIGNED_FLOW_FIELDS,
                            *flows.COUNT_FIELDS,
                        )
                    }
                    raw.update(
                        interval_index=0,
                        invalid_events=0,
                        position_events=1,
                        trades=0,
                        long_account_delta=0,
                        short_account_delta=0,
                        fee_tokens=["USDC"],
                    )
                    raw.update(
                        {
                            kind + "_events": 1,
                            kind + "_size": 4,
                            kind + "_" + side + "_size": 4,
                            kind + "_value": 400,
                            kind + "_" + side + "_value": 400,
                        }
                    )
                    before = {
                        "long_size": 10,
                        "short_size": 10,
                        "long_accounts": 1,
                        "short_accounts": 1,
                    }
                    after = before.copy()
                    if kind == "backstop":
                        raw[side + "_open_size"] = raw[side + "_close_size"] = 4
                        raw[side + "_account_delta"] = 1
                        after[side + "_accounts"] += 1
                    else:
                        raw["long_close_size"] = raw["short_close_size"] = 4
                        after["long_size"] = after["short_size"] = 6
                    with patch.object(flows, "query", return_value=[raw]):
                        row = flows.position_flows("BTC", points, START, END)[0]
                    flows.validate_flow(before, after, row)
                    self.assertEqual(row["trades"], 0)
                    self.assertEqual(row["liquidated_size"], 0)
                    self.assertEqual(row["taker_buy_size"] + row["taker_sell_size"], 0)

    def test_missing_terminal_snapshot_leaves_that_hour_retryable(self):
        data = fixture()
        terminal = data["funding"].pop()["block_number"]
        data["last_blocks"] = [
            r for r in data["last_blocks"] if r["target"] != terminal
        ]
        required = {r["last_block"] for r in data["last_blocks"]}
        data["last_trades"] = [
            r for r in data["last_trades"] if r["block_number"] in required
        ]
        data["tiers"] = [r for r in data["tiers"] if r["block_number"] != terminal]
        data["flows"].pop()
        rows = collect(data)
        self.assertEqual(
            [r["timestamp"] for r in rows],
            list(pd.date_range(START, periods=3, freq="h")),
        )

    def test_missing_price_keeps_cohorts_retryable(self):
        data = fixture()
        target = data["last_blocks"][0]["target"]
        data["last_blocks"][0]["last_block"] = 0
        required = {r["last_block"] for r in data["last_blocks"]}
        data["last_trades"] = [
            r for r in data["last_trades"] if r["block_number"] in required
        ]
        data["tiers"] = [r for r in data["tiers"] if r["block_number"] != target]
        rows = collect(data)
        self.assertEqual(
            [r["timestamp"] for r in rows],
            list(pd.date_range(START + pd.Timedelta(hours=1), periods=3, freq="h")),
        )

    def test_first_funding_hour_is_available_despite_subsecond_settlement(self):
        market_history.funding_history_start.cache_clear()
        self.addCleanup(market_history.funding_history_start.cache_clear)
        with patch.object(
            market_history,
            "query",
            return_value=[{"time": "2025-09-27 10:00:00.021257"}],
        ):
            self.assertEqual(
                market_history.funding_history_start(END.date()),
                pd.Timestamp("2025-09-27T10:00Z"),
            )

    def test_unfinished_hours_and_precoverage_history_do_not_make_requests(self):
        for start, end, first in (
            (START, START + pd.Timedelta(minutes=40), START),
            (START, END, END),
        ):
            with (
                self.subTest(start=start, end=end, first=first),
                patch.object(market_history, "get_current_time", return_value=END),
                patch.object(
                    market_history, "funding_history_start", return_value=first
                ),
                patch.object(market_history, "market_window") as request,
            ):
                frame = market_history.quicknode_market_history("BTC", start, end)
                self.assertTrue(frame.empty)
                request.assert_not_called()

    def test_query_windows_are_bounded_and_keep_partial_edge_hours(self):
        with (
            patch.object(
                market_history,
                "get_current_time",
                return_value=END + pd.Timedelta(days=1),
            ),
            patch.object(market_history, "funding_history_start", return_value=START),
            patch.object(market_history, "BATCH", pd.Timedelta(hours=2)),
            patch.object(market_history, "market_window", return_value=[]) as request,
        ):
            market_history.quicknode_market_history(
                "BTC", START - pd.Timedelta(minutes=10), END + pd.Timedelta(minutes=10)
            )
        self.assertEqual(
            [(c.args[1], c.args[2]) for c in request.call_args_list],
            [
                (START, START + pd.Timedelta(hours=2)),
                (START + pd.Timedelta(hours=2), END),
            ],
        )

    def test_preceding_prices_follow_execution_order_not_hashed_ids(self):
        path = Path(__file__).parent.parent / "fixtures/quicknode/ordered_block.json"
        rows = json.loads(path.read_text())["rows"]
        block = rows[0]["block_number"]
        moment = pd.Timestamp(rows[0]["timestamp"])
        with (
            patch.object(
                positions,
                "query",
                side_effect=[[{"target": block + 1, "last_block": block}], rows],
            ),
            patch("quant_tick.exchanges.quicknode.blocks.get_trade_blocks") as raw,
        ):
            result = positions.preceding_prices(
                "BTC",
                {moment: {"block_number": block + 1}},
                moment - pd.Timedelta(hours=1),
                moment + pd.Timedelta(hours=1),
                None,
            )
        self.assertEqual(result[block + 1], Decimal(64887))
        raw.assert_not_called()

    def test_provider_errors_propagate(self):
        with (
            patch.object(
                market_history,
                "block_range",
                side_effect=RuntimeError("provider error"),
            ),
            self.assertRaisesRegex(RuntimeError, "provider error"),
        ):
            market_history.market_window("BTC", START, END)


class QuickNodeStatsPersistenceTest(BaseSymbolTest, TestCase):
    def setUp(self):
        super().setUp()
        self.symbol = self.get_symbol(
            exchange=Exchange.HYPERLIQUID,
            api_symbol="BTC",
            symbol_type=SymbolType.PERPETUAL,
        )

    def test_older_data_is_preserved_and_new_metadata_survives_export(self):
        row = collect()[0]
        old = START - pd.Timedelta(days=365)
        PerpetualStatsData.write(
            self.symbol,
            60,
            old,
            END,
            pd.DataFrame(
                [
                    {
                        "timestamp": old,
                        "open_interest": Decimal(12),
                        "market_history_source": "femtos",
                    },
                    {
                        "timestamp": START,
                        "open_interest": row["open_interest"],
                        "open_interest_high": Decimal(999999),
                        "market_history_source": "femtos",
                    },
                ]
            ).set_index("timestamp"),
        )
        with (
            patch.dict(os.environ, {"QUICKNODE_API_KEY": "test-key"}),
            patch(
                "quant_tick.exchanges.perpetual_stats.quicknode_market_history",
                return_value=pd.DataFrame([row]).set_index("timestamp"),
            ),
        ):
            perpetual_stats(
                self.symbol, START, START + pd.Timedelta(hours=1), retry=True
            )
        self.assertEqual(PerpetualStatsData.objects.count(), 2)
        self.assertEqual(
            PerpetualStatsData.objects.get(timestamp=old).json_data[
                "market_history_source"
            ],
            "femtos",
        )
        stored = PerpetualStatsData.objects.get(timestamp=START).to_row()
        self.assertIsNone(stored.get("open_interest_high"))
        self.assertEqual(stored["cohort_price_source"], "last_trade_before_funding")
        self.assertEqual(
            Decimal(stored["liquidated_long_size"]), row["liquidated_long_size"]
        )

    def test_no_key_does_not_start_positioning_requests(self):
        with (
            patch.dict(os.environ, {"QUICKNODE_API_KEY": " "}),
            patch(
                "quant_tick.exchanges.perpetual_stats.quicknode_market_history"
            ) as request,
        ):
            perpetual_stats(self.symbol, START, END)
        request.assert_not_called()
