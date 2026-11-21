import gzip
import json
import os
import struct
from copy import deepcopy
from decimal import Decimal
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

import pandas as pd
from django.core.files.storage import default_storage
from django.db import transaction
from django.test import SimpleTestCase, TestCase, override_settings

from quant_tick.constants import CheckpointType, Exchange, SymbolType
from quant_tick.exchanges import liquidation_snapshots as snapshots
from quant_tick.exchanges.hyperliquid import liquidation as hyperliquid
from quant_tick.exchanges.liquidation_exposure import VERSION, summarize
from quant_tick.exchanges.perpetual_stats import perpetual_stats
from quant_tick.exchanges.phoenix import accounts
from quant_tick.exchanges.phoenix import liquidation as phoenix
from quant_tick.exchanges.quicknode import portfolios
from quant_tick.models import Checkpoint, PerpetualStatsData, Symbol


def fixture(name):
    path = Path(__file__).parents[1] / "fixtures" / name
    return json.loads(gzip.decompress(path.read_bytes()))


def portfolio(size, equity=80, required=None, mode="disabled"):
    size = Decimal(size)
    price = Decimal(1000)
    maintenance = abs(size) * price / 20
    required = maintenance if required is None else required
    boundary = (size * price - Decimal(equity) + required - maintenance) / (
        size - abs(size) / 20
    )
    return {
        "userAbstraction": mode,
        "clearinghouseState": {
            "native": {
                "time": 1795248000000,
                "crossMarginSummary": {"accountValue": str(equity)},
                "crossMaintenanceMarginUsed": str(required),
                "assetPositions": [
                    {
                        "position": {
                            "coin": "BTC",
                            "szi": str(size),
                            "positionValue": str(abs(size) * price),
                            "leverage": {"type": "cross"},
                            "liquidationPx": str(boundary),
                        }
                    }
                ],
            }
        },
    }


class LiquidationMathTest(SimpleTestCase):
    def test_cumulative_bands_keep_unknown_and_already_exposed_separate(self):
        for sign, side in ((1, "long"), (-1, "short")):
            census = {
                key: sign * Decimal(size) for key, size in enumerate((1, 2, 3, 4, 5))
            }
            rows = {
                key: {
                    "size": census[key],
                    "price": Decimal(100),
                    "buffers": list(map(Decimal, values)),
                }
                for key, values in enumerate(
                    ((10, 1, -1, -2), (10, -1, -2, -3), (0, -1, -2, -3))
                )
            }
            rows[3] = {
                "size": census[3],
                "unknown": "unsupported_account_mode",
            }
            result = summarize(census, rows)[side]
            self.assertEqual(result["coverage"], Decimal(6) / 15)
            self.assertEqual(
                result["unknown_by_reason"],
                {
                    "unsupported_account_mode": Decimal(4),
                    "missing_portfolio": Decimal(5),
                },
            )
            self.assertEqual(
                [result[k]["size"] for k in ("already", "1", "3", "5")],
                list(map(Decimal, (3, 2, 3, 3))),
            )
            self.assertEqual(result["3"]["observed_share"], Decimal("0.2"))
            self.assertEqual(result["3"]["largest_account_share"], Decimal(2) / 3)
            result = summarize(census, {})[side]
            self.assertEqual(result["coverage"], 0)
            self.assertIsNone(result["5"]["size"])

            rows[3]["size"] *= 2
            with self.assertRaisesRegex(ValueError, "quantities disagree"):
                summarize(census, rows)

    def test_selection_coverage_does_not_claim_current_quantities(self):
        totals = {
            "long_size": Decimal(10),
            "short_size": Decimal(10),
            "long_accounts": 8,
            "short_accounts": 8,
        }
        result = hyperliquid.selection_coverage({"one": Decimal(2)}, totals, {})
        self.assertEqual(
            result["long"]["unobserved_by_reason"],
            {"missing_portfolio": Decimal(2)},
        )
        self.assertEqual(result["long"]["sampled_share"], Decimal("0.2"))
        self.assertEqual(result["short"]["sampled_share"], 0)
        self.assertEqual(result["short"]["unobserved_accounts"], 0)
        with self.assertRaisesRegex(ValueError, "exceeds"):
            hyperliquid.selection_coverage({"one": Decimal(11)}, totals, {})


class HyperliquidLiquidationTest(SimpleTestCase):
    metas: ClassVar[list] = [
        {
            "collateralToken": 0,
            "universe": [{"name": "BTC", "marginTableId": 10}],
            "marginTables": [],
        }
    ]
    received = pd.Timestamp(1795248000000, unit="ms", tz="UTC")

    def test_cross_isolated_and_both_shock_directions(self):
        for sign, expected in (
            (1, (30, "20.5", "1.5", "-17.5")),
            (-1, (30, "19.5", "-1.5", "-22.5")),
        ):
            for margin_mode in ("cross", "isolated"):
                state = portfolio(sign)
                if margin_mode == "isolated":
                    state["clearinghouseState"]["native"]["assetPositions"][0][
                        "position"
                    ]["leverage"] = {
                        "type": "isolated",
                        "rawUsd": str(80 - sign * 1000),
                    }
                    state["clearinghouseState"]["native"]["crossMarginSummary"][
                        "accountValue"
                    ] = "999999"
                result = hyperliquid.observe(state, self.metas, [None], self.received)
                self.assertEqual(result["buffers"], list(map(Decimal, expected)))

    def test_unified_balance_is_shared_and_isolated_allocations_are_removed_once(self):
        state = portfolio(1, equity=70, required=60, mode="unifiedAccount")
        state["spotClearinghouseState"] = {
            "balances": [
                {"token": 0, "total": "80"},
                {"coin": "+14720", "total": "39789.0", "hold": "0.0"},
            ]
        }
        native = state["clearinghouseState"]["native"]
        native["crossMarginSummary"]["accountValue"] = "9000"
        native["crossMaintenanceMarginUsed"] = "50"
        state["clearinghouseState"]["xyz"] = {
            "time": native["time"],
            "crossMaintenanceMarginUsed": "10",
            "assetPositions": [
                {"position": {"leverage": {"type": "isolated"}, "marginUsed": "10"}}
            ],
        }
        metas = self.metas + [{"collateralToken": 0}, {"collateralToken": 1}]
        dexes = [None, {"name": "xyz"}, {"name": "other"}]
        result = hyperliquid.observe(state, metas, dexes, self.received)
        self.assertEqual(
            result["buffers"], list(map(Decimal, (10, "0.5", "-18.5", "-37.5")))
        )
        del state["clearinghouseState"]["xyz"]
        self.assertEqual(
            hyperliquid.observe(state, metas, dexes, self.received)["unknown"],
            "incomplete_unified_portfolio",
        )

    def test_live_portfolios_agree_with_native_liquidation_prices(self):
        data = fixture("quicknode/portfolios.json.gz")
        supported = 0
        for state in data["portfolios"]:
            result = hyperliquid.observe(
                state, data["metas"], data["dexes"], data["received"]
            )
            if state["userAbstraction"] in ("disabled", "unifiedAccount"):
                self.assertNotIn("unknown", result)
                supported += 1
            else:
                self.assertEqual(result["unknown"], "unsupported_account_mode")
        self.assertEqual(supported, 6)

    def test_bad_or_old_evidence_is_unknown(self):
        for kind in ("mode", "collateral", "time", "missing", "null", "invalid"):
            state = portfolio(1)
            if kind == "mode":
                state["userAbstraction"] = "portfolioMargin"
            elif kind == "collateral":
                state["clearinghouseState"]["native"]["crossMarginSummary"][
                    "accountValue"
                ] = "800"
            elif kind == "time":
                state["clearinghouseState"]["native"]["time"] -= 600000
            elif kind == "missing":
                del state["clearinghouseState"]["native"]["assetPositions"]
            else:
                state["clearinghouseState"]["native"]["assetPositions"] = (
                    None if kind == "null" else {}
                )
            self.assertIn(
                "unknown", hyperliquid.observe(state, self.metas, [None], self.received)
            )

    def test_tier_boundary_uses_progressive_maintenance(self):
        tiers = [
            {"lowerBound": "0", "maxLeverage": 20},
            {"lowerBound": "1000", "maxLeverage": 10},
        ]
        self.assertEqual(
            hyperliquid.maintenance(Decimal(2), Decimal(1000), tiers), Decimal(75)
        )
        self.assertEqual(
            hyperliquid.maintenance(Decimal(-2), Decimal(500), tiers), Decimal(25)
        )

    @patch.dict(os.environ, {"QUICKNODE_HYPERCORE_URL": "configured"})
    def collect(self, census, states):
        metas = deepcopy(self.metas)
        metas[0]["universe"][0]["szDecimals"] = 0
        totals = {
            "long_size": 100,
            "short_size": 100,
            "long_accounts": 100,
            "short_accounts": 100,
        }
        selected = self.received - pd.Timedelta(minutes=8)
        completed = self.received + pd.Timedelta(seconds=1)
        with (
            patch.object(
                hyperliquid,
                "position_census",
                return_value=(
                    {"snapshot_time": selected.isoformat(), "block_number": 100},
                    census,
                    totals,
                ),
            ) as select,
            patch.object(
                hyperliquid, "post_hyperliquid_info", side_effect=[metas, [None]]
            ) as metadata,
            patch.object(
                hyperliquid,
                "portfolio_states",
                return_value=(states, list(census.keys() - states.keys())),
            ) as fetch,
            patch.object(
                hyperliquid,
                "get_current_time",
                side_effect=[
                    self.received - pd.Timedelta(seconds=5),
                    self.received,
                    completed,
                ],
            ),
        ):
            result = hyperliquid.fetch_snapshot()
        self.assertEqual(result["available_at"], completed)
        self.assertEqual(result["summary"]["available_at"], completed)
        self.assertEqual(
            result["summary"]["selection"]["timestamp"], selected.isoformat()
        )
        self.assertEqual(
            result["summary"]["coverage_basis"], "observed_sample_positions"
        )
        select.assert_called_once_with(1000)
        self.assertEqual(metadata.call_count, 2)
        fetch.assert_called_once_with(list(census))
        return result["summary"]

    def test_missing_collateral_stays_unknown_and_availability_follows_calculation(
        self,
    ):
        missing = portfolio(1)
        del missing["clearinghouseState"]["native"]["crossMarginSummary"][
            "accountValue"
        ]
        result = self.collect(
            {"a": Decimal(1), "b": Decimal(1)}, {"a": portfolio(1), "b": missing}
        )
        self.assertEqual(
            result["long"]["unknown_by_reason"],
            {"incomplete_portfolio": Decimal(1)},
        )
        self.assertEqual(result["long"]["coverage"], Decimal("0.5"))
        self.assertEqual(result["selection"]["long"]["unobserved_accounts"], 0)

    def test_changed_positions_use_fresh_quantities_and_directions(self):
        census = {
            "increased": Decimal(1),
            "reduced": Decimal(-3),
            "flipped_short": Decimal(1),
            "flipped_long": Decimal(-1),
            "closed": Decimal(2),
            "missing": Decimal(-2),
            "stale": Decimal(1),
            "unsupported": Decimal(-1),
            "incomplete": Decimal(1),
        }
        closed, stale = portfolio(1), portfolio(1)
        closed["clearinghouseState"]["native"]["assetPositions"] = []
        stale["clearinghouseState"]["native"]["time"] -= 600000
        result = self.collect(
            census,
            {
                "increased": portfolio(3, equity=240),
                "reduced": portfolio(-1),
                "flipped_short": portfolio(-1),
                "flipped_long": portfolio(1),
                "closed": closed,
                "stale": stale,
                "unsupported": portfolio(-2, mode="portfolioMargin"),
                "incomplete": {},
            },
        )
        self.assertEqual(result["long"]["observed_size"], 4)
        self.assertEqual(result["long"]["observed_accounts"], 2)
        self.assertEqual(result["long"]["coverage"], 1)
        self.assertEqual(result["short"]["observed_size"], 4)
        self.assertEqual(result["short"]["observed_accounts"], 3)
        self.assertEqual(result["short"]["coverage"], Decimal("0.5"))
        self.assertEqual(
            result["short"]["unknown_by_reason"],
            {"unsupported_account_mode": Decimal(2)},
        )
        self.assertEqual(
            [result["long"][k]["size"] for k in ("1", "3", "5")], [0, 0, 4]
        )
        self.assertEqual(
            [result["short"][k]["size"] for k in ("1", "3", "5")], [0, 2, 2]
        )
        self.assertEqual(result["long"]["5"]["value"], 4000)
        self.assertEqual(result["short"]["5"]["value"], 2000)
        self.assertEqual(result["selection"]["long"]["sampled_size"], 6)
        self.assertEqual(result["selection"]["short"]["sampled_size"], 7)
        self.assertEqual(
            result["selection"]["long"]["unobserved_by_reason"],
            {"stale_portfolio": Decimal(1), "incomplete_portfolio": Decimal(1)},
        )
        self.assertEqual(
            result["selection"]["short"]["unobserved_by_reason"],
            {"missing_portfolio": Decimal(2)},
        )

    def test_current_sample_can_exceed_older_market_size(self):
        result = self.collect({"a": Decimal(1)}, {"a": portfolio(200, equity=16000)})
        self.assertEqual(result["long"]["observed_size"], 200)
        self.assertEqual(result["long"]["coverage"], 1)
        self.assertEqual(result["long"]["5"]["observed_share"], 1)
        self.assertEqual(result["selection"]["long"]["size"], 100)
        self.assertEqual(result["selection"]["long"]["sampled_share"], Decimal("0.01"))


class PhoenixLiquidationTest(SimpleTestCase):
    def test_snapshot_saves_complete_calculation_inputs_without_unrelated_accounts(
        self,
    ):
        raw = fixture("phoenix/accounts.json.gz")["result"]
        decoded = accounts.decode_snapshot(raw, "BTC")
        decoded["accounts"] = raw
        observed = pd.Timestamp("2026-11-18T00:00Z")
        completed = observed + pd.Timedelta(seconds=5)
        with (
            patch.object(phoenix, "available", return_value=True),
            patch.object(
                phoenix,
                "fetch_position_snapshot",
                return_value=(decoded, observed, completed),
            ),
            patch.object(phoenix, "get_current_time", return_value=completed),
        ):
            result = phoenix.fetch_snapshot()
        self.assertEqual(len(result["portfolios"]), 403)
        self.assertNotIn("accounts", result)
        self.assertEqual(result["available_at"], completed)
        self.assertEqual(result["summary"]["reference_timestamp"], observed)
        account = next(iter(result["portfolios"].values()))
        restored = {
            "collateral": account["collateral"],
            "positions": {
                key: bytes.fromhex(value) for key, value in account["positions"].items()
            },
        }
        check = phoenix.observe(
            restored,
            result["markets"],
            result["target"],
            decoded["slot"],
            result["quote_decimals"],
        )
        self.assertEqual(check, next(iter(result["observations"].values())))

    def test_full_portfolio_buffers_match_official_sdk_for_captured_accounts(self):
        raw = fixture("phoenix/accounts.json.gz")["result"]
        expected = fixture("phoenix/margin.json.gz")["buffers"]
        decoded = accounts.decode_snapshot(raw, "BTC")
        portfolios, markets, quote_decimals = phoenix.portfolios(raw)
        matched, excluded = 0, 0
        for key in decoded["positions"]:
            result = phoenix.observe(
                portfolios[key],
                markets,
                decoded["asset_id"],
                decoded["slot"],
                quote_decimals,
            )
            if key.hex() in expected:
                self.assertNotIn("unknown", result)
                self.assertEqual(
                    [v * 10**quote_decimals for v in result["buffers"]],
                    expected[key.hex()],
                )
                matched += 1
            else:
                self.assertEqual(result["unknown"], "unsupported_spot_collateral")
                excluded += 1
        self.assertEqual((matched, excluded), (388, 15))

    def test_oracle_staleness_and_missing_markets_are_unknown(self):
        raw = fixture("phoenix/accounts.json.gz")["result"]
        decoded = accounts.decode_snapshot(raw, "BTC")
        portfolios, markets, quote_decimals = phoenix.portfolios(raw)
        target = decoded["asset_id"]
        account = next(
            v
            for v in portfolios.values()
            if target in v["positions"] and all(k < 0xFF000000 for k in v["positions"])
        )
        stale = deepcopy(markets)
        stale[target]["price_components"][0]["prices"][0] = (1, 1)
        self.assertEqual(
            phoenix.observe(account, stale, target, decoded["slot"], quote_decimals)[
                "unknown"
            ],
            "stale_market_mark",
        )
        account = deepcopy(account)
        account["positions"][123456] = struct.pack("<q", 1) + bytes(56)
        self.assertEqual(
            phoenix.observe(account, markets, target, decoded["slot"], quote_decimals)[
                "unknown"
            ],
            "missing_market_parameters",
        )


class PortfolioRequestTest(SimpleTestCase):
    @patch.dict(
        os.environ, {"QUICKNODE_HYPERCORE_URL": "https://example.quiknode.pro/test/"}
    )
    def test_batch_failure_is_unknown_but_omissions_and_duplicates_fail(self):
        response = {
            "id": 1,
            "result": {
                "successful_states": [["a", {"userAbstraction": "disabled"}]],
                "failed_wallets": ["b"],
            },
        }
        with patch.object(
            portfolios, "get_public_json", return_value=response
        ) as request:
            states, failed = portfolios.portfolio_states(["a", "b"])
        self.assertEqual(list(states), ["a"])
        self.assertEqual(failed, ["b"])
        self.assertEqual(
            request.call_args.kwargs["json_body"]["params"]["dex"], "ALL_DEXES"
        )
        self.assertTrue(request.call_args.kwargs["redact_url"])
        for missing in ([], ["a", "b"], ["b", "b"]):
            response["result"]["failed_wallets"] = missing
            with (
                patch.object(portfolios, "get_public_json", return_value=response),
                self.assertRaises(ValueError),
            ):
                portfolios.portfolio_states(["a", "b"])

    def test_census_keeps_totals_separate_from_bounded_sample(self):
        now = pd.Timestamp("2026-11-21T12:00Z")
        totals = {
            "long_size": 100,
            "short_size": 100,
            "long_accounts": 20,
            "short_accounts": 20,
        }
        rows = [{"user": "0x" + "a" * 40, "size": 50}]
        with (
            patch.object(portfolios, "get_current_time", return_value=now),
            patch.object(
                portfolios,
                "query",
                side_effect=[
                    [{"block_number": 100, "snapshot_time": now.isoformat()}],
                    [totals],
                    rows,
                ],
            ) as query,
        ):
            _, selected, result = portfolios.position_census(1)
        self.assertEqual(result, totals)
        self.assertEqual(len(selected), 1)
        self.assertIn("sumIf", query.call_args_list[1].args[0])
        self.assertTrue(query.call_args.args[0].endswith("LIMIT 1"))


class LiquidationPersistenceTest(TestCase):
    def setUp(self):
        self.enterContext(
            override_settings(
                STORAGES={
                    "default": {"BACKEND": "django.core.files.storage.InMemoryStorage"}
                }
            )
        )
        self.symbol = Symbol.objects.create(
            exchange=Exchange.HYPERLIQUID,
            api_symbol="BTC",
            symbol_type=SymbolType.PERPETUAL,
        )
        self.hour = pd.Timestamp("2026-11-21T12:00Z")
        self.state = {
            "version": VERSION,
            "available_at": self.hour + pd.Timedelta(seconds=5),
            "summary": {
                "reference_timestamp": self.hour - pd.Timedelta(minutes=5),
                "available_at": self.hour + pd.Timedelta(seconds=5),
                "long": {"coverage": Decimal("0.5")},
            },
        }

    def test_checkpoint_and_summary_share_availability_and_survive_history_retry(self):
        snapshots.persist(self.symbol, self.state)
        checkpoint = Checkpoint.objects.get()
        self.assertEqual(checkpoint.timestamp, self.state["available_at"])
        self.assertEqual(
            checkpoint.checkpoint_type, CheckpointType.LIQUIDATION_EXPOSURE
        )
        with checkpoint.file_data.open("rb") as source:
            saved = json.loads(gzip.decompress(source.read()))
        row = PerpetualStatsData.objects.get()
        self.assertEqual(row.timestamp, self.hour)
        self.assertEqual(
            json.loads(json.dumps(row.json_data["liquidation_exposure"], default=str)),
            saved["summary"],
        )
        frame = pd.DataFrame(
            [
                {
                    "timestamp": self.hour,
                    "open_interest": Decimal(20),
                    "single_open_interest": Decimal(10),
                    "long_account_ratio": Decimal("0.5"),
                    "short_account_ratio": Decimal("0.5"),
                }
            ]
        )
        PerpetualStatsData.write(
            self.symbol, 60, self.hour, self.hour + pd.Timedelta(hours=1), frame
        )
        self.assertEqual(
            json.loads(
                json.dumps(
                    PerpetualStatsData.objects.get().json_data["liquidation_exposure"],
                    default=str,
                )
            ),
            saved["summary"],
        )
        with (
            patch.object(
                snapshots,
                "get_current_time",
                return_value=self.hour + pd.Timedelta(minutes=10),
            ),
            patch.object(snapshots, "hyperliquid_snapshot") as fetch,
        ):
            snapshots.collect(self.symbol)
        fetch.assert_not_called()

    def test_failed_write_preserves_prior_checkpoint_and_removes_only_new_file(self):
        snapshots.persist(self.symbol, self.state)
        original = Checkpoint.objects.get()
        state = deepcopy(self.state)
        state["available_at"] += pd.Timedelta(hours=1)
        with (
            patch.object(
                PerpetualStatsData, "save", side_effect=RuntimeError("write failed")
            ),
            self.assertRaisesRegex(RuntimeError, "write failed"),
        ):
            snapshots.persist(self.symbol, state)
        self.assertEqual(Checkpoint.objects.count(), 1)
        self.assertEqual(PerpetualStatsData.objects.count(), 1)
        self.assertTrue(default_storage.exists(original.file_data.name))
        _, files = default_storage.listdir(str(Path(original.file_data.name).parent))
        self.assertEqual(files, [Path(original.file_data.name).name])

    def test_duplicate_in_same_hour_preserves_original_evidence(self):
        snapshots.persist(self.symbol, self.state)
        original = Checkpoint.objects.get()
        later = deepcopy(self.state)
        later["available_at"] += pd.Timedelta(minutes=20)
        later["summary"]["long"]["coverage"] = Decimal("0.1")
        with self.captureOnCommitCallbacks(execute=True):
            snapshots.persist(self.symbol, later)
        self.assertEqual(Checkpoint.objects.count(), 1)
        self.assertEqual(
            json.loads(
                json.dumps(
                    PerpetualStatsData.objects.get().json_data["liquidation_exposure"],
                    default=str,
                )
            ),
            original.json_data["summary"],
        )
        _, files = default_storage.listdir(str(Path(original.file_data.name).parent))
        self.assertEqual(files, [Path(original.file_data.name).name])

    def test_checkpoint_deletion_only_removes_file_after_commit(self):
        snapshots.persist(self.symbol, self.state)
        checkpoint = Checkpoint.objects.get()
        name = checkpoint.file_data.name
        with self.assertRaises(RuntimeError), transaction.atomic():
            checkpoint.delete()
            raise RuntimeError("rollback")
        self.assertTrue(Checkpoint.objects.filter(file_data=name).exists())
        self.assertTrue(default_storage.exists(name))
        with self.captureOnCommitCallbacks(execute=True):
            Checkpoint.objects.only("pk").delete()
            self.assertTrue(default_storage.exists(name))
        self.assertFalse(default_storage.exists(name))

    @patch.dict(os.environ, {"QUICKNODE_API_KEY": "test"})
    def test_collection_only_fetches_live_portfolios_for_current_requests(self):
        module = "quant_tick.exchanges.perpetual_stats"
        with (
            patch(
                module + ".get_current_time",
                return_value=self.hour + pd.Timedelta(minutes=5),
            ),
            patch(module + ".perpetual_stats_api", return_value=pd.DataFrame()),
            patch(
                module + ".price_context", return_value={"price_timestamp": self.hour}
            ),
            patch(module + ".collect_liquidation_snapshot") as collect,
        ):
            perpetual_stats(
                self.symbol, self.hour - pd.Timedelta(hours=1), self.hour, retry=True
            )
            collect.assert_not_called()
            perpetual_stats(self.symbol, self.hour, self.hour + pd.Timedelta(minutes=5))
            collect.assert_called_once_with(self.symbol, assert_lease_owned=None)
