from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import patch

import pandas as pd
from django.test import SimpleTestCase, TestCase, override_settings

from quant_tick.constants import RETRY_INDETERMINATE, Exchange, SymbolType
from quant_tick.exchanges.api import api
from quant_tick.exchanges.phoenix.trades import get_trades, has_trades
from quant_tick.models import Symbol, TradeData


def fill(timestamp, *, base="0.1", quote="-100", signature="same-transaction"):
    return {
        "marketSymbol": "BTC",
        "timestamp": timestamp.isoformat(),
        "baseQty": base,
        "quoteQty": quote,
        "price": "1000",
        "transactionSignature": signature,
        "instructionType": "PlaceMarketOrder",
    }


class PhoenixTradePageTest(SimpleTestCase):
    def test_rejects_invalid_or_stalled_pages(self):
        start = datetime(2026, 9, 23, tzinfo=UTC)
        end = start + timedelta(minutes=1)
        for page, message in (
            ({"data": [], "hasMore": True, "nextCursor": "next"}, "did not advance"),
            ({"data": [fill(end)], "hasMore": False}, "outside"),
            (
                {
                    "data": [fill(start), fill(start + timedelta(seconds=1))],
                    "hasMore": False,
                },
                "newest first",
            ),
            ({"data": [], "hasMore": "false"}, "malformed"),
        ):
            for query in (get_trades, has_trades):
                with (
                    self.subTest(page=page, query=query.__name__),
                    patch(
                        "quant_tick.exchanges.phoenix.trades.get_phoenix_response",
                        return_value=page,
                    ),
                    self.assertRaisesRegex((ValueError, TypeError), message),
                ):
                    query("BTC", start, end)

    def test_rejects_repeated_cursor(self):
        start = datetime(2026, 9, 23, tzinfo=UTC)
        end = start + timedelta(minutes=1)
        with (
            patch(
                "quant_tick.exchanges.phoenix.trades.get_phoenix_response",
                return_value={
                    "data": [fill(start)],
                    "hasMore": True,
                    "nextCursor": "next",
                },
            ),
            self.assertRaisesRegex(ValueError, "did not advance"),
        ):
            get_trades("BTC", start, end)


@override_settings(
    STORAGES={"default": {"BACKEND": "django.core.files.storage.InMemoryStorage"}}
)
class PhoenixTradeCollectionTest(TestCase):
    def test_backfill_crosses_empty_hours_and_stops_after_older_history_is_empty(self):
        start = datetime(2025, 11, 18, tzinfo=UTC)
        end = start + timedelta(hours=8)
        trades = [fill(start + timedelta(hours=hour, seconds=10)) for hour in (5, 2)]
        symbol = Symbol.objects.create(
            exchange=Exchange.PHOENIX,
            api_symbol="BTC",
            symbol_type=SymbolType.PERPETUAL,
            save_raw=True,
        )

        def trade_response(path, params):
            self.assertEqual(path, "/v1/trades/BTC/fills")
            rows = [
                row
                for row in trades
                if params["startTime"]
                <= int(datetime.fromisoformat(row["timestamp"]).timestamp() * 1000)
                < params["endTime"]
            ]
            return {
                "data": rows[: params["limit"]],
                "hasMore": len(rows) > params["limit"],
                "nextCursor": "older",
            }

        def candles(api_symbol, timestamp_from, timestamp_to):
            frame = pd.DataFrame(
                {"notional": Decimal(0)},
                index=pd.date_range(
                    timestamp_from, timestamp_to, freq="min", inclusive="left"
                ),
            )
            for row in trades:
                timestamp = pd.Timestamp(row["timestamp"]).floor("min")
                if timestamp in frame.index:
                    frame.loc[timestamp, "notional"] += abs(Decimal(row["baseQty"]))
            return frame

        with (
            patch(
                "quant_tick.exchanges.phoenix.controllers.phoenix_candles",
                side_effect=candles,
            ) as candle_api,
            patch(
                "quant_tick.exchanges.phoenix.trades.get_phoenix_response",
                side_effect=trade_response,
            ) as response,
        ):
            api(symbol, start, end)

        self.assertEqual(
            [item.args[1] for item in candle_api.call_args_list],
            [start + timedelta(hours=hour) for hour in range(7, 0, -1)],
        )
        probes = [
            item.args[1]
            for item in response.call_args_list
            if item.args[1]["limit"] == 1
        ]
        self.assertEqual(
            probes,
            [
                {
                    "startTime": 0,
                    "endTime": int((start + timedelta(hours=hour)).timestamp() * 1000),
                    "limit": 1,
                }
                for hour in (7, 6, 4, 3, 1)
            ],
        )
        partitions = TradeData.objects.filter(symbol=symbol)
        self.assertEqual(partitions.count(), 7)
        self.assertFalse(partitions.exclude(ok=True).exists())
        self.assertEqual(
            sum(
                (partition.json_data or {}).get("candle", {}).get("notional", 0)
                for partition in partitions
            ),
            Decimal("0.2"),
        )

    def test_empty_bounded_range_preserves_partitions_when_older_history_exists(self):
        start = datetime(2025, 11, 18, tzinfo=UTC)
        end = start + timedelta(hours=3)
        earlier = fill(start - timedelta(seconds=1))
        symbol = Symbol.objects.create(
            exchange=Exchange.PHOENIX,
            api_symbol="BTC",
            symbol_type=SymbolType.PERPETUAL,
        )
        candles = pd.DataFrame(
            {"notional": Decimal(0)},
            index=pd.date_range(start, end, freq="min", inclusive="left"),
        )

        def trade_response(path, params):
            return {
                "data": [earlier] if params["startTime"] == 0 else [],
                "hasMore": False,
            }

        with (
            patch(
                "quant_tick.exchanges.phoenix.controllers.phoenix_candles",
                return_value=candles,
            ),
            patch(
                "quant_tick.exchanges.phoenix.trades.get_phoenix_response",
                side_effect=trade_response,
            ),
        ):
            api(symbol, start, end)

        partitions = TradeData.objects.filter(symbol=symbol)
        self.assertEqual(partitions.count(), 3)
        self.assertFalse(partitions.exclude(ok=True).exists())

    def test_retry_scans_all_requested_empty_partitions(self):
        start = datetime(2025, 11, 18, tzinfo=UTC)
        end = start + timedelta(hours=3)
        symbol = Symbol.objects.create(
            exchange=Exchange.PHOENIX,
            api_symbol="BTC",
            symbol_type=SymbolType.PERPETUAL,
        )
        for retry in (True, RETRY_INDETERMINATE):
            with (
                self.subTest(retry=retry),
                patch(
                    "quant_tick.exchanges.phoenix.controllers.phoenix_candles",
                    return_value=pd.DataFrame([]),
                ),
                patch(
                    "quant_tick.exchanges.phoenix.trades.get_phoenix_response",
                    return_value={"data": [], "hasMore": False},
                ) as response,
            ):
                api(symbol, start, end, retry=retry)
            self.assertEqual(response.call_count, 3)
            self.assertTrue(
                all(item.args[1]["limit"] == 1000 for item in response.call_args_list)
            )

    def test_collection_preserves_same_transaction_fills_across_start_boundary_pages(
        self,
    ):
        start = datetime(2026, 9, 23, tzinfo=UTC)
        end = start + timedelta(minutes=1)
        symbol = Symbol.objects.create(
            exchange=Exchange.PHOENIX,
            api_symbol="BTC",
            symbol_type=SymbolType.PERPETUAL,
            save_raw=True,
            save_aggregated=True,
        )
        candles = pd.DataFrame({"notional": [Decimal("0.3")]}, index=[start])
        with (
            patch(
                "quant_tick.controllers.rest.TradeDataIterator.iter_all",
                return_value=[(start, end)],
            ),
            patch(
                "quant_tick.exchanges.phoenix.controllers.phoenix_candles",
                return_value=candles,
            ),
            patch(
                "quant_tick.exchanges.phoenix.trades.get_phoenix_response",
                side_effect=[
                    {
                        "data": [fill(start, base="-0.1", quote="100"), fill(start)],
                        "hasMore": True,
                        "nextCursor": "next",
                    },
                    {"data": [fill(start)], "hasMore": False},
                ],
            ) as response,
        ):
            api(symbol, start, end)

        self.assertEqual(response.call_count, 2)
        first_params, next_params = [item.args[1] for item in response.call_args_list]
        self.assertEqual(next_params, first_params | {"cursor": "next"})
        self.assertEqual(first_params["startTime"], int(start.timestamp() * 1000))
        self.assertEqual(first_params["endTime"], int(end.timestamp() * 1000))
        partition = TradeData.objects.get(symbol=symbol)
        self.assertTrue(partition.ok)
        raw = partition.get_data_frame("raw_data")
        self.assertEqual(len(raw), 3)
        self.assertEqual(raw.tickRule.tolist(), [1, 1, -1])
        self.assertEqual(raw.notional.sum(), Decimal("0.3"))
        self.assertEqual(raw.volume.sum(), Decimal(300))
