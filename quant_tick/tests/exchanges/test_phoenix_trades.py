from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import patch

import pandas as pd
from django.test import SimpleTestCase, TestCase, override_settings

from quant_tick.constants import Exchange, SymbolType
from quant_tick.exchanges.api import api
from quant_tick.exchanges.phoenix.trades import get_trades
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
            (
                {"data": [fill(start)], "hasMore": True, "nextCursor": "next"},
                "did not advance",
            ),
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
            with (
                self.subTest(page=page),
                patch(
                    "quant_tick.exchanges.phoenix.trades.get_phoenix_response",
                    return_value=page,
                ),
                self.assertRaisesRegex((ValueError, TypeError), message),
            ):
                get_trades("BTC", start, end)


@override_settings(
    STORAGES={"default": {"BACKEND": "django.core.files.storage.InMemoryStorage"}}
)
class PhoenixTradeCollectionTest(TestCase):
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
