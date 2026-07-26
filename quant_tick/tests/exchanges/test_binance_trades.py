from datetime import UTC, date, datetime
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd
from django.test import SimpleTestCase

from quant_tick.exchanges.binance.controllers import (
    BinanceTradesS3,
    binance_trades,
)
from quant_tick.exchanges.binance.trades import get_trades


class BinanceTradesTest(SimpleTestCase):
    def test_trades_probe_archives_then_fill_missing_ranges_with_rest(self):
        symbol = SimpleNamespace(api_symbol="BTCUSDT")
        timestamp_from = datetime(2026, 4, 1, tzinfo=UTC)
        timestamp_to = datetime(2026, 4, 4, tzinfo=UTC)
        on_data_frame = Mock()
        calls = []

        with (
            patch(
                "quant_tick.exchanges.binance.controllers.BinanceTradesS3"
            ) as archive,
            patch(
                "quant_tick.exchanges.binance.controllers.BinanceTradesREST"
            ) as rest,
        ):
            archive.return_value.main.side_effect = lambda: calls.append("archive")
            rest.return_value.main.side_effect = lambda: calls.append("rest")
            binance_trades(
                symbol,
                timestamp_from,
                timestamp_to,
                on_data_frame,
            )

        expected_kwargs = {
            "timestamp_from": timestamp_from,
            "timestamp_to": timestamp_to,
            "on_data_frame": on_data_frame,
            "retry": False,
            "verbose": False,
        }
        archive.assert_called_once_with(symbol, **expected_kwargs)
        rest.assert_called_once_with(symbol, **expected_kwargs)
        self.assertEqual(calls, ["archive", "rest"])

    def test_get_trades_uses_spot_raw_trade_api(self):
        with patch(
            "quant_tick.exchanges.binance.trades.iter_api",
            return_value=([], False, None),
        ) as mocked:
            get_trades("BTCUSDT", datetime(2026, 4, 1, tzinfo=UTC), 1)

        url = mocked.call_args.args[0]
        self.assertTrue(
            url.startswith("https://api.binance.com/api/v3/historicalTrades?")
        )

    def test_s3_uses_spot_raw_trade_archive(self):
        controller = BinanceTradesS3.__new__(BinanceTradesS3)
        controller.symbol = SimpleNamespace(api_symbol="BTCUSDT")

        url = controller.get_url(date(2026, 4, 1))

        self.assertEqual(
            url,
            "https://data.binance.vision/data/spot/daily/trades/"
            "BTCUSDT/BTCUSDT-trades-2026-04-01.zip",
        )

    def test_s3_tick_rule_maps_is_buyer_maker_values(self):
        controller = BinanceTradesS3.__new__(BinanceTradesS3)
        data = pd.DataFrame(
            [
                {
                    "id": "id",
                    "price": "price",
                    "qty": "qty",
                    "quoteQty": "quote_qty",
                    "time": "time",
                    "isBuyerMaker": "is_buyer_maker",
                    "isBestMatch": None,
                },
                {
                    "id": "1",
                    "price": "100",
                    "qty": "2",
                    "quoteQty": "200",
                    "time": "1775606400000",
                    "isBuyerMaker": True,
                    "isBestMatch": True,
                },
                {
                    "id": "2",
                    "price": "101",
                    "qty": "3",
                    "quoteQty": "303",
                    "time": "1775606401000",
                    "isBuyerMaker": "True",
                    "isBestMatch": True,
                },
                {
                    "id": "3",
                    "price": "102",
                    "qty": "4",
                    "quoteQty": "408",
                    "time": "1775606402000",
                    "isBuyerMaker": False,
                    "isBestMatch": True,
                },
                {
                    "id": "4",
                    "price": "103",
                    "qty": "5",
                    "quoteQty": "515",
                    "time": "1775606403000",
                    "isBuyerMaker": "False",
                    "isBestMatch": True,
                },
            ]
        )

        parsed = controller.parse_dtypes_and_strip_columns(data)

        self.assertEqual(parsed["tickRule"].tolist(), [-1, -1, 1, 1])
