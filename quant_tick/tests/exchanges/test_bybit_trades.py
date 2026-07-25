from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd
from django.test import SimpleTestCase

from quant_tick.constants import SymbolType
from quant_tick.exchanges.bybit.controllers import BybitTradesS3, bybit_trades


class BybitTradesTest(SimpleTestCase):
    def get_controller(self, api_symbol="BTCUSDT"):
        controller = BybitTradesS3.__new__(BybitTradesS3)
        controller.symbol = SimpleNamespace(
            api_symbol=api_symbol,
            symbol_type=SymbolType.PERPETUAL,
        )
        return controller

    def test_s3_uses_exact_daily_archive_url(self):
        url = self.get_controller().get_url(date(2026, 7, 23))

        self.assertEqual(
            url,
            "https://public.bybit.com/trading/BTCUSDT/"
            "BTCUSDT2026-07-23.csv.gz",
        )

    def test_trades_use_archive_without_fixed_publication_clamp(self):
        symbol = SimpleNamespace(
            api_symbol="BTCUSDT",
            symbol_type=SymbolType.PERPETUAL,
        )
        timestamp_from = datetime(2026, 7, 23, tzinfo=UTC)
        timestamp_to = timestamp_from + timedelta(days=2)
        on_data_frame = Mock()

        with patch(
            "quant_tick.exchanges.bybit.controllers.BybitTradesS3"
        ) as archive:
            bybit_trades(
                symbol,
                timestamp_from,
                timestamp_to,
                on_data_frame,
            )

        archive.assert_called_once_with(
            symbol,
            timestamp_from=timestamp_from,
            timestamp_to=timestamp_to,
            on_data_frame=on_data_frame,
            retry=False,
            verbose=False,
        )
        archive.return_value.main.assert_called_once_with()

    def test_trades_reject_spot_archive_path(self):
        symbol = SimpleNamespace(
            api_symbol="BTCUSDT",
            symbol_type=SymbolType.SPOT,
        )

        with self.assertRaisesRegex(ValueError, "perpetuals only"):
            bybit_trades(
                symbol,
                datetime(2026, 7, 23, tzinfo=UTC),
                datetime(2026, 7, 24, tzinfo=UTC),
                Mock(),
            )

    def test_linear_archive_normalizes_units_side_and_order(self):
        data = pd.DataFrame(
            [
                {
                    "timestamp": "1585180700.0647",
                    "symbol": "BTCUSDT",
                    "side": "Buy",
                    "size": "0.042",
                    "price": "6698.5",
                    "tickDirection": "MinusTick",
                    "trdMatchID": "later",
                    "grossValue": "28133700000",
                    "foreignNotional": "281.337",
                },
                {
                    "timestamp": "1585180700.0200",
                    "symbol": "BTCUSDT",
                    "side": "Sell",
                    "size": "0.072",
                    "price": "6698",
                    "tickDirection": "PlusTick",
                    "trdMatchID": "earlier",
                    "grossValue": "48225600000",
                    "foreignNotional": "482.256",
                },
            ]
        )

        parsed = self.get_controller().parse_dtypes_and_strip_columns(data)

        self.assertEqual(parsed["uid"].tolist(), ["earlier", "later"])
        self.assertEqual(parsed["tickRule"].tolist(), [-1, 1])
        self.assertEqual(
            parsed["notional"].tolist(),
            [Decimal("0.072"), Decimal("0.042")],
        )
        self.assertEqual(
            parsed["volume"].tolist(),
            [Decimal("482.256"), Decimal("281.3370")],
        )
        self.assertEqual(parsed.iloc[1]["timestamp"].microsecond, 64700)
        self.assertEqual(parsed.iloc[1]["nanoseconds"], 0)

    def test_inverse_archive_normalizes_contracts_to_base_notional(self):
        data = pd.DataFrame(
            [
                {
                    "timestamp": "1585180700",
                    "symbol": "BTCUSD",
                    "side": "Buy",
                    "size": "100",
                    "price": "10000",
                    "tickDirection": "PlusTick",
                    "trdMatchID": "trade",
                    "grossValue": "1000000",
                    "foreignNotional": "100",
                }
            ]
        )

        parsed = self.get_controller("BTCUSD").parse_dtypes_and_strip_columns(data)

        self.assertEqual(parsed.iloc[0]["volume"], Decimal("100"))
        self.assertEqual(parsed.iloc[0]["notional"], Decimal("0.01"))
