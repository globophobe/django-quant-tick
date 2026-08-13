from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from django.test import SimpleTestCase, TestCase

from quant_tick.constants import Exchange, SymbolType
from quant_tick.exchanges.derivative_market import (
    derivative_market_data,
    derivative_market_data_api,
    derivative_market_frequency,
)
from quant_tick.models import DerivativeMarketData

from ..base import BaseSymbolTest


def binance_market_row(timestamp: datetime, index: int = 0) -> dict:
    return {
        "timestamp": timestamp,
        "open_interest": Decimal(str(100 + index)),
        "open_interest_value": Decimal(str(1_000_000 + index)),
        "top_trader_long_short_account_ratio": Decimal("1.1"),
        "top_trader_long_short_position_ratio": Decimal("1.2"),
        "long_short_account_ratio": Decimal("1.3"),
        "taker_long_short_volume_ratio": Decimal("1.4"),
    }


class DerivativeMarketAdapterTest(SimpleTestCase):
    def test_dispatches_bybit_identity_to_matching_category(self):
        timestamp_from = datetime(2026, 4, 25, tzinfo=UTC)
        timestamp_to = timestamp_from + timedelta(hours=4)
        expected = pd.DataFrame()
        symbol = SimpleNamespace(
            exchange=Exchange.BYBIT_INVERSE,
            api_symbol="BTCUSD",
            symbol_type=SymbolType.PERPETUAL,
        )

        with patch(
            "quant_tick.exchanges.derivative_market.bybit_market_history",
            return_value=expected,
        ) as mocked:
            result = derivative_market_data_api(
                symbol,
                timestamp_from,
                timestamp_to,
            )

        self.assertIs(result, expected)
        mocked.assert_called_once_with(
            "BTCUSD",
            timestamp_from,
            timestamp_to,
            category="inverse",
        )
        self.assertEqual(derivative_market_frequency(symbol), 240)

    def test_rejects_spot_symbol(self):
        symbol = SimpleNamespace(
            exchange=Exchange.BYBIT,
            api_symbol="BTCUSDT",
            symbol_type=SymbolType.SPOT,
        )

        with self.assertRaisesRegex(ValueError, "only available for perpetuals"):
            derivative_market_frequency(symbol)


class DerivativeMarketCollectionTest(BaseSymbolTest, TestCase):
    def test_collects_complete_rows_and_repairs_partial_observations(self):
        symbol = self.get_symbol(
            exchange=Exchange.BINANCE_FUTURES,
            api_symbol="BTCUSDT",
            symbol_type=SymbolType.PERPETUAL,
        )
        timestamp_from = datetime(2026, 4, 25, tzinfo=UTC)
        timestamp_to = timestamp_from + timedelta(hours=1)
        expected_timestamps = [
            timestamp_from + timedelta(minutes=5 * index) for index in range(12)
        ]
        frame = pd.DataFrame(
            [
                binance_market_row(timestamp, index)
                for index, timestamp in enumerate(expected_timestamps)
            ]
        ).set_index("timestamp")
        missing_timestamp = expected_timestamps[-1]
        partial_frame = frame.copy()
        partial_frame.loc[
            missing_timestamp,
            "top_trader_long_short_account_ratio",
        ] = None

        with patch(
            "quant_tick.exchanges.derivative_market.derivative_market_data_api",
            return_value=partial_frame,
        ) as mocked:
            derivative_market_data(symbol, timestamp_from, timestamp_to)

        mocked.assert_called_once_with(symbol, timestamp_from, timestamp_to)
        self.assertEqual(
            list(
                DerivativeMarketData.objects.filter(symbol=symbol).values_list(
                    "timestamp",
                    flat=True,
                )
            ),
            expected_timestamps[:-1],
        )

        with patch(
            "quant_tick.exchanges.derivative_market.derivative_market_data_api",
            return_value=frame.loc[[missing_timestamp]],
        ) as mocked:
            derivative_market_data(symbol, timestamp_from, timestamp_to)
        mocked.assert_called_once_with(
            symbol,
            missing_timestamp,
            missing_timestamp + timedelta(minutes=5),
        )
        self.assertEqual(
            DerivativeMarketData.objects.filter(symbol=symbol).count(),
            12,
        )

        with patch(
            "quant_tick.exchanges.derivative_market.derivative_market_data_api",
            return_value=partial_frame,
        ):
            derivative_market_data(
                symbol,
                timestamp_from,
                timestamp_to,
                retry=True,
            )
        self.assertEqual(
            DerivativeMarketData.objects.complete_for_exchange(
                symbol.exchange
            ).count(),
            12,
        )

        legacy_partial_timestamp = expected_timestamps[1]
        DerivativeMarketData.objects.filter(
            symbol=symbol,
            timestamp=legacy_partial_timestamp,
        ).update(taker_long_short_volume_ratio=None)
        with patch(
            "quant_tick.exchanges.derivative_market.derivative_market_data_api",
            return_value=frame.loc[[legacy_partial_timestamp]],
        ) as mocked:
            derivative_market_data(symbol, timestamp_from, timestamp_to)

        mocked.assert_called_once_with(
            symbol,
            legacy_partial_timestamp,
            legacy_partial_timestamp + timedelta(minutes=5),
        )
        self.assertEqual(
            DerivativeMarketData.objects.complete_for_exchange(
                symbol.exchange
            ).count(),
            12,
        )

        with patch(
            "quant_tick.exchanges.derivative_market.derivative_market_data_api"
        ) as mocked:
            derivative_market_data(symbol, timestamp_from, timestamp_to)
        mocked.assert_not_called()

    def test_bybit_waits_for_both_market_history_endpoints(self):
        symbol = self.get_symbol(
            exchange=Exchange.BYBIT_LINEAR,
            api_symbol="BTCUSDT",
            symbol_type=SymbolType.PERPETUAL,
        )
        timestamp_from = datetime(2026, 4, 25, tzinfo=UTC)
        timestamp_to = timestamp_from + timedelta(hours=4)
        frame = pd.DataFrame(
            [
                {
                    "timestamp": timestamp_from,
                    "open_interest": Decimal(100),
                    "single_open_interest": Decimal(50),
                    "long_account_ratio": Decimal("0.6"),
                    "short_account_ratio": Decimal("0.4"),
                    "long_short_account_ratio": Decimal("1.5"),
                }
            ]
        ).set_index("timestamp")
        partial_frame = frame.copy()
        partial_frame.loc[timestamp_from, "short_account_ratio"] = None

        with patch(
            "quant_tick.exchanges.derivative_market.derivative_market_data_api",
            return_value=partial_frame,
        ):
            derivative_market_data(symbol, timestamp_from, timestamp_to)
        self.assertFalse(DerivativeMarketData.objects.filter(symbol=symbol).exists())

        with patch(
            "quant_tick.exchanges.derivative_market.derivative_market_data_api",
            return_value=frame,
        ) as mocked:
            derivative_market_data(symbol, timestamp_from, timestamp_to)
        mocked.assert_called_once_with(symbol, timestamp_from, timestamp_to)
        self.assertEqual(
            DerivativeMarketData.objects.complete_for_exchange(
                symbol.exchange
            ).count(),
            1,
        )

    def test_aligns_partial_range_to_native_grid_and_complete_hour(self):
        symbol = self.get_symbol(
            exchange=Exchange.BINANCE_FUTURES,
            api_symbol="BTCUSDT",
            symbol_type=SymbolType.PERPETUAL,
        )
        requested_from = datetime(2026, 4, 25, 0, 2, tzinfo=UTC)
        requested_to = datetime(2026, 4, 25, 1, 12, tzinfo=UTC)
        expected_from = datetime(2026, 4, 25, 0, 5, tzinfo=UTC)
        expected_to = datetime(2026, 4, 25, 1, 0, tzinfo=UTC)

        with patch(
            "quant_tick.exchanges.derivative_market.derivative_market_data_api",
            return_value=pd.DataFrame(),
        ) as mocked:
            derivative_market_data(
                symbol,
                requested_from,
                requested_to,
            )

        mocked.assert_called_once_with(
            symbol,
            expected_from,
            expected_to,
        )

    def test_defers_subhour_observations_until_hour_close(self):
        symbol = self.get_symbol(
            exchange=Exchange.BINANCE_FUTURES,
            api_symbol="BTCUSDT",
            symbol_type=SymbolType.PERPETUAL,
        )
        timestamp_from = datetime(2026, 4, 25, tzinfo=UTC)
        timestamp_to = timestamp_from + timedelta(minutes=59)

        with patch(
            "quant_tick.exchanges.derivative_market.derivative_market_data_api"
        ) as mocked:
            derivative_market_data(
                symbol,
                timestamp_from,
                timestamp_to,
            )

        mocked.assert_not_called()
