from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
from django.test import SimpleTestCase

from quant_tick.constants import Exchange, SymbolType
from quant_tick.exchanges.api import exchange_candles_api
from quant_tick.exchanges.coinbase_advanced.candles import (
    coinbase_advanced_candles,
    get_coinbase_advanced_candle_url,
    get_coinbase_advanced_fetch_granularity,
)


class CoinbaseAdvancedCandleTest(SimpleTestCase):
    def test_coinbase_advanced_candles_normalizes_base_volume_as_notional(self):
        timestamp_from = datetime(2026, 5, 4, tzinfo=UTC)
        timestamp_to = datetime(2026, 5, 4, 1, tzinfo=UTC)
        data = [
            {
                "start": str(int(timestamp_from.timestamp())),
                "open": "1",
                "high": "2",
                "low": "0.5",
                "close": "1.5",
                "volume": "10",
            }
        ]

        with patch(
            "quant_tick.exchanges.coinbase_advanced.candles."
            "get_coinbase_advanced_candle_response",
            return_value=data,
        ):
            df = coinbase_advanced_candles(
                "BTC-PERP-INTX",
                timestamp_from,
                timestamp_to,
            )

        self.assertEqual(list(df.index), [pd.Timestamp(timestamp_from)])
        self.assertEqual(df.iloc[0].open, Decimal("1"))
        self.assertEqual(df.iloc[0].notional, Decimal("10"))

    def test_coinbase_advanced_candle_url_uses_inclusive_end(self):
        timestamp_from = datetime(2026, 5, 4, tzinfo=UTC)
        timestamp_to = datetime(2026, 5, 4, 0, 59, tzinfo=UTC)

        url = get_coinbase_advanced_candle_url(
            "BTC-PERP-INTX",
            timestamp_from,
            timestamp_to,
            "ONE_MINUTE",
        )
        start = int(timestamp_from.timestamp())
        end = int(timestamp_to.timestamp())
        expected = (
            "https://api.coinbase.com/api/v3/brokerage/market/products/"
            f"BTC-PERP-INTX/candles?start={start}&end={end}"
            "&granularity=ONE_MINUTE&limit=350"
        )

        self.assertEqual(url, expected)

    def test_get_coinbase_advanced_fetch_granularity_resamples_4h_from_2h(self):
        self.assertEqual(
            get_coinbase_advanced_fetch_granularity("4h"),
            (240, 120, "TWO_HOUR"),
        )

    def test_exchange_candles_api_dispatches_coinbase_advanced(self):
        symbol = SimpleNamespace(
            exchange=Exchange.COINBASE_ADVANCED,
            api_symbol="BTC-PERP-INTX",
            symbol_type=SymbolType.PERPETUAL,
            exchange_candle_resolution="1h",
            clamp_timestamp_range=lambda ts_from, ts_to: (ts_from, ts_to),
        )
        timestamp_from = datetime(2026, 5, 4, tzinfo=UTC)
        timestamp_to = datetime(2026, 5, 5, tzinfo=UTC)
        expected = pd.DataFrame([])

        with patch(
            "quant_tick.exchanges.api.coinbase_advanced_candles",
            return_value=expected,
        ) as mocked:
            result = exchange_candles_api(symbol, timestamp_from, timestamp_to)

        mocked.assert_called_once_with(
            "BTC-PERP-INTX",
            timestamp_from=timestamp_from,
            timestamp_to=timestamp_to,
            resolution="1h",
        )
        self.assertTrue(result.equals(expected))
