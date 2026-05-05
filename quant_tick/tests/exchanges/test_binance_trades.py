from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch

from django.test import SimpleTestCase

from quant_tick.constants import SymbolType
from quant_tick.exchanges.binance.constants import BINANCE_API_KEY
from quant_tick.exchanges.binance.controllers import BinanceTradesS3, binance_trades
from quant_tick.exchanges.binance.trades import get_trades


class BinanceTradesTest(SimpleTestCase):
    def test_get_trades_uses_spot_api_by_default(self):
        with patch(
            "quant_tick.exchanges.binance.trades.iter_api",
            return_value=([], False, None),
        ) as mocked:
            get_trades("BTCUSDT", datetime(2026, 4, 1, tzinfo=UTC), 1)

        url = mocked.call_args.args[0]
        self.assertTrue(url.startswith("https://api.binance.com/api/v3/"))

    def test_get_trades_uses_futures_api_for_perpetual(self):
        with patch(
            "quant_tick.exchanges.binance.trades.iter_api",
            return_value=([], False, None),
        ) as mocked:
            get_trades(
                "BTCUSDT",
                datetime(2026, 4, 1, tzinfo=UTC),
                1,
                symbol_type=SymbolType.PERPETUAL,
            )

        url = mocked.call_args.args[0]
        self.assertTrue(
            url.startswith("https://fapi.binance.com/fapi/v1/historicalTrades?")
        )

    def test_s3_uses_futures_archive_for_perpetual(self):
        controller = BinanceTradesS3.__new__(BinanceTradesS3)
        controller.symbol = SimpleNamespace(
            api_symbol="BTCUSDT",
            symbol_type=SymbolType.PERPETUAL,
        )

        url = controller.get_url(date(2026, 4, 1))

        self.assertEqual(
            url,
            "https://data.binance.vision/data/futures/um/daily/trades/"
            "BTCUSDT/BTCUSDT-trades-2026-04-01.zip",
        )

    def test_binance_futures_trades_skip_without_api_key(self):
        symbol = SimpleNamespace(api_symbol="BTCUSDT", symbol_type=SymbolType.PERPETUAL)
        timestamp_from = datetime(2026, 4, 1, tzinfo=UTC)
        timestamp_to = timestamp_from + timedelta(days=1)

        with (
            patch.dict("os.environ", {}, clear=True),
            patch("quant_tick.exchanges.binance.controllers.BinanceTradesREST") as rest,
            patch("quant_tick.exchanges.binance.controllers.BinanceTradesS3") as s3,
        ):
            binance_trades(symbol, timestamp_from, timestamp_to, lambda *args: None)

        rest.assert_not_called()
        s3.assert_not_called()

    def test_binance_futures_trades_run_with_api_key(self):
        symbol = SimpleNamespace(api_symbol="BTCUSDT", symbol_type=SymbolType.PERPETUAL)
        timestamp_from = datetime(2026, 4, 1, tzinfo=UTC)
        timestamp_to = timestamp_from + timedelta(days=1)

        with (
            patch.dict("os.environ", {BINANCE_API_KEY: "test"}, clear=True),
            patch(
                "quant_tick.exchanges.binance.controllers.use_s3",
                return_value=timestamp_from,
            ),
            patch("quant_tick.exchanges.binance.controllers.BinanceTradesREST") as rest,
            patch("quant_tick.exchanges.binance.controllers.BinanceTradesS3") as s3,
        ):
            binance_trades(symbol, timestamp_from, timestamp_to, lambda *args: None)

        rest.assert_called_once()
        rest.return_value.main.assert_called_once()
        s3.assert_not_called()
