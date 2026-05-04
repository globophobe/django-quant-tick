from datetime import UTC, datetime
from unittest.mock import patch

from django.test import TestCase
from django.urls import reverse

from quant_tick.constants import Exchange, SymbolType, TaskType
from quant_tick.models import Symbol, TaskState


class FetchExchangeDataViewTest(TestCase):
    def setUp(self):
        super().setUp()
        Symbol.objects.create(
            exchange=Exchange.HYPERLIQUID,
            api_symbol="BTC",
            symbol_type=SymbolType.PERPETUAL,
            exchange_candle_resolution="8h",
        )
        Symbol.objects.create(
            exchange=Exchange.HYPERLIQUID,
            api_symbol="ETH",
            symbol_type=SymbolType.SPOT,
        )
        Symbol.objects.create(
            exchange=Exchange.HYPERLIQUID,
            api_symbol="SOL",
            symbol_type=SymbolType.SPOT,
            exchange_candle_resolution="1h",
        )
        Symbol.objects.create(
            exchange=Exchange.BITFINEX,
            api_symbol="tBTCF0:USTF0",
            symbol_type=SymbolType.PERPETUAL,
            exchange_candle_resolution="1h",
        )
        Symbol.objects.create(
            exchange=Exchange.BINANCE_FUTURES,
            api_symbol="BTCUSDT",
            symbol_type=SymbolType.PERPETUAL,
        )
        Symbol.objects.create(
            exchange=Exchange.COINBASE,
            api_symbol="BTC-USD",
            symbol_type=SymbolType.SPOT,
            exchange_candle_resolution="1d",
        )
        Symbol.objects.create(
            exchange=Exchange.COINBASE_ADVANCED,
            api_symbol="BTC-PERP-INTX",
            symbol_type=SymbolType.PERPETUAL,
        )

    def get_url(self) -> str:
        return reverse("fetch_exchange_data")

    @patch("quant_tick.views.fetch_exchange_data.fetch_symbol_exchange_candles")
    @patch("quant_tick.views.fetch_exchange_data.fetch_symbol_funding")
    def test_get_fetches_supported_exchange_data(self, mock_funding, mock_candles):
        response = self.client.get(self.get_url())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["funding"], 3)
        self.assertEqual(response.json()["exchange_candles"], 4)
        self.assertEqual(mock_funding.call_count, 3)
        funding_symbols = {
            call.args[0].api_symbol for call in mock_funding.call_args_list
        }
        self.assertEqual(funding_symbols, {"BTC", "BTC-PERP-INTX", "BTCUSDT"})
        self.assertEqual(mock_candles.call_count, 4)
        candle_symbols = {
            call.args[0].api_symbol for call in mock_candles.call_args_list
        }
        self.assertEqual(candle_symbols, {"BTC", "SOL", "BTC-USD", "tBTCF0:USTF0"})
        candle_resolutions = {
            call.kwargs["resolution"] for call in mock_candles.call_args_list
        }
        self.assertEqual(candle_resolutions, {"1d", "1h", "8h"})
        self.assertEqual(
            set(response.json()["exchanges"]),
            {
                Exchange.BINANCE_FUTURES,
                Exchange.BITFINEX,
                Exchange.COINBASE,
                Exchange.COINBASE_ADVANCED,
                Exchange.HYPERLIQUID,
            },
        )
        task_state = TaskState.objects.get(
            task_type=TaskType.FETCH_EXCHANGE_DATA,
            exchange="",
        )
        self.assertEqual(task_state.recent_error_count, 0)
        self.assertIsNone(task_state.locked_until)

    @patch("quant_tick.views.fetch_exchange_data.fetch_symbol_exchange_candles")
    @patch("quant_tick.views.fetch_exchange_data.fetch_symbol_funding")
    def test_get_fetches_funding_only_for_binance_futures(
        self,
        mock_funding,
        mock_candles,
    ):
        response = self.client.get(
            self.get_url(),
            {"exchange": Exchange.BINANCE_FUTURES},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["funding"], 1)
        self.assertEqual(response.json()["exchange_candles"], 0)
        mock_funding.assert_called_once()
        mock_candles.assert_not_called()

    @patch("quant_tick.views.fetch_exchange_data.fetch_symbol_exchange_candles")
    @patch("quant_tick.views.fetch_exchange_data.fetch_symbol_funding")
    def test_get_fetches_bitfinex_perp_exchange_candles(
        self,
        mock_funding,
        mock_candles,
    ):
        response = self.client.get(self.get_url(), {"exchange": Exchange.BITFINEX})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["funding"], 0)
        self.assertEqual(response.json()["exchange_candles"], 1)
        mock_funding.assert_not_called()
        mock_candles.assert_called_once()
        self.assertEqual(mock_candles.call_args.args[0].api_symbol, "tBTCF0:USTF0")
        self.assertEqual(mock_candles.call_args.kwargs["resolution"], "1h")

    @patch("quant_tick.views.fetch_exchange_data.fetch_symbol_exchange_candles")
    @patch("quant_tick.views.fetch_exchange_data.fetch_symbol_funding")
    def test_get_fetches_spot_exchange_candles(self, mock_funding, mock_candles):
        response = self.client.get(self.get_url(), {"exchange": Exchange.COINBASE})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["funding"], 0)
        self.assertEqual(response.json()["exchange_candles"], 1)
        mock_funding.assert_not_called()
        mock_candles.assert_called_once()
        self.assertEqual(mock_candles.call_args.args[0].api_symbol, "BTC-USD")
        self.assertEqual(mock_candles.call_args.kwargs["resolution"], "1d")

    @patch("quant_tick.views.fetch_exchange_data.fetch_symbol_funding")
    def test_get_skips_when_task_is_backed_off(self, mock_funding):
        TaskState.objects.create(
            task_type=TaskType.FETCH_EXCHANGE_DATA,
            next_fetch_at=datetime(2099, 1, 1, tzinfo=UTC),
        )

        response = self.client.get(self.get_url())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["skipped"], "backoff")
        mock_funding.assert_not_called()
