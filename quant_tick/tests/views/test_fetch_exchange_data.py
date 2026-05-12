from datetime import UTC, datetime
from unittest.mock import patch

import httpx
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
            exchange_candle_resolution="4h",
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
        self.assertEqual(funding_symbols, {"BTC", "BTCUSDT", "tBTCF0:USTF0"})
        self.assertEqual(mock_candles.call_count, 4)
        candle_symbols = {
            call.args[0].api_symbol for call in mock_candles.call_args_list
        }
        self.assertEqual(candle_symbols, {"BTC", "SOL", "BTC-USD", "tBTCF0:USTF0"})
        candle_resolutions = {
            call.kwargs["resolution"] for call in mock_candles.call_args_list
        }
        self.assertEqual(candle_resolutions, {"1d", "1h", "4h"})
        self.assertEqual(
            set(response.json()["exchanges"]),
            {
                Exchange.BINANCE_FUTURES,
                Exchange.BITFINEX,
                Exchange.COINBASE,
                Exchange.HYPERLIQUID,
            },
        )
        task_states = TaskState.objects.filter(
            task_type=TaskType.FETCH_EXCHANGE_DATA,
        ).order_by("exchange", "api_symbol")
        self.assertEqual(
            [(task_state.exchange, task_state.api_symbol) for task_state in task_states],
            [
                (Exchange.BINANCE_FUTURES, "BTCUSDT"),
                (Exchange.BITFINEX, "tBTCF0:USTF0"),
                (Exchange.COINBASE, "BTC-USD"),
                (Exchange.HYPERLIQUID, "BTC"),
                (Exchange.HYPERLIQUID, "SOL"),
            ],
        )
        for task_state in task_states:
            self.assertEqual(task_state.recent_error_count, 0)
            self.assertIsNone(task_state.locked_until)
        self.assertFalse(
            TaskState.objects.filter(
                task_type=TaskType.FETCH_EXCHANGE_DATA,
                exchange="",
            ).exists()
        )

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
    def test_get_fetches_bitfinex_perp_funding_and_exchange_candles(
        self,
        mock_funding,
        mock_candles,
    ):
        response = self.client.get(self.get_url(), {"exchange": Exchange.BITFINEX})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["funding"], 1)
        self.assertEqual(response.json()["exchange_candles"], 1)
        mock_funding.assert_called_once()
        self.assertEqual(mock_funding.call_args.args[0].api_symbol, "tBTCF0:USTF0")
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

    @patch("quant_tick.views.fetch_exchange_data.fetch_symbol_exchange_candles")
    @patch("quant_tick.views.fetch_exchange_data.fetch_symbol_funding")
    def test_get_skips_backed_off_symbol_only(self, mock_funding, mock_candles):
        TaskState.objects.create(
            task_type=TaskType.FETCH_EXCHANGE_DATA,
            exchange=Exchange.BINANCE_FUTURES,
            api_symbol="BTCUSDT",
            next_fetch_at=datetime(2099, 1, 1, tzinfo=UTC),
        )

        response = self.client.get(self.get_url())

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["skipped"], 1)
        self.assertEqual(response.json()["funding"], 2)
        funding_symbols = {
            call.args[0].api_symbol for call in mock_funding.call_args_list
        }
        self.assertEqual(funding_symbols, {"BTC", "tBTCF0:USTF0"})
        self.assertEqual(mock_candles.call_count, 4)

    @patch("quant_tick.views.fetch_exchange_data.fetch_symbol_exchange_candles")
    @patch("quant_tick.views.fetch_exchange_data.fetch_symbol_funding")
    def test_get_marks_failed_symbol_without_stopping_others(
        self,
        mock_funding,
        mock_candles,
    ):
        def funding_side_effect(symbol, *_args):
            if symbol.api_symbol == "BTC":
                raise RuntimeError("boom")

        mock_funding.side_effect = funding_side_effect

        with self.assertLogs("quant_tick.views.fetch_exchange_data", level="ERROR"):
            response = self.client.get(self.get_url())

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["ok"])
        self.assertEqual(response.json()["failed"], 1)
        self.assertEqual(response.json()["funding"], 2)
        self.assertEqual(response.json()["exchange_candles"], 3)
        failed_task = TaskState.objects.get(
            task_type=TaskType.FETCH_EXCHANGE_DATA,
            exchange=Exchange.HYPERLIQUID,
            api_symbol="BTC",
        )
        self.assertEqual(failed_task.recent_error_count, 1)
        self.assertIsNone(failed_task.locked_until)
        self.assertTrue(
            TaskState.objects.filter(
                task_type=TaskType.FETCH_EXCHANGE_DATA,
                exchange=Exchange.COINBASE,
                api_symbol="BTC-USD",
            ).exists()
        )

    @patch("quant_tick.views.fetch_exchange_data.fetch_symbol_exchange_candles")
    @patch("quant_tick.views.fetch_exchange_data.fetch_symbol_funding")
    def test_get_skips_http_530_without_backoff(
        self,
        mock_funding,
        mock_candles,
    ):
        request = httpx.Request("GET", "https://www.bitmex.com/api/v1/trade")
        response = httpx.Response(530, request=request)
        mock_funding.side_effect = httpx.HTTPStatusError(
            "Server error",
            request=request,
            response=response,
        )

        with self.assertLogs("quant_tick.views.fetch_exchange_data", level="WARNING"):
            response = self.client.get(
                self.get_url(),
                {"exchange": Exchange.BINANCE_FUTURES},
            )

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["ok"])
        self.assertEqual(response.json()["failed"], 1)
        task_state = TaskState.objects.get(
            task_type=TaskType.FETCH_EXCHANGE_DATA,
            exchange=Exchange.BINANCE_FUTURES,
            api_symbol="BTCUSDT",
        )
        self.assertEqual(task_state.recent_error_count, 0)
        self.assertIsNone(task_state.next_fetch_at)
        self.assertIsNone(task_state.locked_until)
