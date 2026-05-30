import os
from unittest.mock import Mock, patch

from django.test import SimpleTestCase, TestCase

from quant_tick.constants import Exchange
from quant_tick.models import Symbol

from tasks import _callback_condition, get_workflow, push_workflow


class WorkflowTest(SimpleTestCase):
    def test_workflow_calls_callback_before_compaction(self):
        workflow = get_workflow(
            "https://test.123/",
            [{"exchange": "coinbase", "api_symbol": "BTC-USD"}],
            callback_url="https://test.456/callback/",
            callback_window_period_minutes=15,
            callback_window_duration_minutes=5,
            callback_strategies=["callback-target-a", "callback-target-b"],
        )
        steps = workflow["main"]["steps"]
        self.assertEqual(
            [next(iter(step)) for step in steps],
            [
                "getRunTime",
                "getTradeData",
                "fetchExchangeData",
                "maybeCallback",
                "callback",
                "notify",
                "compact",
            ],
        )
        trade_step = steps[1]["getTradeData"]["parallel"]["for"]["steps"][0][
            "tradeData"
        ]["try"]
        self.assertEqual(trade_step["call"], "http.get")
        self.assertEqual(trade_step["args"]["url"], "${item.url}")
        self.assertEqual(
            steps[2]["fetchExchangeData"]["try"]["args"]["url"],
            "https://test.123/fetch-exchange-data/?time_ago=7d",
        )
        self.assertEqual(steps[2]["fetchExchangeData"]["except"]["steps"], [])
        self.assertEqual(steps[3]["maybeCallback"]["next"], "compact")
        self.assertEqual(steps[4]["callback"]["next"], "notify")
        callback_fanout = steps[4]["callback"]["parallel"]["for"]
        self.assertEqual(callback_fanout["value"], "callbackTarget")
        self.assertEqual(
            callback_fanout["in"],
            [
                {"url": "https://test.456/callback/callback-target-a/"},
                {"url": "https://test.456/callback/callback-target-b/"},
            ],
        )
        callback_step = callback_fanout["steps"][0]["callbackStrategy"]
        self.assertEqual(callback_step["args"]["url"], "${callbackTarget.url}")
        self.assertEqual(
            callback_step["args"]["body"],
            {
                "as_of": "${runTime}",
                "final_retry": "${runMinutes % 15 == 5}",
            },
        )
        self.assertEqual(steps[5]["notify"]["args"]["url"], "https://test.456/notify/")
        self.assertEqual(steps[5]["notify"]["next"], "compact")
        self.assertEqual(
            steps[6]["compact"]["args"]["url"],
            "https://test.123/compact/?time_ago=7d",
        )


    def test_callback_condition_uses_window_only(self):
        condition = _callback_condition(
            callback_window_period_minutes=15,
            callback_window_duration_minutes=5,
        )

        self.assertEqual(condition, "runMinutes % 15 <= 5")

    def test_callback_condition_rejects_missing_or_invalid_window(self):
        with self.assertRaisesRegex(ValueError, "required when CALLBACK_URL is set"):
            _callback_condition(callback_window_period_minutes=15)
        with self.assertRaisesRegex(ValueError, "must be <="):
            _callback_condition(
                callback_window_period_minutes=10,
                callback_window_duration_minutes=20,
            )

    def test_workflow_requires_callback_strategies_when_callback_url_is_set(self):
        with self.assertRaisesRegex(ValueError, "CALLBACK_STRATEGIES"):
            get_workflow(
                "https://test.123/",
                [{"exchange": "coinbase", "api_symbol": "BTC-USD"}],
                callback_url="https://test.456/callback/",
                callback_window_period_minutes=15,
                callback_window_duration_minutes=5,
            )

    def test_workflow_collects_trades(self):
        workflow = get_workflow(
            "https://test.123/",
            [{"exchange": "coinbase", "api_symbol": "BTC-USD"}],
        )
        steps = workflow["main"]["steps"]
        self.assertEqual(
            [next(iter(step)) for step in steps],
            ["getTradeData", "fetchExchangeData", "compact"],
        )
        trade_step = steps[0]["getTradeData"]["parallel"]["for"]["steps"][0][
            "tradeData"
        ]["try"]
        self.assertEqual(
            trade_step["args"]["url"],
            "${item.url}",
        )
        self.assertEqual(trade_step["call"], "http.get")
        self.assertEqual(
            trade_step["args"]["auth"],
            {"type": "OIDC"},
        )
        self.assertEqual(
            steps[1]["fetchExchangeData"]["try"]["args"]["url"],
            "https://test.123/fetch-exchange-data/?time_ago=7d",
        )
        self.assertEqual(steps[1]["fetchExchangeData"]["except"]["steps"], [])
        self.assertEqual(
            steps[2]["compact"]["args"]["url"],
            "https://test.123/compact/?time_ago=7d",
        )

    def test_workflow_fanout_is_symbol_scoped(self):
        workflow = get_workflow(
            "https://test.123/",
            [
                {"exchange": "coinbase", "api_symbol": "BTC-USD"},
                {"exchange": "binance-futures", "api_symbol": "BTC/USDT"},
            ],
        )
        requests = workflow["main"]["steps"][0]["getTradeData"]["parallel"]["for"][
            "in"
        ]

        self.assertEqual(
            requests,
            [
                {
                    "url": "https://test.123/aggregate-trades/coinbase/?time_ago=7d&api_symbol=BTC-USD"
                },
                {
                    "url": "https://test.123/aggregate-trades/binance-futures/?time_ago=7d&api_symbol=BTC%2FUSDT"
                },
            ],
        )


class WorkflowDeployTest(TestCase):
    @patch.dict(
        os.environ,
        {
            "PRODUCTION_API_URL": "https://test.123/",
            "CALLBACK_URL": "",
            "CALLBACK_WINDOW_PERIOD_MINUTES": "",
            "CALLBACK_WINDOW_DURATION_MINUTES": "",
        },
    )
    @patch("tasks.django_settings")
    @patch("tasks.get_workflow", return_value={"main": {"steps": []}})
    def test_push_workflow_only_fans_out_trade_data_symbols(
        self,
        mock_get_workflow,
        _mock_django_settings,
    ):
        raw = Symbol.objects.create(
            exchange=Exchange.COINBASE,
            api_symbol="BTC-USD",
            save_raw=True,
        )
        aggregated = Symbol.objects.create(
            exchange=Exchange.BITFINEX,
            api_symbol="tBTCUSD",
            save_aggregated=True,
        )
        filtered = Symbol.objects.create(
            exchange=Exchange.BITMEX,
            api_symbol="XBTUSD",
            significant_trade_filter=1000,
        )
        Symbol.objects.create(
            exchange=Exchange.BINANCE_FUTURES,
            api_symbol="BTCUSDT",
            exchange_candle_resolution="1h",
        )
        Symbol.objects.create(
            exchange=Exchange.HYPERLIQUID,
            api_symbol="BTC",
        )
        Symbol.objects.create(
            exchange=Exchange.BINANCE,
            api_symbol="BTCUSDT",
            save_raw=True,
            is_active=False,
        )

        push_workflow.body(Mock())

        self.assertEqual(
            mock_get_workflow.call_args.args[1],
            [
                {"exchange": aggregated.exchange, "api_symbol": aggregated.api_symbol},
                {"exchange": filtered.exchange, "api_symbol": filtered.api_symbol},
                {"exchange": raw.exchange, "api_symbol": raw.api_symbol},
            ],
        )

    @patch.dict(
        os.environ,
        {
            "PRODUCTION_API_URL": "https://test.123/",
            "CALLBACK_URL": "https://test.456/callback/",
            "CALLBACK_STRATEGIES": "callback-target-a, callback-target-b",
            "CALLBACK_WINDOW_PERIOD_MINUTES": "15",
            "CALLBACK_WINDOW_DURATION_MINUTES": "5",
        },
    )
    @patch("tasks.django_settings")
    @patch("tasks.get_workflow", return_value={"main": {"steps": []}})
    def test_push_workflow_passes_callback_window_settings(
        self,
        mock_get_workflow,
        _mock_django_settings,
    ):
        Symbol.objects.create(
            exchange=Exchange.COINBASE,
            api_symbol="BTC-USD",
            save_raw=True,
        )

        push_workflow.body(Mock())

        self.assertEqual(mock_get_workflow.call_args.kwargs["callback_url"], "https://test.456/callback/")
        self.assertEqual(
            mock_get_workflow.call_args.kwargs["callback_strategies"],
            ["callback-target-a", "callback-target-b"],
        )
        self.assertEqual(mock_get_workflow.call_args.kwargs["callback_window_period_minutes"], 15)
        self.assertEqual(mock_get_workflow.call_args.kwargs["callback_window_duration_minutes"], 5)
