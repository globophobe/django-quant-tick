from django.test import SimpleTestCase

from tasks import get_workflow


class WorkflowTest(SimpleTestCase):
    def test_workflow_calls_callback_before_compaction(self):
        workflow = get_workflow(
            "https://test.123/",
            [{"exchange": "coinbase", "api_symbol": "BTC-USD"}],
            callback_url="https://test.456/callback/",
            callback_interval_minutes=15,
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
                "compact",
            ],
        )
        trade_steps = steps[1]["getTradeData"]["parallel"]["for"]["steps"][0][
            "tradeAndCandleData"
        ]["try"]["steps"]
        self.assertEqual(
            trade_steps[0]["tradeData"]["args"]["url"],
            "${item.url}",
        )
        self.assertEqual(trade_steps[0]["tradeData"]["result"], "tradeResponse")
        self.assertEqual(
            trade_steps[1]["candleData"]["args"]["url"],
            "https://test.123/aggregate-candles/",
        )
        self.assertEqual(
            trade_steps[1]["candleData"]["args"]["body"],
            "${tradeResponse.body}",
        )
        self.assertEqual(
            steps[2]["fetchExchangeData"]["try"]["args"]["url"],
            "https://test.123/fetch-exchange-data/?time_ago=7d",
        )
        self.assertEqual(steps[2]["fetchExchangeData"]["except"]["steps"], [])
        self.assertEqual(steps[3]["maybeCallback"]["next"], "compact")
        self.assertEqual(steps[4]["callback"]["next"], "compact")
        self.assertEqual(steps[4]["callback"]["args"]["body"], {"as_of": "${runTime}"})
        self.assertEqual(
            steps[5]["compact"]["args"]["url"],
            "https://test.123/compact/?time_ago=7d",
        )

    def test_workflow_compacts_without_callback(self):
        workflow = get_workflow(
            "https://test.123/",
            [{"exchange": "coinbase", "api_symbol": "BTC-USD"}],
        )
        steps = workflow["main"]["steps"]
        self.assertEqual(
            [next(iter(step)) for step in steps],
            ["getTradeData", "fetchExchangeData", "compact"],
        )
        trade_steps = steps[0]["getTradeData"]["parallel"]["for"]["steps"][0][
            "tradeAndCandleData"
        ]["try"]["steps"]
        self.assertEqual(
            trade_steps[0]["tradeData"]["args"]["url"],
            "${item.url}",
        )
        self.assertEqual(trade_steps[0]["tradeData"]["result"], "tradeResponse")
        self.assertEqual(
            trade_steps[1]["candleData"]["call"],
            "http.post",
        )
        self.assertEqual(
            trade_steps[1]["candleData"]["args"],
            {
                "url": "https://test.123/aggregate-candles/",
                "auth": {"type": "OIDC"},
                "body": "${tradeResponse.body}",
            },
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
