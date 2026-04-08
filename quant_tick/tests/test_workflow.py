from django.test import SimpleTestCase

from tasks import get_workflow


class WorkflowTest(SimpleTestCase):
    def test_workflow_calls_callback_before_compaction(self):
        workflow = get_workflow(
            "https://test.123/",
            ["coinbase"],
            callback_url="https://test.456/callback/",
            callback_interval_minutes=15,
        )
        steps = workflow["main"]["steps"]
        self.assertEqual(
            [next(iter(step)) for step in steps],
            [
                "getRunTime",
                "getTradeData",
                "aggregateCandles",
                "maybeCallback",
                "callback",
                "compact",
            ],
        )
        self.assertEqual(
            steps[1]["getTradeData"]["parallel"]["for"]["steps"][0]["tradeData"][
                "try"
            ]["args"]["url"],
            "https://test.123/aggregate-trades/${exchange}/?time_ago=7d",
        )
        self.assertEqual(
            steps[2]["aggregateCandles"]["args"]["url"],
            "https://test.123/aggregate-candles/?time_ago=7d",
        )
        self.assertEqual(steps[3]["maybeCallback"]["next"], "compact")
        self.assertEqual(steps[4]["callback"]["next"], "compact")
        self.assertEqual(
            steps[5]["compact"]["args"]["url"],
            "https://test.123/compact/?time_ago=7d",
        )

    def test_workflow_compacts_without_callback(self):
        workflow = get_workflow("https://test.123/", ["coinbase"])
        steps = workflow["main"]["steps"]
        self.assertEqual(
            [next(iter(step)) for step in steps],
            ["getTradeData", "aggregateCandles", "compact"],
        )
        self.assertEqual(
            steps[0]["getTradeData"]["parallel"]["for"]["steps"][0]["tradeData"][
                "try"
            ]["args"]["url"],
            "https://test.123/aggregate-trades/${exchange}/?time_ago=7d",
        )
        self.assertEqual(
            steps[1]["aggregateCandles"]["args"]["url"],
            "https://test.123/aggregate-candles/?time_ago=7d",
        )
        self.assertEqual(
            steps[2]["compact"]["args"]["url"],
            "https://test.123/compact/?time_ago=7d",
        )
