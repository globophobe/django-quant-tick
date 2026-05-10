from datetime import UTC, datetime

from django.test import SimpleTestCase

from quant_tick.constants import Exchange
from quant_tick.forms import get_candle_request_data


class AggregateCandleRequestFormTest(SimpleTestCase):
    def test_get_candle_request_data_accepts_single_payload(self):
        data = get_candle_request_data(
            {
                "exchange": Exchange.COINBASE,
                "api_symbol": "BTC-USD",
                "timestamp_from": "2026-05-09T11:42:00Z",
            }
        )

        self.assertEqual(
            data,
            [
                {
                    "exchange": Exchange.COINBASE,
                    "api_symbol": "BTC-USD",
                    "timestamp_from": datetime(2026, 5, 9, 11, tzinfo=UTC),
                    "time_ago": None,
                }
            ],
        )

    def test_get_candle_request_data_accepts_request_list(self):
        data = get_candle_request_data(
            {
                "candle_requests": [
                    {
                        "exchange": Exchange.COINBASE,
                        "api_symbol": "BTC-USD",
                    },
                    {
                        "exchange": Exchange.BINANCE,
                        "api_symbol": "BTCUSDT",
                        "time_ago": "7d",
                    },
                ]
            }
        )

        self.assertEqual(data[0]["exchange"], Exchange.COINBASE)
        self.assertEqual(data[0]["api_symbol"], "BTC-USD")
        self.assertIsNone(data[0]["timestamp_from"])
        self.assertIsNone(data[0]["time_ago"])
        self.assertEqual(data[1]["exchange"], Exchange.BINANCE)
        self.assertEqual(data[1]["api_symbol"], "BTCUSDT")
        self.assertIsNone(data[1]["timestamp_from"])
        self.assertEqual(data[1]["time_ago"].days, 7)

    def test_get_candle_request_data_rejects_invalid_exchange(self):
        with self.assertRaisesRegex(ValueError, "exchange"):
            get_candle_request_data(
                {"candle_requests": [{"exchange": "not-an-exchange"}]}
            )

    def test_get_candle_request_data_rejects_non_list_requests(self):
        with self.assertRaisesRegex(ValueError, "candle_requests must be a list"):
            get_candle_request_data({"candle_requests": "not-a-list"})

    def test_get_candle_request_data_rejects_non_object_request_items(self):
        with self.assertRaisesRegex(
            ValueError,
            "candle_requests items must be objects",
        ):
            get_candle_request_data({"candle_requests": ["not-an-object"]})
