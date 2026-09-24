from datetime import UTC, datetime, timedelta
from decimal import Decimal
from unittest.mock import patch

from django.test import SimpleTestCase, TestCase

from quant_tick.constants import Exchange, SymbolType
from quant_tick.exchanges.api import funding
from quant_tick.exchanges.phoenix.funding import phoenix_funding
from quant_tick.models import FundingData, Symbol


class PhoenixFundingTest(SimpleTestCase):
    def test_chunked_hourly_percentages_become_fractional_rates(self):
        start = datetime(2026, 9, 23, tzinfo=UTC)
        end = start + timedelta(hours=3)
        with (
            patch("quant_tick.exchanges.phoenix.funding.FUNDING_MAX_RESULTS", 3),
            patch(
                "quant_tick.exchanges.phoenix.funding.get_phoenix_response",
                side_effect=[
                    {
                        "symbol": "BTC",
                        "rates": [
                            {
                                "timestamp": int(start.timestamp()),
                                "fundingRatePercentage": "0.01",
                            },
                            {
                                "timestamp": int(start.timestamp()) + 3601,
                                "fundingRatePercentage": "-0.02",
                            },
                        ],
                    },
                    {
                        "symbol": "BTC",
                        "rates": [
                            {
                                "timestamp": int(start.timestamp()) + 7200,
                                "fundingRatePercentage": "0.03",
                            },
                        ],
                    },
                ],
            ) as response,
        ):
            result = phoenix_funding("btc", start, end)
        self.assertEqual(
            list(result.index), [start + timedelta(hours=hour) for hour in range(3)]
        )
        self.assertEqual(
            result.funding_rate.tolist(),
            [Decimal("0.0001"), Decimal("-0.0002"), Decimal("0.0003")],
        )
        self.assertEqual(result.iloc[1].raw_timestamp, start + timedelta(seconds=3601))
        self.assertEqual(result.iloc[1].timestamp_offset_ms, 1000)
        self.assertFalse(result.iloc[1].timestamp_anomaly)
        params = [item.args[1] for item in response.call_args_list]
        self.assertEqual(params[0]["endTime"] + 1, params[1]["startTime"])
        self.assertEqual(params[1]["endTime"], int(end.timestamp() * 1000) - 1)

    def test_saturated_window_is_not_silently_truncated(self):
        start = datetime(2026, 9, 23, tzinfo=UTC)
        row = {"timestamp": int(start.timestamp()), "fundingRatePercentage": "0.01"}
        with (
            patch("quant_tick.exchanges.phoenix.funding.FUNDING_MAX_RESULTS", 2),
            patch(
                "quant_tick.exchanges.phoenix.funding.get_phoenix_response",
                return_value={"symbol": "BTC", "rates": [row, row]},
            ),
            self.assertRaisesRegex(ValueError, "response limit"),
        ):
            phoenix_funding("BTC", start, start + timedelta(hours=1))


class PhoenixFundingCollectionTest(TestCase):
    def test_dispatch_persists_hourly_schedule_and_raw_timestamp(self):
        start = datetime(2026, 9, 23, tzinfo=UTC)
        symbol = Symbol.objects.create(
            exchange=Exchange.PHOENIX,
            api_symbol="BTC",
            symbol_type=SymbolType.PERPETUAL,
        )
        with patch(
            "quant_tick.exchanges.phoenix.funding.get_phoenix_response",
            return_value={
                "symbol": "BTC",
                "rates": [
                    {
                        "timestamp": int(start.timestamp()) + 1,
                        "fundingRatePercentage": "0.001182",
                    }
                ],
            },
        ):
            funding(symbol, start, start + timedelta(hours=1))
        stored = FundingData.objects.get(symbol=symbol)
        self.assertEqual(stored.timestamp, start)
        self.assertEqual(stored.funding_rate, Decimal("0.00001182"))
        self.assertEqual(stored.json_data["timestamp_offset_ms"], 1000)
        symbol.refresh_from_db()
        self.assertEqual(symbol.funding_interval, timedelta(hours=1))
