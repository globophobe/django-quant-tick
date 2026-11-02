from datetime import UTC, datetime, timedelta
from decimal import Decimal
from io import StringIO
from unittest.mock import patch

from django.core.management import call_command
from django.test import SimpleTestCase, TestCase

from quant_tick.constants import Exchange, SymbolType
from quant_tick.exchanges.aster.funding import aster_funding
from quant_tick.models import FundingData, Symbol

START = datetime(2021, 8, 27, tzinfo=UTC)
MS = int(START.timestamp() * 1000)


def payment(timestamp, rate="0.0001", symbol="BTCUSDT"):
    return {"symbol": symbol, "fundingTime": timestamp, "fundingRate": rate}


class AsterFundingTest(SimpleTestCase):
    def test_nonstandard_interval_and_exclusive_end_preserve_settled_rates(self):
        points = [payment(MS + 2), payment(MS + 14400000, "-0.0002")]
        with patch(
            "quant_tick.exchanges.aster.funding.get_aster_response", return_value=points
        ) as request:
            frame = aster_funding(
                "btcusdt",
                START,
                START + timedelta(hours=8),
                funding_interval=timedelta(hours=4),
            )
        self.assertEqual(list(frame.index), [START, START + timedelta(hours=4)])
        self.assertEqual(
            frame.funding_rate.tolist(), [Decimal("0.0001"), Decimal("-0.0002")]
        )
        self.assertEqual(frame.iloc[0].timestamp_offset_ms, 2)
        self.assertEqual(frame.iloc[0].raw_timestamp, START + timedelta(milliseconds=2))
        self.assertEqual(request.call_args.args[1]["endTime"], MS + 28800000 - 1)

    def test_rejects_wrong_market_duplicate_payments_and_incompatible_settlement(self):
        cases = (
            ([payment(MS, symbol="ETHUSDT")], "unexpected market"),
            ([payment(MS), payment(MS)], "timestamp"),
            ([payment(MS + 3600000)], "funding interval"),
            ([payment(MS, rate="NaN")], "not finite"),
        )
        for points, message in cases:
            with (
                self.subTest(points=points),
                patch(
                    "quant_tick.exchanges.aster.funding.get_aster_response",
                    return_value=points,
                ),
                self.assertRaisesRegex(ValueError, message),
            ):
                aster_funding("BTCUSDT", START, START + timedelta(hours=8))


class AsterFundingCollectionTest(TestCase):
    def test_inactive_symbol_backfill_pages_then_repairs_a_missing_payment(self):
        symbol = Symbol.objects.create(
            exchange=Exchange.ASTER,
            api_symbol="BTCUSDT",
            symbol_type=SymbolType.PERPETUAL,
            is_active=False,
        )
        points = [
            payment(MS + 2),
            payment(MS + 28800000, "-0.0002"),
            payment(MS + 57600004, "0"),
        ]
        requests = []

        def response(path, params):
            if path == "/fundingInfo":
                return [{"symbol": "BTCUSDT", "fundingIntervalHours": 8}]
            self.assertEqual(path, "/fundingRate")
            requests.append(params)
            return [
                point
                for point in points
                if params["startTime"] <= point["fundingTime"] <= params["endTime"]
            ][: params["limit"]]

        def collect():
            call_command(
                "funding",
                "--code-name",
                symbol.code_name,
                "--date-from",
                "2021-08-27",
                "--date-to",
                "2021-08-28",
                stdout=StringIO(),
            )

        with (
            patch(
                "quant_tick.exchanges.aster.funding.get_aster_response",
                side_effect=response,
            ),
            patch("quant_tick.exchanges.aster.funding.FUNDING_MAX_RESULTS", 2),
        ):
            collect()
            self.assertEqual(len(requests), 2)
            self.assertEqual(requests[1]["startTime"], MS + 28800001)
            rows = list(FundingData.objects.filter(symbol=symbol))
            self.assertEqual(
                [row.timestamp for row in rows],
                [START + timedelta(hours=hour) for hour in (0, 8, 16)],
            )
            self.assertEqual(
                [row.funding_rate for row in rows],
                [Decimal("0.0001"), Decimal("-0.0002"), Decimal(0)],
            )
            self.assertEqual(rows[-1].json_data["timestamp_offset_ms"], 4)
            symbol.refresh_from_db()
            self.assertEqual(symbol.funding_interval, timedelta(hours=8))
            collect()
            self.assertEqual(len(requests), 2)
            rows[1].delete()
            collect()
            repaired = FundingData.objects.get(
                symbol=symbol, timestamp=START + timedelta(hours=8)
            )
            self.assertEqual(repaired.funding_rate, Decimal("-0.0002"))
            self.assertEqual(FundingData.objects.filter(symbol=symbol).count(), 3)
            self.assertEqual(requests[-1]["startTime"], MS + 28800000)
