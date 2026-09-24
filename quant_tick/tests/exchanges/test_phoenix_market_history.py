from datetime import UTC, datetime, timedelta
from decimal import Decimal
from io import StringIO
from unittest.mock import patch

from django.core.management import call_command
from django.test import SimpleTestCase, TestCase

from quant_tick.constants import Exchange, SymbolType
from quant_tick.exchanges.perpetual_stats import perpetual_stats
from quant_tick.exchanges.phoenix.market_history import phoenix_market_history
from quant_tick.models import PerpetualStatsData, Symbol


def observation(timestamp, **overrides):
    return {
        "timestamp": timestamp.isoformat(),
        "open_interest": Decimal("24.4222"),
        "mark_price": Decimal("84587.0"),
        "spot_price": Decimal("84563.0"),
        "total_maker_fees": Decimal("15009.333919"),
        "total_taker_fees": Decimal("224907.999017"),
        "slot": 449845940,
    } | overrides


def response(points, **overrides):
    return {
        "market_id": 2,
        "symbol": "BTC",
        "timeframe": "1h",
        "stats": points,
    } | overrides


class PhoenixMarketHistoryTest(SimpleTestCase):
    def setUp(self):
        self.start = datetime(2026, 9, 23, 23, tzinfo=UTC)
        self.api = self.enterContext(
            patch("quant_tick.exchanges.phoenix.market_history.get_phoenix_response")
        )

    def test_base_open_interest_and_mark_value_preserve_source_metadata(self):
        self.api.return_value = response([observation(self.start)])
        result = phoenix_market_history(
            "btc", self.start, self.start + timedelta(hours=1)
        )
        self.assertEqual(list(result.index), [self.start])
        row = result.iloc[0]
        self.assertEqual(row.open_interest, Decimal("24.4222"))
        self.assertEqual(row.open_interest_value, Decimal("2065800.6314"))
        self.assertEqual(row.open_interest_unit, "base_asset")
        self.assertEqual(row.mark_price, Decimal(84587))
        self.assertEqual(row.spot_price, Decimal(84563))
        self.assertEqual(row.slot, 449845940)
        self.assertEqual(row.market_history_timestamp_convention, "interval_start")
        self.assertNotIn("long_short_account_ratio", result.columns)
        self.api.assert_called_once_with(
            "/v1/market/BTC/stats",
            {
                "start_time": self.start.isoformat(),
                "end_time": (
                    self.start + timedelta(hours=1, microseconds=-1)
                ).isoformat(),
                "timeframe": "1h",
                "limit": 1000,
            },
        )

    def test_aligned_chunks_preserve_missing_hours_without_filling(self):
        timestamps = [self.start + timedelta(hours=hour) for hour in (1, 2, 4)]
        self.api.side_effect = [
            response([observation(timestamp) for timestamp in timestamps[:2]]),
            response([observation(timestamps[2], open_interest=0)]),
        ]
        with patch(
            "quant_tick.exchanges.phoenix.market_history.MARKET_HISTORY_MAX_RESULTS", 3
        ):
            result = phoenix_market_history(
                "BTC",
                self.start + timedelta(minutes=1),
                self.start + timedelta(hours=5, minutes=20),
            )
        self.assertEqual(list(result.index), timestamps)
        self.assertEqual(result.iloc[-1].open_interest, Decimal(0))
        self.assertEqual(result.iloc[-1].open_interest_value, Decimal(0))
        params = [item.args[1] for item in self.api.call_args_list]
        self.assertEqual(params[0]["start_time"], timestamps[0].isoformat())
        self.assertEqual(
            datetime.fromisoformat(params[0]["end_time"]) + timedelta(microseconds=1),
            datetime.fromisoformat(params[1]["start_time"]),
        )
        self.assertEqual(
            params[1]["end_time"],
            (self.start + timedelta(hours=5, microseconds=-1)).isoformat(),
        )

    def test_unfinished_hour_does_not_fetch_or_create_an_observation(self):
        result = phoenix_market_history(
            "BTC", self.start, self.start + timedelta(minutes=59)
        )
        self.assertTrue(result.empty)
        self.assertIn("open_interest", result.columns)
        self.api.assert_not_called()

    def test_rejects_response_mismatches_and_invalid_points(self):
        end = self.start + timedelta(hours=2)
        cases = [
            (response([], symbol="SOL"), "market or timeframe"),
            (response([], timeframe="1d"), "market or timeframe"),
            (response([observation(end)]), "outside"),
            (
                response([observation(self.start + timedelta(seconds=1))]),
                "hourly buckets",
            ),
            (
                response([observation(self.start), observation(self.start)]),
                "hourly buckets",
            ),
            (
                response(
                    [
                        observation(self.start + timedelta(hours=1)),
                        observation(self.start),
                    ]
                ),
                "hourly buckets",
            ),
            (
                response([observation(self.start, open_interest=-1)]),
                "invalid quantities",
            ),
            (
                response([observation(self.start, mark_price="NaN")]),
                "invalid quantities",
            ),
        ]
        for payload, message in cases:
            with self.subTest(payload=payload):
                self.api.return_value = payload
                with self.assertRaisesRegex(ValueError, message):
                    phoenix_market_history("BTC", self.start, end)

    def test_response_limit_cannot_silently_truncate_history(self):
        self.api.return_value = response([observation(self.start)] * 2)
        with (
            patch(
                "quant_tick.exchanges.phoenix.market_history.MARKET_HISTORY_MAX_RESULTS",
                2,
            ),
            self.assertRaisesRegex(ValueError, "response limit"),
        ):
            phoenix_market_history("BTC", self.start, self.start + timedelta(hours=1))


class PhoenixMarketHistoryCollectionTest(TestCase):
    def test_management_collection_preserves_gaps_and_repairs_them(self):
        symbol = Symbol.objects.create(
            exchange=Exchange.PHOENIX,
            api_symbol="BTC",
            symbol_type=SymbolType.PERPETUAL,
        )
        start = datetime(2026, 9, 23, tzinfo=UTC)
        end = start + timedelta(hours=3, minutes=15)
        points = [observation(start + timedelta(hours=hour)) for hour in (0, 2)]
        with patch(
            "quant_tick.exchanges.phoenix.market_history.get_phoenix_response",
            return_value=response(points),
        ):
            call_command(
                "perpetual_stats",
                "--exchange",
                "phoenix",
                "--api-symbol",
                "BTC",
                "--date-from",
                "2026-09-23",
                "--date-to",
                "2026-09-23",
                "--time-to",
                "03:15",
                stdout=StringIO(),
            )

        rows = PerpetualStatsData.objects.filter(symbol=symbol)
        self.assertEqual(rows.count(), 2)
        self.assertEqual(rows.complete_for_exchange(Exchange.PHOENIX).count(), 2)
        first = rows.first()
        self.assertEqual(first.frequency, 60)
        self.assertEqual(first.open_interest, Decimal("24.4222"))
        self.assertEqual(first.open_interest_value, Decimal("2065800.6314"))
        self.assertEqual(first.json_data["slot"], 449845940)
        self.assertEqual(
            first.json_data["market_history_timestamp_convention"], "interval_start"
        )
        self.assertIsNone(first.long_short_account_ratio)

        missing = start + timedelta(hours=1)
        with patch(
            "quant_tick.exchanges.phoenix.market_history.get_phoenix_response",
            return_value=response([observation(missing)]),
        ) as api:
            perpetual_stats(symbol, start, end)
        self.assertEqual(api.call_args.args[1]["start_time"], missing.isoformat())
        self.assertEqual(
            api.call_args.args[1]["end_time"],
            (missing + timedelta(hours=1, microseconds=-1)).isoformat(),
        )
        self.assertEqual(rows.count(), 3)

        with patch(
            "quant_tick.exchanges.phoenix.market_history.get_phoenix_response"
        ) as api:
            perpetual_stats(symbol, start, end)
        api.assert_not_called()
