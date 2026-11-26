import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import httpx2
import pandas as pd
import time_machine
from django.core.management import call_command
from django.core.management.base import CommandError
from django.test import TestCase

from quant_tick.constants import Exchange, FileData
from quant_tick.lib import aggregate_candle
from quant_tick.models import (
    AdaptiveCandle,
    CandleCache,
    CandleData,
    ConstantCandle,
    TimeBasedCandle,
    TradeData,
)
from quant_tick.services.check_data import (
    AuditUnavailable,
    check_candle,
    check_exchange,
    daily_candles,
    trade_totals,
)
from quant_tick.tests.base import BaseRandomTradeTest, BaseSymbolTest

START = datetime(2026, 11, 1, tzinfo=UTC)
DAY = timedelta(days=1)


@time_machine.travel(START + 4 * DAY, tick=False)
class DataCheckTest(BaseRandomTradeTest, BaseSymbolTest, TestCase):
    def setUp(self):
        super().setUp()
        self.symbol = self.get_symbol()

    def reference(self, day, *, notional=Decimal(8), volume=Decimal(800)):
        return pd.DataFrame([{"notional": notional, "volume": volume}], index=[day])

    def write_summary(self, start=START, frequency=1440, notional=Decimal(8)):
        return TradeData.objects.create(
            symbol=self.symbol,
            timestamp=start,
            frequency=frequency,
            ok=True,
            json_data={"candle": {"notional": notional, "volume": notional * 100}},
        )

    def make_candle(self, model=ConstantCandle, **config):
        rows = [
            (START + timedelta(hours=12), 4),
            (START + DAY + timedelta(hours=6), 3),
            (START + DAY + timedelta(hours=18), 5),
            (START + 2 * DAY + timedelta(hours=12), 2),
        ]
        frames = {}
        for index in range(3):
            day = START + index * DAY
            frame = pd.DataFrame(
                [
                    self.get_random_trade(
                        timestamp=timestamp,
                        price=Decimal(100),
                        notional=Decimal(size),
                        tick_rule=1,
                    )
                    for timestamp, size in rows
                    if day <= timestamp < day + DAY
                ]
            )
            TradeData.objects.create(
                symbol=self.symbol,
                timestamp=day,
                frequency=1440,
                ok=True,
                json_data={"candle": aggregate_candle(frame)},
            )
            frames[day] = frame
        candle = model.objects.create(
            symbol=self.symbol,
            json_data={
                "source_data": FileData.RAW,
                "sample_type": "notional",
                "target_value": 6,
                **config,
            },
        )
        with patch.object(
            TradeData,
            "get_data_frame",
            autospec=True,
            side_effect=lambda row, field: frames[row.timestamp],
        ):
            candle.candles(START, START + 3 * DAY)
        return candle

    def test_compacted_and_uncompacted_summaries_use_the_same_quantity_check(self):
        for frequency in (1, 60, 1440):
            with self.subTest(frequency=frequency):
                TradeData.objects.all().delete()
                TradeData.objects.bulk_create(
                    [
                        TradeData(
                            symbol=self.symbol,
                            timestamp=START + timedelta(minutes=minute),
                            frequency=frequency,
                            ok=True,
                            json_data={
                                "candle": {
                                    "notional": Decimal(frequency),
                                    "volume": Decimal(frequency * 100),
                                }
                            },
                        )
                        for minute in range(0, 1440, frequency)
                    ]
                )
                with patch.object(
                    TradeData,
                    "get_data_frame",
                    side_effect=AssertionError("No file reads"),
                ):
                    totals, flags = trade_totals(self.symbol, START, START + DAY)
                    result = check_exchange(
                        self.symbol,
                        START,
                        totals,
                        self.reference(
                            START, notional=Decimal(1440), volume=Decimal(144000)
                        ),
                    )
                self.assertEqual(result["status"], "PASS")
                self.assertEqual(flags, {"True": 1440 // frequency})

    def test_valid_flag_does_not_hide_a_daily_volume_mismatch(self):
        row = self.write_summary()
        totals, _ = trade_totals(self.symbol, START, START + DAY)
        result = check_exchange(
            self.symbol, START, totals, self.reference(START, notional=Decimal(9))
        )
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["difference"], -1)
        row.refresh_from_db()
        self.assertTrue(row.ok)

    def test_native_inverse_quantity_and_missing_exchange_candles(self):
        totals = {"volume": Decimal(800), "notional": Decimal(8)}
        self.symbol.exchange = Exchange.BYBIT_INVERSE
        self.assertEqual(
            check_exchange(
                self.symbol, START, totals, self.reference(START, notional=Decimal(9))
            )["status"],
            "PASS",
        )
        self.assertEqual(
            check_exchange(self.symbol, START, totals, pd.DataFrame())["status"],
            "UNKNOWN",
        )
        self.symbol.exchange = Exchange.BITFINEX
        self.assertEqual(
            check_exchange(
                self.symbol, START, dict.fromkeys(totals, Decimal(0)), pd.DataFrame()
            )["status"],
            "PASS",
        )

    def test_hyperliquid_check_uses_native_daily_candles(self):
        self.symbol.exchange = Exchange.HYPERLIQUID
        with (
            patch(
                "quant_tick.services.check_data.native_hyperliquid_candles",
                return_value=self.reference(START),
            ) as native,
            patch("quant_tick.services.check_data.candles_api") as dispatch,
        ):
            daily_candles(self.symbol, START, START + DAY)
        native.assert_called_once_with(self.symbol.api_symbol, START, START + DAY, "1d")
        dispatch.assert_not_called()

    def test_missing_or_ambiguous_partitions_are_not_reported_as_valid(self):
        for problem in ("gap", "overlap", "missing_summary", "unvalidated_empty"):
            with self.subTest(problem=problem):
                TradeData.objects.all().delete()
                row = self.write_summary(frequency=60 if problem == "gap" else 1440)
                if problem == "overlap":
                    self.write_summary(START + timedelta(hours=1), frequency=1)
                elif problem == "missing_summary":
                    row.json_data = {}
                    row.save()
                elif problem == "unvalidated_empty":
                    row.json_data = None
                    row.ok = None
                    row.save()
                with self.assertRaises(AuditUnavailable):
                    trade_totals(self.symbol, START, START + DAY)
        TradeData.objects.all().delete()
        row = self.write_summary()
        row.json_data = None
        row.save()
        totals, _ = trade_totals(self.symbol, START, START + DAY)
        self.assertEqual(totals, {"volume": 0, "notional": 0})

    def test_crossing_candles_and_pending_remainders_reconcile_without_replay(self):
        candle = self.make_candle()
        day = START + DAY
        totals, _ = trade_totals(self.symbol, day, day + DAY)
        with (
            patch.object(
                TradeData, "get_data_frame", side_effect=AssertionError("No file reads")
            ),
            patch.object(
                ConstantCandle, "aggregate", side_effect=AssertionError("No replay")
            ),
        ):
            result = check_candle(candle, day, day + DAY, totals)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["completed_candles"], 1)
        values = result["quantities"]["notional"]
        self.assertEqual(
            (
                values["completed"],
                values["pending_start"],
                values["pending_end"],
                values["accounted"],
            ),
            (7, 4, 5, 8),
        )
        self.assertEqual(
            CandleData.objects.count(), 2
        )  # The second finishes on the following day.
        completed = CandleData.objects.first()
        completed.notional += 1
        completed.save()
        self.assertEqual(check_candle(candle, day, day + DAY, totals)["status"], "FAIL")

    def test_time_and_adaptive_candles_use_the_same_accounting_check(self):
        for model, config in (
            (TimeBasedCandle, {"window": "8h"}),
            (
                AdaptiveCandle,
                {"moving_average_number_of_days": 1, "target_candles_per_day": 2},
            ),
        ):
            with self.subTest(model=model.__name__):
                TradeData.objects.all().delete()
                candle = self.make_candle(model, **config)
                day = START + DAY
                totals, _ = trade_totals(self.symbol, day, day + DAY)
                self.assertEqual(
                    check_candle(candle, day, day + DAY, totals)["status"], "PASS"
                )

    def test_bad_or_missing_cache_is_detected(self):
        candle = self.make_candle()
        day = START + DAY
        totals, _ = trade_totals(self.symbol, day, day + DAY)
        closing = CandleCache.objects.get(candle=candle, timestamp=day)
        closing.json_data["next"]["notional"] += 1
        closing.save()
        self.assertEqual(check_candle(candle, day, day + DAY, totals)["status"], "FAIL")
        closing.delete()
        with self.assertRaisesMessage(AuditUnavailable, "No cache ending"):
            check_candle(candle, day, day + DAY, totals)

    def test_command_reports_failures_without_changing_rows_or_fetching_trades(self):
        candle = self.make_candle()
        before = [
            list(model.objects.values())
            for model in (TradeData, CandleData, CandleCache)
        ]
        with (
            TemporaryDirectory() as directory,
            patch(
                "quant_tick.services.check_data.candles_api",
                return_value=self.reference(START + DAY, notional=Decimal(9)),
            ) as native,
            patch.object(
                TradeData, "get_data_frame", side_effect=AssertionError("No file reads")
            ),
            patch.object(
                ConstantCandle, "aggregate", side_effect=AssertionError("No replay")
            ),
        ):
            output = Path(directory) / "audit.json"
            with self.assertRaises(CommandError):
                call_command(
                    "check_data",
                    code_name=[self.symbol.code_name],
                    candle_code_name=[candle.code_name],
                    date_from="2026-11-02",
                    date_to="2026-11-03",
                    output=output,
                    stdout=StringIO(),
                )
            results = json.loads(output.read_text())
        self.assertEqual([row["status"] for row in results], ["FAIL", "PASS"])
        native.assert_called_once_with(
            self.symbol, START + DAY, START + 2 * DAY, resolution="1d"
        )
        self.assertEqual(
            before,
            [
                list(model.objects.values())
                for model in (TradeData, CandleData, CandleCache)
            ],
        )

    def test_candle_only_check_does_not_make_venue_requests(self):
        self.make_candle()
        with patch(
            "quant_tick.management.commands.check_data.daily_candles",
            side_effect=AssertionError("No API requests"),
        ) as native:
            call_command(
                "check_data",
                code_name=[self.symbol.code_name],
                checks=["candles"],
                date_from="2026-11-02",
                date_to="2026-11-03",
                stdout=StringIO(),
            )
        native.assert_not_called()

    def test_exchange_failure_does_not_prevent_the_database_check(self):
        self.make_candle()
        request = httpx2.Request("GET", "https://example.com/candles")
        error = httpx2.HTTPStatusError(
            "Server error",
            request=request,
            response=httpx2.Response(500, request=request),
        )
        output = StringIO()
        with (
            patch("quant_tick.services.check_data.candles_api", side_effect=error),
            self.assertRaises(CommandError),
        ):
            call_command(
                "check_data",
                code_name=[self.symbol.code_name],
                date_from="2026-11-02",
                date_to="2026-11-03",
                stdout=output,
            )
        results = [
            json.loads(line)
            for line in output.getvalue().splitlines()
            if line.startswith("{")
        ]
        self.assertEqual([row["status"] for row in results], ["UNKNOWN", "PASS"])

    def test_command_requires_a_bounded_completed_range_and_selection(self):
        for options in (
            {},
            {"date_from": "2026-11-01", "date_to": "2026-11-02"},
            {
                "date_from": "2026-11-01",
                "date_to": "2026-11-02",
                "time_from": "00:15",
                "code_name": [self.symbol.code_name],
            },
        ):
            with (
                self.subTest(options=options),
                patch(
                    "quant_tick.management.commands.check_data.daily_candles"
                ) as native,
                self.assertRaises(CommandError),
            ):
                call_command("check_data", **options, stdout=StringIO())
            native.assert_not_called()
