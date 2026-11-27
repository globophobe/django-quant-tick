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
from quant_tick.services.check_trade_data import (
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

    def table_rows(self, output):
        return [
            tuple(value.strip() for value in line.split(" | "))
            for line in output.getvalue().splitlines()
            if " | " in line
        ]

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
                "quant_tick.services.check_trade_data.native_hyperliquid_candles",
                return_value=self.reference(START),
            ) as native,
            patch("quant_tick.services.check_trade_data.candles_api") as dispatch,
        ):
            daily_candles(self.symbol, START, START + DAY)
        native.assert_called_once_with(self.symbol.api_symbol, START, START + DAY, "1d")
        dispatch.assert_not_called()

    def test_deribit_check_uses_only_complete_utc_days(self):
        self.symbol.exchange = Exchange.DERIBIT
        for instrument in ("BTC-PERPETUAL", "BTC_USDC-PERPETUAL"):
            self.symbol.api_symbol = instrument
            for missing in (None, 0, 12, 23, "all"):
                with self.subTest(instrument=instrument, missing=missing):
                    hours = [
                        offset
                        for offset in range(72)
                        if missing != "all"
                        and (missing is None or offset != 24 + missing)
                    ]
                    response = {
                        "status": "ok",
                        "ticks": [
                            int((START + timedelta(hours=hour)).timestamp() * 1000)
                            for hour in hours
                        ],
                        **dict.fromkeys(
                            ("open", "high", "low", "close"), ["100"] * len(hours)
                        ),
                        "volume": [str(Decimal(hour // 24 + 1) / 10) for hour in hours],
                        "cost": [str((hour // 24 + 1) * 10) for hour in hours],
                    }
                    with patch(
                        "quant_tick.exchanges.deribit.candles.get_deribit_candle_response",
                        return_value=response,
                    ) as native:
                        reference = daily_candles(self.symbol, START, START + 3 * DAY)
                    for index in range(3):
                        day = START + index * DAY
                        size = Decimal(index + 1) * Decimal("2.4")
                        result = check_exchange(
                            self.symbol,
                            day,
                            {"notional": size, "volume": size * 100},
                            reference,
                        )
                        incomplete = missing == "all" or (
                            missing is not None and index == 1
                        )
                        self.assertEqual(
                            result["status"], "UNKNOWN" if incomplete else "PASS"
                        )
                        self.assertEqual(
                            result["exchange_quantity"], None if incomplete else size
                        )
                    native.assert_called_once_with(
                        instrument,
                        int(START.timestamp() * 1000),
                        int((START + 3 * DAY).timestamp() * 1000) - 1,
                        "60",
                    )

    def test_deribit_check_rejects_duplicate_or_misaligned_hours(self):
        self.symbol.exchange = Exchange.DERIBIT
        hours = pd.date_range(START, periods=24, freq="h")
        for index in (hours.append(hours[:1]), hours + timedelta(minutes=1)):
            with (
                self.subTest(index=index),
                patch(
                    "quant_tick.services.check_trade_data.candles_api",
                    return_value=pd.DataFrame({"notional": Decimal(1)}, index=index),
                ),
                self.assertRaises(AuditUnavailable),
            ):
                daily_candles(self.symbol, START, START + DAY)

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
        stdout = StringIO()
        before = [
            list(model.objects.values())
            for model in (TradeData, CandleData, CandleCache)
        ]
        with (
            TemporaryDirectory() as directory,
            patch(
                "quant_tick.services.check_trade_data.candles_api",
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
            with self.assertRaisesMessage(CommandError, "1 failed; 0 unavailable."):
                call_command(
                    "check_trade_data",
                    code_name=[self.symbol.code_name],
                    date_from="2026-11-02",
                    date_to="2026-11-03",
                    output=output,
                    stdout=stdout,
                )
            results = json.loads(output.read_text())
        rows = self.table_rows(stdout)
        self.assertIn(
            (
                self.symbol.exchange,
                self.symbol.api_symbol,
                self.symbol.code_name,
                "Trades vs exchange",
                "0",
                "1",
                "0",
            ),
            rows,
        )
        self.assertIn(
            (
                self.symbol.exchange,
                self.symbol.api_symbol,
                self.symbol.code_name,
                f"Candles/cache: {candle.code_name}",
                "1",
                "0",
                "0",
            ),
            rows,
        )
        self.assertEqual(rows[-1], ("TOTAL", "", "", "", "1", "1", "0"))
        self.assertIn(f"Detailed results: {output}", stdout.getvalue())
        self.assertNotIn('"status":', stdout.getvalue())
        self.assertEqual([row["status"] for row in results], ["FAIL", "PASS"])
        self.assertEqual(
            [row["exchange"] for row in results], [self.symbol.exchange] * 2
        )
        self.assertEqual(results[0]["exchange_quantity"], "9")
        self.assertEqual(results[0]["trade_data"], "8")
        self.assertEqual(results[1]["candle"], candle.code_name)
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

    def test_command_uses_each_symbols_configured_history(self):
        expected_dates = {}
        starts = {}
        symbols = []
        for index, first in enumerate((START + DAY, START, START + 2 * DAY)):
            symbol = self.get_symbol(api_symbol=f"test{index}")
            symbol.date_from = first.date()
            symbol.save()
            symbols.append(symbol)
            if first >= START + 2 * DAY:
                continue
            starts[symbol.pk] = first
            expected_dates[symbol.code_name] = []
            day = first
            while day < START + 2 * DAY:
                TradeData.objects.create(
                    symbol=symbol,
                    timestamp=day,
                    frequency=1440,
                    ok=True,
                    json_data={"candle": {"notional": "8", "volume": "800"}},
                )
                expected_dates[symbol.code_name].append(day.date().isoformat())
                day += DAY
        output = StringIO()
        with patch(
            "quant_tick.services.check_trade_data.candles_api",
            return_value=pd.concat(
                [self.reference(START), self.reference(START + DAY)]
            ),
        ) as native:
            call_command(
                "check_trade_data",
                code_name=[symbol.code_name for symbol in symbols],
                date_from="2026-11-01",
                date_to="2026-11-03",
                verbosity=2,
                stdout=output,
            )
        self.assertEqual(native.call_count, len(starts))
        for request in native.call_args_list:
            symbol, first, last = request.args
            self.assertEqual((first, last), (starts[symbol.pk], START + 2 * DAY))
            self.assertEqual(request.kwargs, {"resolution": "1d"})
        results = [
            json.loads(line)
            for line in output.getvalue().splitlines()
            if line.startswith("{")
        ]
        self.assertEqual(len(results), 3)
        for symbol, dates in expected_dates.items():
            self.assertEqual(
                [row["date"] for row in results if row["symbol"] == symbol], dates
            )
        self.assertTrue(all(row["status"] == "PASS" for row in results))
        rows = self.table_rows(output)
        self.assertEqual(rows[-1], ("TOTAL", "", "", "", "3", "0", "0"))
        for symbol in symbols:
            if symbol.code_name in expected_dates:
                self.assertIn(
                    (
                        symbol.exchange,
                        symbol.api_symbol,
                        symbol.code_name,
                        "Trades vs exchange",
                        str(len(expected_dates[symbol.code_name])),
                        "0",
                        "0",
                    ),
                    rows,
                )

    def test_command_checks_all_symbol_candles_with_fresh_exchange_candles(self):
        candle = self.make_candle()
        inactive = ConstantCandle.objects.create(
            symbol=self.symbol, json_data=candle.json_data, is_active=False
        )
        ConstantCandle.objects.create(
            symbol=self.get_symbol(api_symbol="other"), json_data=candle.json_data
        )
        self.symbol.date_from = (START + DAY).date()
        self.symbol.save()
        output = StringIO()
        with (
            patch(
                "quant_tick.services.check_trade_data.candles_api",
                return_value=self.reference(START + DAY),
            ) as native,
            self.assertRaises(CommandError),
        ):
            call_command(
                "check_trade_data",
                code_name=[self.symbol.code_name],
                date_from="2026-11-01",
                date_to="2026-11-03",
                verbosity=2,
                stdout=output,
            )
        native.assert_called_once_with(
            self.symbol, START + DAY, START + 2 * DAY, resolution="1d"
        )
        results = [
            json.loads(line)
            for line in output.getvalue().splitlines()
            if line.startswith("{")
        ]
        self.assertEqual([row["date"] for row in results], ["2026-11-02"] * 3)
        self.assertEqual(
            (results[0]["check"], results[0]["status"]), ("trades", "PASS")
        )
        self.assertEqual(
            {row["candle"]: row["status"] for row in results[1:]},
            {candle.code_name: "PASS", inactive.code_name: "UNKNOWN"},
        )
        rows = self.table_rows(output)
        self.assertIn(
            (
                self.symbol.exchange,
                self.symbol.api_symbol,
                self.symbol.code_name,
                f"Candles/cache: {inactive.code_name}",
                "0",
                "0",
                "1",
            ),
            rows,
        )
        self.assertEqual(rows[-1], ("TOTAL", "", "", "", "2", "0", "1"))

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
            patch(
                "quant_tick.services.check_trade_data.candles_api", side_effect=error
            ),
            self.assertRaises(CommandError),
        ):
            call_command(
                "check_trade_data",
                code_name=[self.symbol.code_name],
                date_from="2026-11-02",
                date_to="2026-11-03",
                verbosity=2,
                stdout=output,
            )
        results = [
            json.loads(line)
            for line in output.getvalue().splitlines()
            if line.startswith("{")
        ]
        self.assertEqual([row["status"] for row in results], ["UNKNOWN", "PASS"])
        self.assertEqual(
            self.table_rows(output)[-1], ("TOTAL", "", "", "", "1", "0", "1")
        )

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
                    "quant_tick.management.commands.check_trade_data.daily_candles"
                ) as native,
                self.assertRaises(CommandError),
            ):
                call_command("check_trade_data", **options, stdout=StringIO())
            native.assert_not_called()
