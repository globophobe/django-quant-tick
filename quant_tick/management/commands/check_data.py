import json
from collections import Counter
from datetime import timedelta
from pathlib import Path

import httpx2
from django.core.management.base import CommandError

from quant_tick.lib import get_current_time, get_min_time, parse_period_from_to
from quant_tick.management.base import BaseTradeDataCommand
from quant_tick.models import Candle
from quant_tick.models.base import QuantTickEncoder
from quant_tick.services.check_data import (
    AuditUnavailable,
    check_candle,
    check_exchange,
    daily_candles,
    trade_totals,
)


class Command(BaseTradeDataCommand):
    help = (
        "Compare daily TradeData summaries with fresh exchange candles and stored "
        "candle/cache totals. No trade downloads or regeneration. Both dates are "
        "required; date-to is exclusive (UTC). Exit nonzero for FAIL or UNKNOWN."
    )

    def add_arguments(self, parser):
        super().add_arguments(parser)
        parser.add_argument(
            "--checks",
            nargs="+",
            choices=("trades", "candles"),
            default=["trades", "candles"],
            help="trades fetches fresh daily candles; candles checks only the database. Default: both.",
        )
        parser.add_argument(
            "--candle-code-name",
            nargs="+",
            help="Check these candles instead of all active candles for the selected symbols.",
        )
        parser.add_argument(
            "--output", type=Path, help="Also save the results as JSON."
        )

    def handle(self, *args, **options):
        if not options["date_from"] or not options["date_to"]:
            raise CommandError("Both --date-from and --date-to are required.")
        if not any(options.get(key) for key in ("code_name", "exchange", "api_symbol")):
            raise CommandError(
                "Select symbols with --code-name, --exchange or --api-symbol."
            )
        start, end = parse_period_from_to(
            **{
                key: options[key]
                for key in ("date_from", "date_to", "time_from", "time_to")
            }
        )
        if start >= end:
            raise CommandError("The requested range must have a positive duration.")
        if (
            start != get_min_time(start, "1d")
            or end != get_min_time(end, "1d")
            or end > get_min_time(get_current_time(), "1d")
        ):
            raise CommandError(
                "Checks require completed UTC days, with an exclusive midnight end."
            )
        jobs = list(super().handle(*args, **options))
        if not jobs:
            raise CommandError("No matching symbols in the requested range.")
        candle_names = options.get("candle_code_name")
        candles = Candle.objects.filter(symbol__in=[job["symbol"] for job in jobs])
        candles = (
            candles.filter(code_name__in=candle_names)
            if candle_names
            else candles.filter(is_active=True)
        )
        if candle_names and set(candles.values_list("code_name", flat=True)) != set(
            candle_names
        ):
            raise CommandError(
                "Some candle names do not belong to the selected symbols."
            )
        results = []

        def record(context, operation, *arguments):
            try:
                result = {**context, **operation(*arguments)}
            except AuditUnavailable as exc:
                result = {
                    **context,
                    "status": "UNKNOWN",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            results.append(result)
            self.stdout.write(json.dumps(result, cls=QuantTickEncoder))

        for job in jobs:
            symbol = job["symbol"]
            symbol_start = job["timestamp_from"]
            symbol_end = job["timestamp_to"]
            context = {
                "symbol": symbol.code_name,
                "exchange": symbol.exchange,
                "api_symbol": symbol.api_symbol,
            }
            selected = (
                list(candles.filter(symbol=symbol))
                if "candles" in options["checks"]
                else []
            )
            if "candles" in options["checks"] and not selected:
                self.stdout.write(f"{symbol}: no selected candles")
            reference = None
            if "trades" in options["checks"]:
                try:
                    reference = daily_candles(symbol, symbol_start, symbol_end)
                except (
                    httpx2.HTTPError,
                    ValueError,
                    NotImplementedError,
                    RuntimeError,
                ) as exc:
                    result = {
                        **context,
                        "check": "trades",
                        "status": "UNKNOWN",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    results.append(result)
                    self.stdout.write(json.dumps(result, cls=QuantTickEncoder))
            day = symbol_start
            while day < symbol_end:
                next_day = day + timedelta(days=1)
                dated = {**context, "date": day.date()}
                try:
                    totals, flags = trade_totals(symbol, day, next_day)
                except AuditUnavailable as exc:
                    result = {
                        **dated,
                        "check": "trade summaries",
                        "status": "UNKNOWN",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    results.append(result)
                    self.stdout.write(json.dumps(result, cls=QuantTickEncoder))
                else:
                    if reference is not None:
                        record(
                            {**dated, "check": "trades", "stored_flags": flags},
                            check_exchange,
                            symbol,
                            day,
                            totals,
                            reference,
                        )
                    for candle in selected:
                        record(
                            {**dated, "check": "candles", "candle": candle.code_name},
                            check_candle,
                            candle,
                            day,
                            next_day,
                            totals,
                        )
                day = next_day
        if options.get("output"):
            Path(options["output"]).write_text(
                json.dumps(results, cls=QuantTickEncoder, indent=2) + "\n"
            )
        counts = Counter(result["status"] for result in results)
        self.stdout.write(
            f"PASS {counts['PASS']}; FAIL {counts['FAIL']}; UNKNOWN {counts['UNKNOWN']}"
        )
        if not results or counts["FAIL"] or counts["UNKNOWN"]:
            raise CommandError("Data checks did not all pass; see the results above.")
