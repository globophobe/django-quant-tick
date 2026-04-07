import logging

from django.core.management.base import BaseCommand, CommandParser
from django.db.models import QuerySet

from quant_tick.constants import Exchange
from quant_tick.lib import parse_period_from_to
from quant_tick.models import Candle, Symbol

logger = logging.getLogger(__name__)


class BaseDateCommand(BaseCommand):
    """Base command with date range arguments."""

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument("--date-to", type=str, default=None)
        parser.add_argument("--date-from", type=str, default=None)


class BaseDateTimeCommand(BaseCommand):
    """Base command with date and time range arguments."""

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument("--date-to", type=str, default=None)
        parser.add_argument("--time-to", type=str, default=None)
        parser.add_argument("--date-from", type=str, default=None)
        parser.add_argument("--time-from", type=str, default=None)


class BaseTradeDataCommand(BaseDateTimeCommand):
    """Base command for iterating trade-data jobs."""

    def get_queryset(self) -> QuerySet:
        return Symbol.objects.filter(is_active=True)

    def add_arguments(self, parser: CommandParser) -> None:
        super().add_arguments(parser)
        queryset = self.get_queryset()
        parser.add_argument(
            "--exchange", type=Exchange, choices=Exchange.values, nargs="+"
        )
        parser.add_argument(
            "--code-name",
            choices=queryset.values_list("code_name", flat=True),
            nargs="+",
        )
        parser.add_argument(
            "--api-symbol",
            choices=queryset.values_list("api_symbol", flat=True),
            nargs="+",
        )

    def handle(self, *args, **options) -> dict | None:
        exchanges = options.get("exchange")
        api_symbols = options.get("api_symbol")
        code_names = options.get("code_name")
        symbols = self.get_queryset()
        if exchanges:
            symbols = symbols.filter(exchange__in=exchanges)
        if api_symbols:
            symbols = symbols.filter(api_symbol__in=api_symbols)
        if code_names:
            symbols = symbols.filter(code_name__in=code_names)
        if symbols:
            timestamp_from, timestamp_to = parse_period_from_to(
                date_from=options["date_from"],
                time_from=options["time_from"],
                date_to=options["date_to"],
                time_to=options["time_to"],
            )
            for symbol in symbols:
                timestamp_range = symbol.clamp_timestamp_range(
                    timestamp_from, timestamp_to
                )
                if timestamp_range is None:
                    continue
                logger.info("{symbol}: starting...".format(**{"symbol": str(symbol)}))
                ts_from, ts_to = timestamp_range
                yield {
                    "symbol": symbol,
                    "timestamp_from": ts_from,
                    "timestamp_to": ts_to,
                }


class BaseTradeDataWithRetryCommand(BaseTradeDataCommand):
    """Base trade-data command with a retry flag."""

    def add_arguments(self, parser: CommandParser) -> None:
        super().add_arguments(parser)
        parser.add_argument("--retry", action="store_true")

    def handle(self, *args, **options) -> dict | None:
        kwargs = super().handle(*args, **options)
        for k in kwargs:
            k["retry"] = options.get("retry")
            yield k


class BaseCandleCommand(BaseDateTimeCommand):
    """Base command for iterating candle jobs."""

    def get_queryset(self) -> QuerySet:
        return Candle.objects.select_related("symbol")

    def add_arguments(self, parser: CommandParser) -> None:
        super().add_arguments(parser)
        parser.add_argument(
            "--code-name",
            type=str,
            choices=self.get_queryset().values_list("code_name", flat=True),
            nargs="+",
        )
        parser.add_argument("--is-active", action="store_true")
        parser.add_argument("--retry", action="store_true")

    def handle(self, *args, **options) -> None:
        code_names = options.get("code_name")
        candles = self.get_queryset()
        is_active = options.get("is_active")
        if is_active:
            candles = candles.filter(is_active=is_active)
        if code_names:
            candles = candles.filter(code_name__in=code_names)
        if candles:
            timestamp_from, timestamp_to = parse_period_from_to(
                date_from=options["date_from"],
                time_from=options["time_from"],
                date_to=options["date_to"],
                time_to=options["time_to"],
            )
            for candle in candles:
                logger.info("{candle}: starting...".format(**{"candle": str(candle)}))
                yield {
                    "candle": candle,
                    "timestamp_from": timestamp_from,
                    "timestamp_to": timestamp_to,
                    "retry": options["retry"],
                }
