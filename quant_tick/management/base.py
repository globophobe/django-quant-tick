import logging

from django.core.management.base import BaseCommand, CommandParser
from django.db.models import QuerySet

from quant_tick.constants import Exchange
from quant_tick.lib import parse_period_from_to
from quant_tick.models import Candle, Symbol

logger = logging.getLogger(__name__)


class BaseTimeFrameCommand(BaseCommand):
    """Base time frame command."""

    def add_arguments(self, parser: CommandParser) -> None:
        """Add arguments."""
        parser.add_argument("--date-to", type=str, default=None)
        parser.add_argument("--time-to", type=str, default=None)
        parser.add_argument("--date-from", type=str, default=None)
        parser.add_argument("--time-from", type=str, default=None)


class BaseTradeDataCommand(BaseTimeFrameCommand):
    """Base trade data command."""

    def get_queryset(self) -> QuerySet:
        """Get queryset."""
        return Symbol.objects.filter(is_active=True)

    def add_arguments(self, parser: CommandParser) -> None:
        """Add arguments."""
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
        parser.add_argument("--aggregate-trades", type=bool)
        parser.add_argument("--significant-trade-filter", type=int)

    def handle(self, *args, **options) -> dict | None:
        """Run command."""
        exchanges = options.get("exchange")
        api_symbols = options.get("api_symbol")
        code_names = options.get("code_name")
        aggregate_trades = options.get("aggregate_trades")
        significant_trade_filter = options.get("significant_trade_filter")
        symbols = self.get_queryset()
        if exchanges:
            symbols = symbols.filter(exchange__in=exchanges)
        if api_symbols:
            symbols = symbols.filter(api_symbol__in=api_symbols)
        if code_names:
            symbols = symbols.filter(code_name__in=code_names)
        if aggregate_trades:
            symbols = symbols.filter(aggregate_trades=aggregate_trades)
        if significant_trade_filter:
            symbols = symbols.filter(significant_trade_filter=significant_trade_filter)
        if symbols:
            timestamp_from, timestamp_to = parse_period_from_to(
                date_from=options["date_from"],
                time_from=options["time_from"],
                date_to=options["date_to"],
                time_to=options["time_to"],
            )
            for symbol in symbols:
                logger.info("{symbol}: starting...".format(**{"symbol": str(symbol)}))
                yield {
                    "symbol": symbol,
                    "timestamp_from": timestamp_from,
                    "timestamp_to": timestamp_to,
                }


class BaseTradeDataWithRetryCommand(BaseTradeDataCommand):
    """Base trade data with retry."""

    def add_arguments(self, parser: CommandParser) -> None:
        """Add arguments."""
        super().add_arguments(parser)
        parser.add_argument("--retry", action="store_true")

    def handle(self, *args, **options) -> dict | None:
        """Run command."""
        kwargs = super().handle(*args, **options)
        for k in kwargs:
            k["retry"] = options.get("retry")
            yield k


class BaseCandleCommand(BaseTimeFrameCommand):
    """Base candle command."""

    def get_queryset(self) -> QuerySet:
        """Get queryset."""
        return Candle.objects.prefetch_related("symbols")

    def add_arguments(self, parser: CommandParser) -> None:
        """Add arguments."""
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
        """Run command."""
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
