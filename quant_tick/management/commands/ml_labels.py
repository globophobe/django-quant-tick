"""Generate ML labels for LP range optimization."""

import logging
from typing import Any

from django.core.management.base import BaseCommand, CommandParser

from quant_tick.lib.labels import generate_labels
from quant_tick.lib.ml import DEFAULT_ASYMMETRIES, DEFAULT_WIDTHS
from quant_tick.models import Candle, Symbol

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    """Generate ML labels for training ML models."""

    help = "Generate ML labels across multiple bound configurations"

    def add_arguments(self, parser: CommandParser) -> None:
        """Add arguments."""
        parser.add_argument(
            "--candle",
            type=str,
            required=True,
            help="Candle code name",
        )
        parser.add_argument(
            "--symbol",
            type=str,
            required=True,
            help="Symbol code name for position tracking",
        )
        parser.add_argument(
            "--horizon-bars",
            type=int,
            default=60,
            help="Prediction horizon in bars (default: 60)",
        )
        parser.add_argument(
            "--min-bars",
            type=int,
            default=1000,
            help="Minimum bars required (default: 1000)",
        )
        parser.add_argument(
            "--widths",
            type=str,
            default=None,
            help="Comma-separated range widths (e.g., '0.03,0.05,0.07')",
        )
        parser.add_argument(
            "--asymmetries",
            type=str,
            default=None,
            help="Comma-separated asymmetries (e.g., '-0.2,0,0.2')",
        )

    def handle(self, *args: Any, **options: Any) -> None:
        """Run command."""
        candle_code = options["candle"]
        symbol_code = options["symbol"]
        horizon_bars = options["horizon_bars"]
        min_bars = options["min_bars"]

        widths_str = options.get("widths")
        widths = (
            [float(x) for x in widths_str.split(",")]
            if widths_str
            else DEFAULT_WIDTHS
        )

        asym_str = options.get("asymmetries")
        asymmetries = (
            [float(x) for x in asym_str.split(",")]
            if asym_str
            else DEFAULT_ASYMMETRIES
        )

        try:
            candle = Candle.objects.get(code_name=candle_code)
        except Candle.DoesNotExist:
            logger.error(f"Candle '{candle_code}' not found")
            return

        try:
            symbol = Symbol.objects.get(code_name=symbol_code)
        except Symbol.DoesNotExist:
            logger.error(f"Symbol '{symbol_code}' not found")
            return

        generate_labels(
            candle=candle,
            symbol=symbol,
            horizon_bars=horizon_bars,
            min_bars=min_bars,
            widths=widths,
            asymmetries=asymmetries,
        )
