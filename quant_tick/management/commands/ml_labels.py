import logging
from typing import Any

from django.core.management.base import BaseCommand, CommandParser

from quant_tick.lib.labels import generate_labels_from_config
from quant_tick.lib.ml import DEFAULT_ASYMMETRIES, DEFAULT_WIDTHS
from quant_tick.models import Candle, MLConfig, Symbol

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    """Generate survival labels for ML training."""

    help = "Generate survival labels for ML training"

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
            "--decision-horizons",
            type=str,
            default=None,
            help="Comma-separated decision horizons in bars (e.g., '60,120,180')",
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
        min_bars = options["min_bars"]

        horizons_str = options.get("decision_horizons")
        decision_horizons = (
            [int(x) for x in horizons_str.split(",")]
            if horizons_str
            else [60, 120, 180]
        )

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

        # Get or create MLConfig
        max_horizon = max(decision_horizons)
        config, created = MLConfig.objects.get_or_create(
            candle=candle,
            symbol=symbol,
            defaults={
                "horizon_bars": max_horizon,
                "json_data": {
                    "decision_horizons": decision_horizons,
                    "widths": widths,
                    "asymmetries": asymmetries,
                },
            },
        )

        if created:
            logger.info(f"Created MLConfig for {candle_code} / {symbol_code}")
        else:
            # Update existing config if parameters provided
            update_needed = False
            if config.horizon_bars != max_horizon:
                config.horizon_bars = max_horizon
                update_needed = True
            if config.json_data.get("decision_horizons") != decision_horizons:
                config.json_data["decision_horizons"] = decision_horizons
                update_needed = True
            if config.json_data.get("widths") != widths:
                config.json_data["widths"] = widths
                update_needed = True
            if config.json_data.get("asymmetries") != asymmetries:
                config.json_data["asymmetries"] = asymmetries
                update_needed = True

            if update_needed:
                config.save()
                logger.info(f"Updated MLConfig for {candle_code} / {symbol_code}")

        # Generate labels
        generate_labels_from_config(config)
