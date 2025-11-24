"""Run full ML pipeline: labels, train, backtest."""

import logging
from typing import Any

from django.core.management.base import BaseCommand, CommandParser

from quant_tick.lib.backtest import run_backtest
from quant_tick.lib.labels import generate_labels
from quant_tick.lib.train import train_models
from quant_tick.models import Candle, Symbol

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    """Run full ML pipeline."""

    help = "Run full ML pipeline: generate labels, train models, run backtest"

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

    def handle(self, *args: Any, **options: Any) -> None:
        """Run command."""
        candle_code = options["candle"]
        symbol_code = options["symbol"]

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

        # Step 1: Generate labels
        logger.info(f"{'='*50}")
        logger.info("STEP 1: Generating labels")
        logger.info(f"{'='*50}")

        config = generate_labels(candle=candle, symbol=symbol)
        if config is None:
            logger.error("Label generation failed")
            return

        # Step 2: Train models
        logger.info(f"\n{'='*50}")
        logger.info("STEP 2: Training models")
        logger.info(f"{'='*50}")

        success = train_models(config=config)
        if not success:
            logger.error("Model training failed")
            return

        # Step 3: Run backtest
        logger.info(f"\n{'='*50}")
        logger.info("STEP 3: Running backtest")
        logger.info(f"{'='*50}")

        result = run_backtest(config=config)
        if result is None:
            logger.error("Backtest failed")
            return

        logger.info(f"\n{'='*50}")
        logger.info("PIPELINE COMPLETE")
        logger.info(f"{'='*50}")
