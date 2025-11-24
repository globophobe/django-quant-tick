import logging
from typing import Any

from django.core.management.base import BaseCommand, CommandParser

from quant_tick.lib.backtest import run_backtest
from quant_tick.models import MLConfig

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    """Backtest ML model."""

    help = "Backtest ML model."

    def add_arguments(self, parser: CommandParser) -> None:
        """Add arguments."""
        parser.add_argument(
            "--code_name",
            type=str,
            required=True,
            help="MLConfig code name",
        )

    def handle(self, *args: Any, **options: Any) -> None:
        """Run command."""
        code_name = options["code_name"]

        try:
            config = MLConfig.objects.get(code_name=code_name)
        except MLConfig.DoesNotExist:
            logger.error(f"MLConfig '{code_name}' not found")
            return

        run_backtest(config)
