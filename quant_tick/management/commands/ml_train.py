"""Train ML models using Random Forest."""

import logging
from typing import Any

from django.core.management.base import BaseCommand, CommandParser

from quant_tick.lib.train import train_models
from quant_tick.models import MLConfig

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    """Train Random Forest ML models."""

    help = "Train ML models for upper/lower bound prediction"

    def add_arguments(self, parser: CommandParser) -> None:
        """Add arguments."""
        parser.add_argument(
            "--config-code-name",
            type=str,
            required=True,
            help="MLConfig code name",
        )
        parser.add_argument(
            "--n-splits",
            type=int,
            default=5,
            help="Number of CV folds (default: 5)",
        )
        parser.add_argument(
            "--embargo-bars",
            type=int,
            default=96,
            help="Embargo period in bars (default: 96)",
        )
        parser.add_argument(
            "--n-estimators",
            type=int,
            default=500,
            help="Number of trees (default: 500)",
        )
        parser.add_argument(
            "--max-depth",
            type=int,
            default=None,
            help="Max tree depth (default: None, unlimited)",
        )
        parser.add_argument(
            "--min-samples-leaf",
            type=int,
            default=50,
            help="Minimum samples per leaf (default: 50)",
        )
        parser.add_argument(
            "--max-features",
            type=str,
            default="sqrt",
            help="Max features per split (default: sqrt)",
        )

    def handle(self, *args: Any, **options: Any) -> None:
        """Run command."""
        config_code = options["config_code_name"]

        try:
            config = MLConfig.objects.get(code_name=config_code)
        except MLConfig.DoesNotExist:
            logger.error(f"MLConfig '{config_code}' not found")
            return

        train_models(
            config=config,
            n_splits=options["n_splits"],
            embargo_bars=options["embargo_bars"],
            n_estimators=options["n_estimators"],
            max_depth=options["max_depth"],
            min_samples_leaf=options["min_samples_leaf"],
            max_features=options["max_features"],
        )
