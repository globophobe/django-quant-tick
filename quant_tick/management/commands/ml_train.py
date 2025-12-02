import logging
from typing import Any

from django.core.management.base import BaseCommand, CommandParser

from quant_tick.lib.train import train_models
from quant_tick.models import MLConfig

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    """Train ML model."""

    help = "Train per-horizon ML model for upper/lower bound touch prediction."

    def add_arguments(self, parser: CommandParser) -> None:
        """Add arguments."""
        parser.add_argument(
            "--code-name",
            type=str,
            required=True,
            help="MLConfig",
        )
        parser.add_argument(
            "--n-splits",
            type=int,
            default=5,
            help="Number of CV folds",
        )
        parser.add_argument(
            "--embargo-bars",
            type=int,
            default=96,
            help="Embargo bars between folds",
        )
        parser.add_argument(
            "--n-estimators",
            type=int,
            default=300,
            help="Number of boosting iterations",
        )
        parser.add_argument(
            "--max-depth",
            type=int,
            default=6,
            help="Max tree depth",
        )
        parser.add_argument(
            "--min-samples-leaf",
            type=int,
            default=50,
            help="Minimum samples per leaf",
        )
        parser.add_argument(
            "--learning-rate",
            type=float,
            default=0.05,
            help="Boosting learning rate",
        )
        parser.add_argument(
            "--subsample",
            type=float,
            default=0.75,
            help="Fraction of samples for each tree",
        )
        parser.add_argument(
            "--holdout-pct",
            type=float,
            default=0.2,
            help="Holdout percentage",
        )
        parser.add_argument(
            "--optuna-n-trials",
            type=int,
            default=None,
            help="Optuna trials per model (default: 20 from config). Set to 0 to disable.",
        )

    def handle(self, *args: Any, **options: Any) -> None:
        """Run command."""
        code_name = options["code_name"]
        try:
            config = MLConfig.objects.get(code_name=code_name)
        except MLConfig.DoesNotExist:
            logger.error(f"MLConfig '{code_name}' not found")
            return

        # Update config with Optuna trials if specified
        if options["optuna_n_trials"] is not None:
            config.json_data["optuna_n_trials"] = options["optuna_n_trials"]
            config.save(update_fields=["json_data"])

        success = train_models(
            config=config,
            n_splits=options["n_splits"],
            embargo_bars=options["embargo_bars"],
            n_estimators=options["n_estimators"],
            max_depth=options["max_depth"],
            min_samples_leaf=options["min_samples_leaf"],
            learning_rate=options["learning_rate"],
            subsample=options["subsample"],
            holdout_pct=options["holdout_pct"],
        )

        if success:
            self.stdout.write(self.style.SUCCESS(f"✓ Training complete for {config}"))
        else:
            self.stderr.write(self.style.ERROR(f"✗ Training failed for {config}"))
