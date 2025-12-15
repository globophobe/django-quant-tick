import logging
from typing import Any

from django.core.management.base import BaseCommand, CommandParser

from quant_tick.lib.train import train_models
from quant_tick.models import MLArtifact, MLConfig

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    """Train survival models."""

    help = "Train survival models for range touch prediction."

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
        parser.add_argument("--max-depth", type=int, default=6, help="Max tree depth")
        parser.add_argument(
            "--min-samples-leaf", type=int, default=50, help="Minimum samples per leaf"
        )
        parser.add_argument(
            "--learning-rate", type=float, default=0.05, help="Boosting learning rate"
        )
        parser.add_argument(
            "--subsample",
            type=float,
            default=0.75,
            help="Fraction of samples for each tree",
        )
        parser.add_argument(
            "--holdout-pct", type=float, default=0.2, help="Holdout percentage"
        )
        parser.add_argument(
            "--optuna-n-trials", type=int, default=None, help="Optuna trials per model"
        )
        parser.add_argument(
            "--output-path",
            type=str,
            default=None,
            help="GCS path to save model bundle (gs://bucket/path/model.joblib)",
        )
        parser.add_argument(
            "--set-production",
            action="store_true",
            help="Set is_production=True on created artifact",
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

        training_params = config.get_training_params()

        # Apply CLI overrides
        param_defaults = {
            "n_splits": 5,
            "embargo_bars": 96,
            "n_estimators": 300,
            "max_depth": 6,
            "min_samples_leaf": 50,
            "learning_rate": 0.05,
            "subsample": 0.75,
            "holdout_pct": 0.2,
        }
        cli_overrides = {}
        for param in param_defaults:
            if options[param] != param_defaults[param]:
                cli_overrides[param] = options[param]

        training_params.update(cli_overrides)

        output_path = options.get("output_path")
        set_production = options.get("set_production", False)

        success = train_models(config=config, output_path=output_path, **training_params)

        if success:
            self.stdout.write(self.style.SUCCESS(f"✓ Training complete for {config}"))

            if set_production and output_path:
                MLArtifact.objects.filter(
                    ml_config=config, is_production=True
                ).update(is_production=False)

                artifact = MLArtifact.objects.filter(
                    ml_config=config, gcs_path=output_path
                ).first()

                if artifact:
                    artifact.is_production = True
                    artifact.save()
                    self.stdout.write(
                        self.style.SUCCESS(
                            f"✓ Set artifact {artifact.id} as production model"
                        )
                    )
                else:
                    self.stderr.write(
                        self.style.WARNING(
                            f"Could not find artifact with gcs_path={output_path}"
                        )
                    )
        else:
            self.stderr.write(self.style.ERROR(f"✗ Training failed for {config}"))
