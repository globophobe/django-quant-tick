import logging
from typing import Any

from django.core.management.base import BaseCommand, CommandParser

from quant_tick.lib.labels import generate_hazard_labels_from_config
from quant_tick.lib.train import train_models
from quant_tick.models import MLConfig

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    """Run hazard-based ML pipeline."""

    help = "Run hazard-based ML pipeline: generate hazard labels and train survival models."

    def add_arguments(self, parser: CommandParser) -> None:
        """Add arguments."""
        parser.add_argument(
            "--code-name",
            type=str,
            required=True,
            help="MLConfig",
        )

    def handle(self, *args: Any, **options: Any) -> None:
        """Run command."""
        code_name = options["code_name"]
        try:
            config = MLConfig.objects.get(code_name=code_name)
        except MLConfig.DoesNotExist:
            logger.error(f"MLConfig '{code_name}' not found.")
            return

        # Step 1: Generate hazard labels
        logger.info(f"{'='*50}")
        logger.info("STEP 1: Generating hazard labels")
        logger.info(f"{'='*50}")

        feature_data = generate_hazard_labels_from_config(config)
        if feature_data is None:
            logger.error("Label generation failed")
            return

        # Step 2: Train hazard models
        logger.info(f"\n{'='*50}")
        logger.info("STEP 2: Training hazard models")
        logger.info(f"{'='*50}")

        success = train_models(config=config)
        if not success:
            logger.error("Model training failed")
            return

        logger.info(f"\n{'='*50}")
        logger.info("PIPELINE COMPLETE")
        logger.info(f"{'='*50}")
        logger.info(f"Run 'python manage.py ml_simulate --code-name {code_name}' for validation")
