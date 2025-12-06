from typing import Any

from django.core.management.base import BaseCommand, CommandParser

from quant_tick.lib.simulate import ml_simulate
from quant_tick.models import MLConfig


class Command(BaseCommand):
    """Run walk-forward simulation."""

    help = "Run walk-forward simulation."

    def add_arguments(self, parser: CommandParser) -> None:
        """Add arguments."""
        parser.add_argument(
            "--code-name",
            type=str,
            required=True,
            help="MLConfig code name to simulate",
        )
        parser.add_argument(
            "--cadence",
            type=int,
            default=7,
            help="Days between retrains (default: 7)",
        )
        parser.add_argument(
            "--window",
            type=int,
            default=84,
            help="Training window in days (default: 84)",
        )
        parser.add_argument(
            "--holdout",
            type=int,
            default=None,
            help="Final holdout days (default: None)",
        )

    def handle(self, *args: Any, **options: Any) -> None:
        """Execute the command."""
        code_name = options["code_name"]
        cadence = options["cadence"]
        window = options["window"]
        holdout = options["holdout"]

        try:
            config = MLConfig.objects.get(code_name=code_name)
        except MLConfig.DoesNotExist:
            self.stderr.write(self.style.ERROR(f"MLConfig '{code_name}' not found"))
            return

        self.stdout.write(f"Running walk-forward simulation for {config}")
        self.stdout.write(f"Retrain cadence: {cadence} days")
        self.stdout.write(f"Training window: {window} days")
        if holdout:
            self.stdout.write(f"Final holdout: {holdout} days")

        result = ml_simulate(
            config=config,
            retrain_cadence_days=cadence,
            train_window_days=window,
            holdout_days=holdout,
        )

        if not result or not result.aggregate_metrics:
            self.stderr.write(self.style.ERROR("Walk-forward simulation failed"))
            return

        agg = result.aggregate_metrics
        slices = result.slice_results

        self.stdout.write(self.style.SUCCESS("\n" + "=" * 60))
        self.stdout.write(self.style.SUCCESS("WALK-FORWARD SIMULATION RESULTS"))
        self.stdout.write(self.style.SUCCESS("=" * 60))
        self.stdout.write(f"Windows attempted: {result.windows_attempted}")
        self.stdout.write(f"Windows completed: {agg['n_windows']}")
        self.stdout.write(f"Windows skipped: {result.windows_skipped}")
        if result.windows_skipped > 0:
            skip_pct = result.windows_skipped / result.windows_attempted
            self.stdout.write(self.style.WARNING(f"  ({skip_pct:.1%} skipped)"))
        self.stdout.write(f"Avg CV Brier - Lower: {agg['avg_cv_brier_lower']:.4f}")
        self.stdout.write(f"Avg CV Brier - Upper: {agg['avg_cv_brier_upper']:.4f}")
        self.stdout.write(f"Avg Holdout Brier - Lower: {agg['avg_holdout_brier_lower']:.4f}")
        self.stdout.write(f"Avg Holdout Brier - Upper: {agg['avg_holdout_brier_upper']:.4f}")
        self.stdout.write(f"Avg Touch Rate: {agg['avg_touch_rate']:.2%}")
        self.stdout.write(f"Avg % In Range: {agg['avg_pct_in_range']:.2%}")
        self.stdout.write(f"Total Rebalances: {agg['total_rebalances']}")
        self.stdout.write(f"Total Positions: {agg['total_positions']}")
        self.stdout.write(self.style.SUCCESS("=" * 60))

        self.stdout.write(f"\nCreated {len(slices)} simulation windows")
        self.stdout.write(self.style.SUCCESS("✓ Walk-forward simulation complete"))
