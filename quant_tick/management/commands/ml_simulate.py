import logging
from typing import Any

from django.core.management.base import BaseCommand, CommandParser

from quant_tick.lib.simulate import ml_simulate
from quant_tick.models import MLConfig

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    """Run walk-forward simulation."""

    help = "Run walk-forward simulation."

    def add_arguments(self, parser: CommandParser) -> None:
        """Add arguments."""
        parser.add_argument("--code-name", type=str, required=True, help="MLConfig")
        parser.add_argument(
            "--cadence", type=int, default=None, help="Days between retrains"
        )
        parser.add_argument(
            "--window", type=int, default=None, help="Training window in days"
        )
        parser.add_argument(
            "--holdout", type=int, default=None, help="Final holdout days"
        )
        parser.add_argument(
            "--policy-mode",
            type=str,
            default=None,
            choices=["lp", "perps", "directional"],
            help='Policy mode: "lp" (liquidity provision), "perps" (perpetual futures), or "directional" (signed exposure)',
        )

    def handle(self, *args: Any, **options: Any) -> None:
        """Execute the command."""
        code_name = options["code_name"]

        try:
            config = MLConfig.objects.get(code_name=code_name)
        except MLConfig.DoesNotExist:
            self.stderr.write(self.style.ERROR(f"MLConfig '{code_name}' not found"))
            return

        sim_params = config.get_simulation_params()

        # Apply CLI overrides
        cadence = (
            options["cadence"]
            if options["cadence"] is not None
            else sim_params["retrain_cadence_days"]
        )
        window = (
            options["window"]
            if options["window"] is not None
            else sim_params["train_window_days"]
        )
        holdout = (
            options["holdout"]
            if options["holdout"] is not None
            else sim_params["holdout_days"]
        )
        policy_mode = (
            options["policy_mode"]
            if options["policy_mode"] is not None
            else sim_params["policy_mode"]
        )

        logger.info(
            f"Simulation config: cadence={cadence} days, window={window} days, "
            f"holdout={holdout} days, policy_mode={policy_mode}"
        )

        self.stdout.write(f"Running walk-forward simulation for {config}")
        self.stdout.write(f"Retrain cadence: {cadence} days")
        self.stdout.write(f"Training window: {window} days")
        if holdout:
            self.stdout.write(f"Final holdout: {holdout} days")
        self.stdout.write(f"Policy mode: {policy_mode}")

        result = ml_simulate(
            config=config,
            retrain_cadence_days=cadence,
            train_window_days=window,
            holdout_days=holdout,
            policy_mode=policy_mode,
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
        self.stdout.write(f"Avg CV Logloss: {agg['avg_cv_logloss']:.4f}")
        self.stdout.write(f"Avg Holdout Logloss: {agg['avg_holdout_logloss']:.4f}")
        self.stdout.write(f"Avg Touch Rate: {agg['avg_touch_rate']:.2%}")
        self.stdout.write(f"Avg % In Range: {agg['avg_pct_in_range']:.2%}")
        self.stdout.write(f"Total Rebalances: {agg['total_rebalances']}")
        self.stdout.write(f"Total Positions: {agg['total_positions']}")
        self.stdout.write(self.style.SUCCESS("-" * 60))

        # Display policy-specific metrics
        if policy_mode == "perps" and "perps_metrics" in agg:
            perps = agg["perps_metrics"]
            self.stdout.write("PERPS TRADING METRICS")
            self.stdout.write(f"Total trades: {perps['n_trades']}")
            self.stdout.write(f"Total trailing stops: {perps['total_trailing_stops']}")
            self.stdout.write(f"Win rate: {perps['win_rate']:.2%}")
            self.stdout.write(f"Profit factor: {perps['profit_factor']:.2f}")
            self.stdout.write(f"Avg win: {perps['avg_win']:.4%}")
            self.stdout.write(f"Avg loss: {perps['avg_loss']:.4%}")
            self.stdout.write(f"Sum return: {perps['sum_ret']:.4f}")
            self.stdout.write(f"Compound return: {perps['compound_ret']:.4%}")
            self.stdout.write(f"Avg MFE: {perps['avg_mfe']:.4%}")
        elif policy_mode == "directional" and "directional_metrics" in agg:
            direct = agg["directional_metrics"]
            self.stdout.write("DIRECTIONAL RETURN (range-exit)")
            self.stdout.write(f"Total trades: {direct['n_trades']}")
            self.stdout.write(f"Sum return: {direct['sum_ret']:.4f} (raw sum)")
            self.stdout.write(f"Compound return: {direct['compound_ret']:.4%}")
            self.stdout.write(f"Avg return per trade: {direct['avg_ret']:.4%}")

            # By exit reason
            if direct.get("by_exit"):
                self.stdout.write("\nBy Exit Reason:")
                for reason, data in sorted(direct["by_exit"].items()):
                    count = data["count"]
                    sum_ret = data["sum_ret"]
                    self.stdout.write(
                        f"  {reason}: {count} trades, sum_ret={sum_ret:.4f}"
                    )

                # Sanity check
                total_by_exit = sum(data["count"] for data in direct["by_exit"].values())
                if total_by_exit != direct["n_trades"]:
                    self.stdout.write(
                        self.style.WARNING(
                            f"WARNING: by_exit sum ({total_by_exit}) != n_trades ({direct['n_trades']})"
                        )
                    )
        else:
            # LP mode
            if "lp_metrics" in agg:
                lp = agg["lp_metrics"]
                self.stdout.write("LP RETURN PROXY (entry→exit, not true Uniswap IL)")
                self.stdout.write(f"Total trades: {lp['n_trades']}")
                self.stdout.write(f"Sum return: {lp['sum_ret']:.4f} (raw sum)")
                self.stdout.write(f"Compound return: {lp['compound_ret']:.4%}")
                self.stdout.write(f"Avg return per trade: {lp['avg_ret']:.4%}")

                # By exit reason
                if lp.get("by_exit"):
                    self.stdout.write("\nBy Exit Reason:")
                    for reason, data in sorted(lp["by_exit"].items()):
                        count = data["count"]
                        sum_ret = data["sum_ret"]
                        self.stdout.write(
                            f"  {reason}: {count} trades, sum_ret={sum_ret:.4f}"
                        )

                    # Sanity check: sum of by_exit counts should equal n_trades
                    total_by_exit = sum(data["count"] for data in lp["by_exit"].values())
                    if total_by_exit != lp["n_trades"]:
                        self.stdout.write(
                            self.style.WARNING(
                                f"WARNING: by_exit sum ({total_by_exit}) != n_trades ({lp['n_trades']})"
                            )
                        )
            else:
                # Fallback if no LP metrics
                self.stdout.write("LP METRICS (no data)")

        self.stdout.write(self.style.SUCCESS("=" * 60))

        self.stdout.write(f"\nCreated {len(slices)} simulation windows")
        self.stdout.write(self.style.SUCCESS("✓ Walk-forward simulation complete"))
