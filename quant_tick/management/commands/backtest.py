from django.core.management.base import BaseCommand, CommandParser

from quant_tick.models import Strategy


class Command(BaseCommand):
    """Run walk-forward backtest."""

    help = "Run walk-forward backtest."

    def add_arguments(self, parser: CommandParser) -> None:
        """Add arguments."""
        parser.add_argument("--code-name", required=True, help="Strategy code_name")

    def handle(self, *args, **options) -> None:
        """Handle."""
        code_name = options["code_name"]

        try:
            strategy = Strategy.objects.get(code_name=code_name)
        except Strategy.DoesNotExist:
            self.stderr.write(self.style.ERROR(f"Strategy '{code_name}' not found"))
            return

        config = strategy.json_data or {}

        train_months = int(config.get("train_months", 24))
        test_months = int(config.get("test_months", 3))
        step_months = int(config.get("step_months", 3))

        cv_splits = int(config.get("cv_splits", 5))
        embargo_bars = int(config.get("embargo_bars", 10))

        penalties = config.get("penalties") or ["l1", "l2"]
        if isinstance(penalties, str):
            penalties = [p.strip() for p in penalties.split(",") if p.strip()]
        penalties = [p for p in penalties if p in {"l1", "l2"}]
        if not penalties:
            penalties = ["l1", "l2"]

        c_values = config.get("c_values") or [0.1, 1.0, 10.0]
        if isinstance(c_values, str):
            c_values = [float(v) for v in c_values.split(",") if v.strip()]
        else:
            c_values = [float(v) for v in c_values]
        if not c_values:
            c_values = [0.1, 1.0, 10.0]

        calibration_method = (
            str(config.get("calibration_method", "auto")).strip().lower()
        )

        summary = strategy.backtest(
            train_months=train_months,
            test_months=test_months,
            step_months=step_months,
            cv_splits=cv_splits,
            embargo_bars=embargo_bars,
            penalties=penalties,
            c_values=c_values,
            calibration_method=calibration_method,
        )
        self.stdout.write(self.style.SUCCESS("Walk-forward backtest complete"))
        if not summary:
            return
        baseline = summary["baseline"]
        meta = summary["meta"]
        windows = summary.get("windows", {})
        self.stdout.write(
            f"windows evaluated={windows.get('evaluated', 0)} skipped={windows.get('skipped', 0)}"
        )
        self.stdout.write(
            "baseline total={total} taken={taken} take_rate={take_rate:.3f} "
            "win_rate={win_rate:.3f} avg_net_return={avg_net_return:.6f} "
            "cum_net_return={cum_net_return:.6f}".format(**baseline)
        )
        self.stdout.write(
            "meta total={total} taken={taken} take_rate={take_rate:.3f} "
            "win_rate={win_rate:.3f} avg_net_return={avg_net_return:.6f} "
            "cum_net_return={cum_net_return:.6f}".format(**meta)
        )
        self.stdout.write(
            "delta take_rate={:.3f} win_rate={:.3f} avg_net_return={:.6f} "
            "cum_net_return={:.6f}".format(
                meta["take_rate"] - baseline["take_rate"],
                meta["win_rate"] - baseline["win_rate"],
                meta["avg_net_return"] - baseline["avg_net_return"],
                meta["cum_net_return"] - baseline["cum_net_return"],
            )
        )
