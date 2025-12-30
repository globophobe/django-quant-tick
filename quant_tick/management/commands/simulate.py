import pandas as pd
from django.core.management.base import BaseCommand, CommandParser
from sklearn.linear_model import LogisticRegression

from quant_tick.lib.meta import build_event_dataset
from quant_tick.models import MetaArtifact, MetaModel


class Command(BaseCommand):
    """Walk-forward simulation comparing baseline vs meta-filtered."""

    help = "Simulate baseline vs meta-filtered trades over a window."

    def add_arguments(self, parser: CommandParser) -> None:
        """Add arguments."""
        parser.add_argument("--meta-model", required=True, help="MetaModel code_name")
        parser.add_argument(
            "--lookback-hours",
            type=int,
            default=24 * 30,
            help="Lookback window in hours for simulation (default 30 days).",
        )

    def handle(self, *args, **options) -> None:
        """Handle."""
        meta_model_code = options["meta_model"]
        lookback_hours = options["lookback_hours"]

        try:
            meta_model = MetaModel.objects.get(code_name=meta_model_code)
        except MetaModel.DoesNotExist:
            self.stderr.write(
                self.style.ERROR(f"MetaModel '{meta_model_code}' not found")
            )
            return

        candle = meta_model.candle
        if candle is None:
            self.stderr.write(
                self.style.ERROR(
                    f"MetaModel '{meta_model_code}' is not linked to a candle"
                )
            )
            return

        artifact = (
            MetaArtifact.objects.filter(meta_model=meta_model, is_active=True)
            .order_by("-created_at")
            .first()
        )
        if artifact is None:
            self.stderr.write(self.style.ERROR("No active MetaArtifact found"))
            return

        bundle = artifact.file
        import joblib

        model_bundle = joblib.load(bundle)
        model: LogisticRegression = model_bundle["model"]
        metadata = model_bundle.get("metadata", {})
        feature_cols = metadata.get("feature_cols", [])
        threshold = metadata.get("threshold", 0.5)

        ts_to = pd.Timestamp.utcnow()
        ts_from = ts_to - pd.Timedelta(hours=lookback_hours)

        events = build_event_dataset(
            candle,
            timestamp_from=ts_from.to_pydatetime(),
            timestamp_to=ts_to.to_pydatetime(),
            meta_model=meta_model,
            include_incomplete=False,
        )
        if events.empty:
            self.stderr.write(self.style.ERROR("No events to simulate"))
            return

        # Baseline: take all
        baseline_trades = events.dropna(subset=["net_return"])
        baseline_return = baseline_trades["net_return"].astype(float).sum()
        baseline_count = len(baseline_trades)

        # Meta: thresholded
        feats = events[feature_cols].fillna(0)
        probs = model.predict_proba(feats)[:, 1]
        take_mask = probs >= threshold
        meta_trades = events.loc[take_mask].dropna(subset=["net_return"])
        meta_return = meta_trades["net_return"].astype(float).sum()
        meta_count = len(meta_trades)

        self.stdout.write("Simulation results")
        self.stdout.write(f"Window: {ts_from} -> {ts_to}")
        self.stdout.write("")
        self.stdout.write("Baseline (take all):")
        self.stdout.write(f"  trades: {baseline_count}")
        self.stdout.write(f"  sum_net_return: {baseline_return:.6f}")
        self.stdout.write("")
        self.stdout.write("Meta-filtered:")
        self.stdout.write(f"  trades: {meta_count}")
        self.stdout.write(f"  sum_net_return: {meta_return:.6f}")
        self.stdout.write(f"  threshold: {threshold}")
