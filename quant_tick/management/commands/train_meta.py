import tempfile
from pathlib import Path

import joblib
import pandas as pd
from django.core.files.base import File
from django.core.management.base import BaseCommand, CommandParser
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from quant_tick.lib.meta import build_event_dataset
from quant_tick.models import MetaArtifact, MetaModel


class Command(BaseCommand):
    """Train a meta model on Renko events and save a bundle."""

    help = "Train meta-model (logistic regression) on Renko event dataset."

    def add_arguments(self, parser: CommandParser) -> None:
        parser.add_argument("--meta-model", required=True, help="MetaModel code_name")
        parser.add_argument(
            "--threshold",
            type=float,
            default=0.5,
            help="Decision threshold for take/skip (on positive class probability)",
        )
        parser.add_argument(
            "--lookback-hours",
            type=int,
            default=24 * 90,
            help="Lookback window in hours for training data (default 90 days).",
        )

    def handle(self, *args, **options):
        meta_model_code = options["meta_model"]
        threshold = options["threshold"]
        lookback_hours = options["lookback_hours"]

        try:
            meta_model = MetaModel.objects.get(code_name=meta_model_code)
        except MetaModel.DoesNotExist:
            self.stderr.write(self.style.ERROR(f"MetaModel '{meta_model_code}' not found"))
            return

        candle = meta_model.candle
        if candle is None:
            self.stderr.write(
                self.style.ERROR(f"MetaModel '{meta_model_code}' is not linked to a candle")
            )
            return

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
            self.stderr.write(self.style.ERROR("No events found for training"))
            return

        # Feature set: include base directional/run features and any engineered feat_* columns
        base_cols = [
            "direction",
            "run_length_prev",
            "run_duration_prev_seconds",
            "exch_dispersion_close",
            "exch_count",
        ]
        feature_cols = [c for c in base_cols if c in events.columns]
        feature_cols += [c for c in events.columns if c.startswith("feat_")]

        X = events[feature_cols].fillna(0)
        y = events["label"].fillna(0).astype(int)

        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=200)),
            ]
        )
        model.fit(X, y)

        bundle = {
            "model": model,
            "metadata": {
                "feature_cols": feature_cols,
                "threshold": threshold,
                "model_kind": "meta_logreg",
            },
        }

        with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
            joblib.dump(bundle, tmp)
            tmp_path = Path(tmp.name)

        # Deactivate existing artifacts for this meta_model
        MetaArtifact.objects.filter(meta_model=meta_model, is_active=True).update(is_active=False)

        with tmp_path.open("rb") as fh:
            artifact = MetaArtifact.objects.create(
                meta_model=meta_model,
                file=File(fh, name=tmp_path.name),
                model_kind="meta_logreg",
                feature_cols=feature_cols,
                is_active=True,
                json_data={"threshold": threshold},
            )

        self.stdout.write(self.style.SUCCESS(f"Saved bundle as artifact {artifact.id}"))
        tmp_path.unlink(missing_ok=True)
