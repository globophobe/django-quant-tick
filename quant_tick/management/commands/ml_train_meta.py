import logging
from io import BytesIO
from typing import Any

import joblib
import pandas as pd
from django.core.files.base import ContentFile
from django.core.management.base import BaseCommand, CommandParser
from sklearn.ensemble import RandomForestClassifier

from quant_tick.lib.ml import PurgedKFold
from quant_tick.models import MLArtifact, MLConfig, MLRun

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    r"""Train meta model to filter primary signals.

    This is the second stage of meta-labeling. After building the meta dataset with
    ml_meta, this command trains a Random Forest to predict whether a primary signal
    will be profitable.

    The meta model takes as input:
    - All the original features (technical indicators, volatility, etc)
    - The primary model's predicted side (1 for long, -1 for short)

    And outputs a probability: "What's the chance this signal will be profitable?"

    In backtesting and live trading, you use both models together:
    1. Primary model generates a signal (e.g., "buy at 0.7 confidence")
    2. Meta model evaluates: "Should I take this buy signal? 0.4 probability of profit"
    3. If meta probability > threshold (e.g., 0.5), take the trade. Otherwise, skip it.

    This approach significantly improves precision by filtering out false positives
    from the primary model. It's especially useful when you have a directionally-biased
    model that generates too many signals.

    The trained meta model is stored as an MLArtifact attached to the primary run,
    so both models can be loaded together for inference.

    Typical usage:
        python manage.py ml_train_meta --config-code-name my_strategy \\
            --primary-run-id 123 --meta-dataset-path /tmp/meta_dataset.parquet \\
            --n-estimators 500 --n-splits 5
    """

    help = "Train meta model for bet sizing."

    def add_arguments(self, parser: CommandParser) -> None:
        """Add arguments."""
        parser.add_argument("--config-code-name", type=str, required=True)
        parser.add_argument("--primary-run-id", type=int, required=True, help="Primary model MLRun ID to attach meta artifact to")
        parser.add_argument("--meta-dataset-path", type=str, required=True)
        parser.add_argument("--n-estimators", type=int, default=500)
        parser.add_argument("--max-features", type=str, default="sqrt")
        parser.add_argument("--min-samples-leaf", type=int, default=50)
        parser.add_argument("--n-splits", type=int, default=5)

    def handle(self, *args: Any, **options: Any) -> None:
        """Run command."""
        cfg_code = options["config_code_name"]
        primary_run_id = options["primary_run_id"]
        meta_path = options["meta_dataset_path"]
        n_estimators = options["n_estimators"]
        max_features = options["max_features"]
        min_samples_leaf = options["min_samples_leaf"]
        n_splits = options["n_splits"]

        try:
            cfg = MLConfig.objects.get(code_name=cfg_code)
        except MLConfig.DoesNotExist:
            logger.error(f"MLConfig {cfg_code} not found")
            return

        try:
            primary_run = MLRun.objects.get(id=primary_run_id, ml_config=cfg)
        except MLRun.DoesNotExist:
            logger.error(f"Primary MLRun {primary_run_id} not found for config {cfg_code}")
            return

        cfg_json = cfg.json_data
        embargo_bars = cfg_json.get("embargo_bars", 96)

        logger.info(f"{cfg}: loading meta dataset from {meta_path}")

        meta_df = pd.read_parquet(meta_path)

        if "meta_label" not in meta_df.columns:
            logger.error(f"{cfg}: meta_label column not found in dataset")
            return

        if "primary_side" not in meta_df.columns:
            logger.error(f"{cfg}: primary_side column not found in dataset")
            return

        drop_cols = ["meta_label"]
        feature_cols = [c for c in meta_df.columns if c not in drop_cols]

        X = meta_df[feature_cols].fillna(0).values
        y = meta_df["meta_label"].values
        w = meta_df["sample_weight"].values if "sample_weight" in meta_df.columns else None

        event_end_idx = meta_df["event_end_idx"].values if "event_end_idx" in meta_df.columns else None

        logger.info(f"{cfg}: training meta model on {len(X)} samples")
        logger.info(f"Meta label distribution: {pd.Series(y).value_counts().to_dict()}")

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )

        cv = PurgedKFold(n_splits=n_splits, embargo_bars=embargo_bars)

        cv_scores = []
        for train_idx, test_idx in cv.split(X, y, event_end_idx=event_end_idx):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            w_train = w[train_idx] if w is not None else None

            model.fit(X_train, y_train, sample_weight=w_train)
            score = model.score(X_test, y_test)
            cv_scores.append(score)

        mean_score = sum(cv_scores) / len(cv_scores)
        logger.info(f"{cfg}: meta model CV accuracy: {mean_score:.4f}")

        model.fit(X, y, sample_weight=w)

        buf = BytesIO()
        joblib.dump(model, buf)
        buf.seek(0)

        artifact = MLArtifact.objects.create(
            ml_run=primary_run,
            artifact_type="meta_model",
            artifact=ContentFile(buf.read(), f"meta_model_{cfg.code_name}.joblib"),
            version="1.0",
        )

        primary_run.metrics = primary_run.metrics or {}
        primary_run.metrics["meta_cv_accuracy"] = mean_score
        primary_run.metadata = primary_run.metadata or {}
        primary_run.metadata["meta_feature_columns"] = feature_cols
        primary_run.save(update_fields=["metrics", "metadata"])

        logger.info(f"{cfg}: meta model saved as artifact {artifact.id} attached to primary run {primary_run.id}")
        logger.info(f"{cfg}: meta CV accuracy: {mean_score:.4f}")
        logger.info(f"{cfg}: feature columns ({len(feature_cols)}): {feature_cols[:10]}...")
