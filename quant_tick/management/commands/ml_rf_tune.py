import json
import logging
from datetime import datetime
from typing import Any

import pandas as pd
from django.core.management.base import BaseCommand, CommandParser
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from quant_tick.lib.ml import PurgedKFold
from quant_tick.models import MLConfig, MLFeatureData, MLRun

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    r"""Automated hyperparameter tuning for Random Forest models.

    Finding the right hyperparameters (n_estimators, max_features, min_samples_leaf, etc)
    usually requires trial and error. This command automates that process using grid
    search or random search with proper time-series cross-validation.

    How it works:
    1. Define parameter grid (manually, from defaults, or auto-suggested by data size)
    2. For each parameter combination, evaluate with PurgedKFold CV
    3. Rank trials by CV AUC score
    4. Save all results to MLRun, optionally update MLConfig with best params

    Search strategies:
    - Grid search: Tests every combination in the grid. Thorough but slow.
    - Random search: Tests N random combinations. Faster, often finds good params.

    The command uses PurgedKFold for all evaluations to prevent label leakage from
    overlapping events. Results include top 10 parameter sets and their CV scores,
    so you can see if multiple configs perform similarly.

    Auto-suggested ranges adapt to dataset size: smaller datasets use fewer trees
    and smaller min_samples_leaf to avoid underfitting.

    Typical usage:
        python manage.py ml_rf_tune --config-code-name my_strategy \\
            --search-type random --n-iter 30 --use-suggested-ranges \\
            --update-config
    """

    help = "Tune RF hyperparameters using grid or random search."

    def add_arguments(self, parser: CommandParser) -> None:
        """Add arguments."""
        parser.add_argument("--config-code-name", type=str, required=True)
        parser.add_argument("--search-type", type=str, choices=["grid", "random"], default="random")
        parser.add_argument("--n-iter", type=int, default=20, help="Number of iterations for random search")
        parser.add_argument("--param-grid-json", type=str, help="Custom param grid as JSON string")
        parser.add_argument("--use-suggested-ranges", action="store_true", help="Auto-suggest ranges from dataset size")
        parser.add_argument("--update-config", action="store_true", help="Update MLConfig.json_data with best params")

    def handle(self, *args: Any, **options: Any) -> None:
        """Run command."""
        cfg_code = options["config_code_name"]
        search_type = options["search_type"]
        n_iter = options["n_iter"]
        param_grid_json = options["param_grid_json"]
        use_suggested_ranges = options["use_suggested_ranges"]
        update_config = options["update_config"]

        try:
            cfg = MLConfig.objects.get(code_name=cfg_code)
        except MLConfig.DoesNotExist:
            logger.error(f"MLConfig {cfg_code} not found")
            return

        candle = cfg.candle
        cfg_json = cfg.json_data

        timestamp_from = pd.to_datetime(cfg_json["timestamp_from"])
        timestamp_to = pd.to_datetime(cfg_json["timestamp_to"])

        embargo_bars = cfg_json.get("embargo_bars", 96)
        n_splits = cfg_json.get("n_splits", 5)

        logger.info(f"{cfg}: loading feature data from {timestamp_from} to {timestamp_to}")

        feature_data = MLFeatureData.objects.filter(
            candle=candle,
            timestamp_from__gte=timestamp_from,
            timestamp_to__lte=timestamp_to
        ).order_by("timestamp_from")

        if not feature_data.exists():
            logger.error(f"{cfg}: no feature data found")
            return

        dfs = []
        for fd in feature_data:
            df = pd.read_parquet(fd.file_data.open())
            dfs.append(df)

        data = pd.concat(dfs, ignore_index=True)

        if "label" not in data.columns:
            logger.error(f"{cfg}: no labels found in feature data")
            return

        drop_cols = [
            "timestamp",
            "label",
            "event_end_idx",
            "event_end_time",
            "sample_weight",
        ]
        feature_cols = [c for c in data.columns if c not in drop_cols]

        X = data[feature_cols].fillna(0).values
        y = data["label"].values
        weights = data["sample_weight"].values if "sample_weight" in data.columns else None
        event_ends = data["event_end_idx"].values if "event_end_idx" in data.columns else None

        logger.info(f"{cfg}: tuning with {len(data)} samples, {len(feature_cols)} features")

        if param_grid_json:
            param_grid = json.loads(param_grid_json)
            logger.info(f"{cfg}: using custom param grid: {param_grid}")
        elif use_suggested_ranges:
            param_grid = self._suggest_param_ranges(len(data), len(feature_cols))
            logger.info(f"{cfg}: using suggested param ranges: {param_grid}")
        else:
            param_grid = {
                "n_estimators": [100, 300, 500, 1000],
                "max_features": ["sqrt", "log2", 0.5, 0.7],
                "min_samples_leaf": [10, 25, 50, 100],
                "max_depth": [5, 10, 15, 20, None],
                "max_samples": [0.5, 0.7, 0.9, None],
            }
            logger.info(f"{cfg}: using default param grid")

        cv = PurgedKFold(n_splits=n_splits, embargo_bars=embargo_bars, shuffle=False)

        scorer = make_scorer(roc_auc_score, needs_proba=True)

        base_model = RandomForestClassifier(
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )

        if search_type == "grid":
            search = GridSearchCV(
                estimator=base_model,
                param_grid=param_grid,
                scoring=scorer,
                cv=cv.split(X, y, event_end_idx=event_ends),
                n_jobs=1,
                verbose=2,
                return_train_score=True,
            )
        else:
            search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=param_grid,
                n_iter=n_iter,
                scoring=scorer,
                cv=cv.split(X, y, event_end_idx=event_ends),
                n_jobs=1,
                verbose=2,
                return_train_score=True,
                random_state=42,
            )

        logger.info(f"{cfg}: starting {search_type} search with {n_splits}-fold CV")

        search.fit(X, y, sample_weight=weights)

        logger.info(f"{cfg}: tuning complete - best CV AUC: {search.best_score_:.4f}")
        logger.info(f"{cfg}: best params: {search.best_params_}")

        cv_results = pd.DataFrame(search.cv_results_)
        top_trials = cv_results.nsmallest(10, "rank_test_score")[
            ["params", "mean_test_score", "std_test_score", "rank_test_score"]
        ].to_dict(orient="records")

        tuning_metadata = {
            "search_type": search_type,
            "n_iter": n_iter if search_type == "random" else None,
            "param_grid": param_grid,
            "best_params": search.best_params_,
            "best_cv_auc": float(search.best_score_),
            "best_cv_std": float(cv_results.loc[search.best_index_, "std_test_score"]),
            "n_trials": len(cv_results),
            "top_10_trials": top_trials,
            "timestamp": datetime.now().isoformat(),
        }

        run = MLRun.objects.create(
            ml_config=cfg,
            timestamp_from=timestamp_from,
            timestamp_to=timestamp_to,
            metrics={"auc": float(search.best_score_)},
            feature_importances={},
            metadata={
                "tuning_results": tuning_metadata,
                "n_features": len(feature_cols),
                "n_samples": len(X),
                "n_splits": n_splits,
                "embargo_bars": embargo_bars,
            },
            status="completed"
        )

        logger.info(f"{cfg}: created MLRun {run.id} with tuning results")

        if update_config:
            model_hparams = cfg_json.get("model_hparams", {})
            model_hparams.update(search.best_params_)
            cfg_json["model_hparams"] = model_hparams
            cfg.json_data = cfg_json
            cfg.save(update_fields=["json_data"])
            logger.info(f"{cfg}: updated model_hparams in config: {model_hparams}")

    def _suggest_param_ranges(self, n_samples: int, n_features: int) -> dict:
        """Suggest hyperparameter ranges based on dataset size and mlbook heuristics.

        Args:
            n_samples: Number of training samples
            n_features: Number of features

        Returns:
            Dict of parameter distributions
        """
        if n_samples < 1000:
            n_estimators = [50, 100, 200]
            min_samples_leaf = [5, 10, 25]
        elif n_samples < 10000:
            n_estimators = [100, 300, 500]
            min_samples_leaf = [10, 25, 50]
        else:
            n_estimators = [300, 500, 1000]
            min_samples_leaf = [25, 50, 100]

        return {
            "n_estimators": n_estimators,
            "max_features": ["sqrt", "log2", 0.5],
            "min_samples_leaf": min_samples_leaf,
            "max_depth": [5, 10, 15, None],
            "max_samples": [0.5, 0.7, 0.9],
        }
