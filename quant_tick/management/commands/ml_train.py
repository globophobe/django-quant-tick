import logging
from io import BytesIO
from typing import Any

import joblib
import pandas as pd
from django.core.files.base import ContentFile
from django.core.management.base import BaseCommand, CommandParser

from quant_tick.lib.ml import train_model
from quant_tick.models import MLArtifact, MLConfig, MLFeatureData, MLRun

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    r"""Train Random Forest model with AFML time-series cross-validation.

    This command orchestrates the full training pipeline: load features/labels, run
    purged k-fold CV to prevent label leakage, train final model on all data with
    optional sequential bootstrap, and save artifacts.

    Key features:
    - Purged k-fold: Removes overlapping events between train/test to prevent lookahead
    - Embargo bars: Buffer zone between train/test to prevent correlation leakage
    - Sequential bootstrap: Resample accounting for event overlaps in final model
    - RF diagnostics: Permutation importance, iterative pruning, OOB validation
    - Probability calibration: Fix over/under-confident predictions with Platt/isotonic
    - Early stopping: Automatically find optimal n_estimators by watching CV convergence
    - Model interpretation: Generate PDP/ICE plots and SHAP summaries

    Configuration from MLConfig.json_data:
    - embargo_bars: Buffer after test set (default: 96). Use 0 for small datasets.
    - max_holding_bars: Triple-barrier time limit (default: 48). Must match labeling.
    - model_hparams: Override hyperparameters if not provided via CLI flags.

    Typical usage:
        python manage.py ml_train --config-code-name my_strategy \\
            --n-estimators 500 --n-splits 5 --rf-diagnostics \\
            --calibrate-probabilities --generate-interpretations
    """

    help = "Train ML model using features and labels."

    def add_arguments(self, parser: CommandParser) -> None:
        """Add arguments."""
        parser.add_argument("--config-code-name", type=str, required=True)
        parser.add_argument("--n-estimators", type=int, default=500)
        parser.add_argument("--max-features", type=str, default="sqrt")
        parser.add_argument("--min-samples-leaf", type=int, default=50)
        parser.add_argument("--max-depth", type=int, default=None, help="Maximum tree depth for 'many shallow trees' approach")
        parser.add_argument("--n-splits", type=int, default=5)
        parser.add_argument("--use-sequential-bootstrap", action="store_true", help="Use sequential bootstrap for final model training")
        parser.add_argument("--n-bootstrap-samples", type=int, default=None, help="Number of bootstrap samples (default: len(data))")
        parser.add_argument("--max-samples", type=float, default=None, help="Fraction of samples per tree for OOB-like validation (e.g., 0.7)")
        parser.add_argument("--rf-diagnostics", action="store_true", help="Run RF diagnostics (permutation importance, feature pruning, OOB metrics)")
        parser.add_argument("--generate-interpretations", action="store_true", help="Generate PDP/ICE plots and SHAP summary")
        parser.add_argument("--calibrate-probabilities", action="store_true", help="Calibrate model probabilities using CalibratedClassifierCV")
        parser.add_argument("--calibration-method", type=str, default="sigmoid", choices=["sigmoid", "isotonic"], help="Calibration method: sigmoid (Platt) or isotonic")
        parser.add_argument("--early-stopping", action="store_true", help="Enable early stopping to find optimal n_estimators")
        parser.add_argument("--early-stopping-max", type=int, default=1000, help="Maximum n_estimators for early stopping")
        parser.add_argument("--early-stopping-min", type=int, default=100, help="Minimum n_estimators for early stopping")
        parser.add_argument("--early-stopping-step", type=int, default=50, help="Step size for early stopping")
        parser.add_argument("--early-stopping-epsilon", type=float, default=0.001, help="Convergence threshold for early stopping")
        parser.add_argument("--early-stopping-patience", type=int, default=2, help="Patience (iterations) for early stopping")

    def handle(self, *args: Any, **options: Any) -> None:
        """Run command."""
        cfg_code = options["config_code_name"]

        try:
            cfg = MLConfig.objects.get(code_name=cfg_code)
        except MLConfig.DoesNotExist:
            logger.error(f"MLConfig {cfg_code} not found")
            return

        candle = cfg.candle
        cfg_json = cfg.json_data

        model_hparams = cfg_json.get("model_hparams", {})

        n_estimators = options["n_estimators"] if options["n_estimators"] != 500 else model_hparams.get("n_estimators", 500)
        max_features = options["max_features"] if options["max_features"] != "sqrt" else model_hparams.get("max_features", "sqrt")
        min_samples_leaf = options["min_samples_leaf"] if options["min_samples_leaf"] != 50 else model_hparams.get("min_samples_leaf", 50)
        max_depth = options["max_depth"] if options["max_depth"] is not None else model_hparams.get("max_depth")
        n_splits = options["n_splits"] if options["n_splits"] != 5 else model_hparams.get("n_splits", 5)
        max_samples = options["max_samples"] if options["max_samples"] is not None else model_hparams.get("max_samples")

        use_sequential_bootstrap = options["use_sequential_bootstrap"]
        n_bootstrap_samples = options["n_bootstrap_samples"]
        run_diagnostics = options["rf_diagnostics"]
        generate_interpretations = options["generate_interpretations"]
        calibrate_probabilities = options["calibrate_probabilities"]
        calibration_method = options["calibration_method"]
        enable_early_stopping = options["early_stopping"]
        early_stopping_max = options["early_stopping_max"]
        early_stopping_min = options["early_stopping_min"]
        early_stopping_step = options["early_stopping_step"]
        early_stopping_epsilon = options["early_stopping_epsilon"]
        early_stopping_patience = options["early_stopping_patience"]

        timestamp_from = pd.to_datetime(cfg_json["timestamp_from"])
        timestamp_to = pd.to_datetime(cfg_json["timestamp_to"])

        embargo_bars = cfg_json.get("embargo_bars", 96)
        max_holding_bars = cfg_json.get("max_holding_bars", 48)

        logger.info(f"{cfg}: loading feature data from {timestamp_from} to {timestamp_to}")
        logger.info(f"{cfg}: using embargo_bars={embargo_bars}, max_holding_bars={max_holding_bars}")

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

        logger.info(f"{cfg}: training with {len(data)} samples")
        logger.info(
            f"{cfg}: hyperparameters - n_estimators={n_estimators}, max_features={max_features}, "
            f"min_samples_leaf={min_samples_leaf}, max_depth={max_depth}, max_samples={max_samples}"
        )

        if run_diagnostics:
            logger.info(f"{cfg}: RF diagnostics enabled")
        if generate_interpretations:
            logger.info(f"{cfg}: Interpretation plots enabled")
        if calibrate_probabilities:
            logger.info(f"{cfg}: Probability calibration enabled (method: {calibration_method})")
        if enable_early_stopping:
            logger.info(f"{cfg}: Early stopping enabled (max={early_stopping_max}, epsilon={early_stopping_epsilon})")

        final_model, avg_metrics, sorted_importances, metadata, calibrated_model = train_model(
            data,
            n_estimators=n_estimators,
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            n_splits=n_splits,
            embargo_bars=embargo_bars,
            use_sequential_bootstrap=use_sequential_bootstrap,
            n_bootstrap_samples=n_bootstrap_samples,
            max_samples=max_samples,
            run_diagnostics=run_diagnostics,
            generate_interpretations=generate_interpretations,
            calibrate_probabilities=calibrate_probabilities,
            calibration_method=calibration_method,
            enable_early_stopping=enable_early_stopping,
            early_stopping_max_estimators=early_stopping_max,
            early_stopping_min_estimators=early_stopping_min,
            early_stopping_step=early_stopping_step,
            early_stopping_epsilon=early_stopping_epsilon,
            early_stopping_patience=early_stopping_patience,
        )

        interpretation_plots = metadata.get("interpretation_plots", {})

        metadata["max_holding_bars"] = max_holding_bars

        run = MLRun.objects.create(
            ml_config=cfg,
            timestamp_from=timestamp_from,
            timestamp_to=timestamp_to,
            metrics=avg_metrics,
            feature_importances=sorted_importances,
            metadata=metadata,
            status="completed"
        )

        buf = BytesIO()
        joblib.dump(final_model, buf)
        buf.seek(0)

        ts_from = timestamp_from.strftime('%Y%m%d_%H%M%S')
        ts_to = timestamp_to.strftime('%Y%m%d_%H%M%S')
        filename = f"model_{ts_from}_{ts_to}.joblib"
        content = ContentFile(buf.read(), filename)

        MLArtifact.objects.create(
            ml_run=run,
            artifact=content,
            artifact_type="primary_model",
            version="1.0"
        )

        if interpretation_plots:
            from quant_tick.lib.rf_interpretation import save_interpretation_artifacts

            pdp_plots = interpretation_plots.get("pdp", {})
            shap_summary = interpretation_plots.get("shap")

            saved_artifacts = save_interpretation_artifacts(
                pdp_plots,
                shap_summary,
                run,
                artifact_type="primary_model",
            )
            logger.info(f"{cfg}: saved {len(saved_artifacts)} interpretation plot artifacts")

        if calibrated_model:
            from quant_tick.lib.rf_calibration import plot_calibration_curve

            cal_buf = BytesIO()
            joblib.dump(calibrated_model, cal_buf)
            cal_buf.seek(0)

            cal_filename = f"calibrated_model_{ts_from}_{ts_to}.joblib"
            cal_content = ContentFile(cal_buf.read(), cal_filename)

            MLArtifact.objects.create(
                ml_run=run,
                artifact=cal_content,
                artifact_type="calibrated_model",
                version="1.0"
            )

            cal_curve_data = metadata.get("calibration_curve")
            if cal_curve_data:
                curve_plot = plot_calibration_curve(cal_curve_data, title="Calibration Curve")
                curve_content = ContentFile(curve_plot.read(), "calibration_curve.png")

                MLArtifact.objects.create(
                    ml_run=run,
                    artifact=curve_content,
                    artifact_type="calibration_curve",
                    version="1.0"
                )

            cal_metrics = metadata.get("calibration_metrics", {})
            logger.info(
                f"{cfg}: calibrated model saved - "
                f"Brier: {cal_metrics.get('brier_score', 0):.4f}, "
                f"ECE: {cal_metrics.get('ece', 0):.4f}"
            )

        logger.info(f"{cfg}: training complete - AUC: {avg_metrics['auc']:.4f}")

        if run_diagnostics and "rf_diagnostics" in metadata:
            diag = metadata["rf_diagnostics"]
            if "oob_validation" in diag:
                oob = diag["oob_validation"]
                logger.info(
                    f"{cfg}: OOB score: {oob['oob_score']:.4f}, "
                    f"train/val gap: {oob['train_val_gap']:.4f}"
                )
            if "pruning" in diag:
                best_iter = diag["pruning"]["best_iteration"]
                logger.info(
                    f"{cfg}: Best pruning iteration: {best_iter['n_features']} features "
                    f"with CV AUC: {best_iter['cv_auc']:.4f}"
                )
