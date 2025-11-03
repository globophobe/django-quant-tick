import logging
from typing import Any

import pandas as pd
from django.core.management.base import BaseCommand, CommandParser

from quant_tick.lib.ml import (
    apply_triple_barrier,
    compute_sample_weights,
)
from quant_tick.models import MLConfig, MLFeatureData, MLRun, Position

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    r"""Build dataset for meta-labeling: predicting which primary signals to take.

    Meta-labeling is a two-stage approach from AFML Chapter 3:
    - Stage 1 (primary model): Predicts direction (buy/sell/neutral)
    - Stage 2 (meta model): Predicts whether to actually take that bet (yes/no)

    Why use meta-labeling? Your primary model might be good at finding directional
    edges but bad at knowing when to stay out. The meta model learns to filter out
    the low-confidence or poor-timing signals.

    This command creates the training data for the meta model:
    1. Load primary model's backtest positions (entries and sides)
    2. For each position, check if it was profitable using realized triple-barrier labels
    3. Create new dataset: original features + primary_side + meta_label
       - meta_label = 1 if the primary signal was correct (profitable)
       - meta_label = 0 if the primary signal was wrong (unprofitable)

    The resulting dataset can then be used with ml_train_meta to train a model that
    answers: "Given these features and this primary signal, should I take the bet?"

    Typical usage:
        python manage.py ml_meta --config-code-name my_strategy \\
            --primary-run-id 123 --output-path /tmp/meta_dataset.parquet
    """

    help = "Build meta-labeling dataset for bet sizing model."

    def add_arguments(self, parser: CommandParser) -> None:
        """Add arguments."""
        parser.add_argument("--config-code-name", type=str, required=True)
        parser.add_argument("--primary-run-id", type=int, required=True, help="Primary model MLRun ID")
        parser.add_argument("--output-path", type=str, required=True, help="Output parquet path for meta dataset")

    def handle(self, *args: Any, **options: Any) -> None:
        """Run command."""
        cfg_code = options["config_code_name"]
        primary_run_id = options["primary_run_id"]
        output_path = options["output_path"]

        try:
            cfg = MLConfig.objects.get(code_name=cfg_code)
        except MLConfig.DoesNotExist:
            logger.error(f"MLConfig {cfg_code} not found")
            return

        try:
            ml_run = MLRun.objects.get(id=primary_run_id, ml_config=cfg)
        except MLRun.DoesNotExist:
            logger.error(f"Primary MLRun {primary_run_id} not found for config {cfg_code}")
            return

        candle = cfg.candle
        cfg_json = cfg.json_data

        timestamp_from = ml_run.timestamp_from
        timestamp_to = ml_run.timestamp_to

        pt_mult = cfg_json.get("pt_mult", 2.0)
        sl_mult = cfg_json.get("sl_mult", 1.0)
        max_holding_bars = cfg_json.get("max_holding_bars", 48)

        logger.info(f"{cfg}: building meta dataset from {timestamp_from} to {timestamp_to}")

        feature_data_list = MLFeatureData.objects.filter(
            candle=candle,
            timestamp_from__gte=timestamp_from,
            timestamp_to__lte=timestamp_to
        ).order_by("timestamp_from")

        if not feature_data_list.exists():
            logger.error(f"{cfg}: no feature data found")
            return

        dfs = []
        for fd in feature_data_list:
            df = pd.read_parquet(fd.file_data.open())
            dfs.append(df)

        data = pd.concat(dfs, ignore_index=True)
        data = data.sort_values("timestamp").reset_index(drop=True)

        if "label" not in data.columns:
            logger.info(f"{cfg}: applying triple-barrier labels for outcome")
            data = apply_triple_barrier(data, pt_mult, sl_mult, max_holding_bars)
            data = compute_sample_weights(data)

        positions = Position.objects.filter(
            ml_run=ml_run,
            position_type="backtest"
        ).order_by("entry_timestamp")

        if not positions.exists():
            logger.error(f"{cfg}: no backtest positions found for run {ml_run.id}")
            return

        meta_data = []

        for pos in positions:
            entry_timestamp = pos.entry_timestamp
            primary_side = pos.side

            bar_idx = data[data["timestamp"] == entry_timestamp].index
            if len(bar_idx) == 0:
                continue

            bar_idx = bar_idx[0]

            if bar_idx >= len(data):
                continue

            realized_label = data.loc[bar_idx, "label"]

            if primary_side == 1:
                meta_label = 1 if realized_label == 1 else 0
            elif primary_side == -1:
                meta_label = 1 if realized_label == -1 else 0
            else:
                meta_label = 0

            row = data.loc[bar_idx].copy()
            row["primary_side"] = primary_side
            row["meta_label"] = meta_label

            meta_data.append(row)

        if not meta_data:
            logger.error(f"{cfg}: no meta samples created")
            return

        meta_df = pd.DataFrame(meta_data)

        drop_cols = ["timestamp", "label", "event_end_time"]
        meta_df = meta_df.drop(columns=[c for c in drop_cols if c in meta_df.columns])

        meta_df.to_parquet(output_path, index=False)

        logger.info(f"{cfg}: meta dataset saved to {output_path} ({len(meta_df)} samples)")
        logger.info(f"Meta label distribution: {meta_df['meta_label'].value_counts().to_dict()}")
