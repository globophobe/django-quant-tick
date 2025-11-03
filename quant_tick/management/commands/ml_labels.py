import hashlib
import logging
from io import BytesIO
from typing import Any

import pandas as pd
from django.core.files.base import ContentFile
from django.core.management.base import CommandParser

from quant_tick.lib.ml import apply_triple_barrier, compute_sample_weights, cusum_events
from quant_tick.management.base import BaseCandleCommand
from quant_tick.models import MLFeatureData

logger = logging.getLogger(__name__)


class Command(BaseCandleCommand):
    r"""Label feature data using triple-barrier method with optional CUSUM events.

    This command adds labels and sample weights to feature data. Each bar (or event)
    gets labeled based on which barrier is hit first: profit-target (+1), stop-loss (-1),
    or time limit (0). Sample weights are computed based on event uniqueness.

    Two labeling modes:
    1. Event-based (--cusum-threshold): Use CUSUM to detect significant price moves,
       then apply triple-barrier to those events only. More selective, focuses on
       clear directional moves. Recommended for live trading.
    2. Bar-based (no threshold): Apply triple-barrier to every bar. More labels but
       noisier, includes lots of neutral/timeout cases. Useful for research.

    The pt_mult and sl_mult parameters define barriers as multiples of recent volatility.
    For example, pt_mult=2.0 means take-profit at 2× recent EWMA volatility. This
    adapts to changing market conditions automatically.

    Sample weights penalize overlapping events (low uniqueness) to reduce overfitting
    on correlated samples during cross-validation.

    Typical usage:
        python manage.py ml_labels --symbol BTCUSDT --exchange bybit \\
            --bar-type time --resolution 5m --pt-mult 2.0 --sl-mult 1.0 \\
            --max-holding 48 --cusum-threshold 0.02
    """

    help = "Generate triple barrier labels from ML features."

    def add_arguments(self, parser: CommandParser) -> None:
        """Add arguments."""
        super().add_arguments(parser)
        parser.add_argument("--pt-mult", type=float, default=2.0)
        parser.add_argument("--sl-mult", type=float, default=1.0)
        parser.add_argument("--max-holding", type=int, default=48)
        parser.add_argument("--cusum-threshold", type=float, default=None, help="CUSUM threshold for event detection (e.g., 0.02 for 2%% moves). If None, labels all bars.")

    def handle(self, *args: Any, **options: Any) -> None:
        """Run command."""
        pt_mult = options["pt_mult"]
        sl_mult = options["sl_mult"]
        max_holding = options["max_holding"]
        cusum_threshold = options["cusum_threshold"]

        kwargs = super().handle(*args, **options)
        for k in kwargs:
            candle = k["candle"]
            timestamp_from = k["timestamp_from"]
            timestamp_to = k["timestamp_to"]

            logger.info(f"{candle}: generating labels from {timestamp_from} to {timestamp_to}")

            feature_data = MLFeatureData.objects.filter(
                candle=candle,
                timestamp_from=timestamp_from,
                timestamp_to=timestamp_to
            ).first()

            if not feature_data or not feature_data.file_data:
                logger.warning(f"{candle}: no feature data found")
                continue

            df = pd.read_parquet(feature_data.file_data.open())

            event_idx = None
            if cusum_threshold is not None:
                event_idx = cusum_events(df, cusum_threshold)
                logger.info(f"{candle}: detected {len(event_idx)} CUSUM events (threshold={cusum_threshold})")

            df = apply_triple_barrier(df, pt_mult, sl_mult, max_holding, event_idx=event_idx)
            df = compute_sample_weights(df)

            buf = BytesIO()
            df.to_parquet(buf, engine="auto", compression="snappy")
            buf.seek(0)

            schema = str(sorted(df.columns))
            schema_hash = hashlib.sha256(schema.encode()).hexdigest()

            ts_from = timestamp_from.strftime('%Y%m%d_%H%M%S')
            ts_to = timestamp_to.strftime('%Y%m%d_%H%M%S')
            filename = f"features_labels_{ts_from}_{ts_to}.parquet"
            content = ContentFile(buf.read(), filename)

            feature_data.file_data = content
            feature_data.schema_hash = schema_hash
            feature_data.save()

            counts = df["label"].value_counts().to_dict()
            logger.info(f"{candle}: added labels {counts}")
