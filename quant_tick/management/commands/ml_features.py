import hashlib
import logging
from io import BytesIO
from typing import Any

import pandas as pd
from django.core.files.base import ContentFile

from quant_tick.lib.ml import compute_features
from quant_tick.management.base import BaseCandleCommand
from quant_tick.models import CandleData, MLFeatureData

logger = logging.getLogger(__name__)


class Command(BaseCandleCommand):
    r"""Transform raw candle data into ML-ready features.

    This command loads OHLCV candle data and computes engineered features for machine
    learning: technical indicators, volatility measures, returns, and other derived
    signals. Features are stored as Parquet files for fast loading during training.

    The feature engineering follows AFML principles: fractional differentiation for
    stationarity, EWMA-based indicators, and volatility scaling. Features are computed
    once and reused across multiple training runs.

    Output is stored in MLFeatureData with schema hash tracking to detect changes.
    If features are regenerated with different logic, the schema hash changes and
    you'll know the new features are incompatible with old models.

    Typical usage:
        python manage.py ml_features --symbol BTCUSDT --exchange bybit \\
            --bar-type time --resolution 5m --timestamp-from 2024-01-01
    """

    help = "Generate ML features from existing candle bars."

    def handle(self, *args: Any, **options: Any) -> None:
        """Run command."""
        kwargs = super().handle(*args, **options)
        for k in kwargs:
            candle = k["candle"]
            timestamp_from = k["timestamp_from"]
            timestamp_to = k["timestamp_to"]

            logger.info(f"{candle}: generating features from {timestamp_from} to {timestamp_to}")

            candle_data = CandleData.objects.filter(
                candle=candle,
                timestamp__gte=timestamp_from,
                timestamp__lt=timestamp_to
            ).order_by("timestamp")

            if not candle_data.exists():
                logger.warning(f"{candle}: no candle data found")
                continue

            rows = []
            for cd in candle_data:
                row = {"timestamp": cd.timestamp, **cd.json_data}
                rows.append(row)

            data_frame = pd.DataFrame(rows)
            features = compute_features(data_frame)

            buf = BytesIO()
            features.to_parquet(buf, engine="auto", compression="snappy")
            buf.seek(0)

            schema = str(sorted(features.columns))
            schema_hash = hashlib.sha256(schema.encode()).hexdigest()

            ts_from = timestamp_from.strftime('%Y%m%d_%H%M%S')
            ts_to = timestamp_to.strftime('%Y%m%d_%H%M%S')
            filename = f"features_{ts_from}_{ts_to}.parquet"
            content = ContentFile(buf.read(), filename)

            MLFeatureData.objects.update_or_create(
                candle=candle,
                timestamp_from=timestamp_from,
                timestamp_to=timestamp_to,
                defaults={
                    "file_data": content,
                    "schema_hash": schema_hash,
                }
            )

            count = len(features)
            logger.info(f"{candle}: saved {count} feature rows")
