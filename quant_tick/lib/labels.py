import hashlib
import logging

from quant_core.constants import DEFAULT_ASYMMETRIES, DEFAULT_WIDTHS
from quant_core.features import _compute_features

from quant_tick.lib.ml import generate_labels
from quant_tick.models import MLConfig, MLFeatureData

logger = logging.getLogger(__name__)


def generate_labels_from_config(config: MLConfig) -> MLFeatureData | None:
    """Generate labels for MLConfig.

    Args:
        config: MLConfig with candle, widths, asymmetries, horizon_bars

    Returns:
        MLFeatureData
    """
    candle = config.candle

    df = candle.get_candle_data()
    if df is None or len(df) == 0:
        logger.error(f"{config}: no candle data found")
        return None

    # Use config.symbol.exchange as canonical for multi-exchange candles
    df = _compute_features(df, canonical_exchange=config.symbol.exchange)

    widths = config.json_data.get("widths", DEFAULT_WIDTHS)
    asymmetries = config.json_data.get("asymmetries", DEFAULT_ASYMMETRIES)
    max_horizon = config.horizon_bars

    labeled = generate_labels(df, widths, asymmetries, max_horizon)

    schema_hash = hashlib.sha256(
        ",".join(sorted(labeled.columns)).encode()
    ).hexdigest()[:16]

    timestamp_from = labeled["timestamp"].min()
    timestamp_to = labeled["timestamp"].max()

    feature_data, created = MLFeatureData.objects.get_or_create(
        candle=candle,
        timestamp_from=timestamp_from,
        timestamp_to=timestamp_to,
        defaults={"schema_hash": schema_hash},
    )

    feature_data.file_data = MLFeatureData.prepare_data(labeled)
    feature_data.schema_hash = schema_hash
    feature_data.json_data = {
        "schema_type": "hazard",
        "max_horizon": max_horizon,
        "n_bars": len(df) - 1,
        "n_configs": len(widths) * len(asymmetries),
    }
    feature_data.save()

    action = "Created" if created else "Updated"
    logger.info(
        f"{action} hazard MLFeatureData for {candle.code_name}: "
        f"{len(labeled)} rows, schema_hash={schema_hash}"
    )

    return feature_data


# Feature computation functions now imported from quant_core
