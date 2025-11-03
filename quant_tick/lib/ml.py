"""Machine Learning Pipeline for Financial Time Series

This is the core ML module that ties together feature engineering, labeling, training,
and inference for quantitative trading models. It implements techniques from Advances
in Financial Machine Learning (AFML) to handle the unique challenges of financial data:
overlapping events, non-stationarity, and sample weighting.

Main workflow:
1. Feature engineering: Transform raw price data into stationary, informative features
   using fractional differentiation, technical indicators, and volatility measures.
2. Event-based labeling: Use CUSUM to detect significant price movements, then apply
   triple-barrier method to label each event with a side (long/short/neutral) and
   sample weight based on event uniqueness.
3. Time-series CV: Train with PurgedKFold to prevent lookahead bias from overlapping
   events. Embargo bars between train/test to avoid leakage.
4. Sample weighting: Weight samples by their uniqueness (inverse of concurrent events)
   and optionally use sequential bootstrap for the final model.
5. Inference: Generate signals by computing features on new data and applying the model.

Key components:
- PurgedKFold: Time-series cross-validation with embargo to prevent label leakage
- Triple-barrier labeling: Simultaneous profit-take, stop-loss, and time barriers
- CUSUM events: Detect structural breaks for event-driven sampling
- Sequential bootstrap: Resample accounting for event overlaps
- Feature alignment: Handle schema drift between training and inference
"""

import logging
from collections.abc import Generator
from decimal import Decimal
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import KFold

from quant_tick.lib.fracdiff_sklearn import fdiff

logger = logging.getLogger(__name__)


class PurgedKFold(KFold):
    """Time-series cross-validation that prevents label leakage from overlapping events.

    Standard KFold fails for financial time series because events overlap: a trade opened
    at time T might not close until T+50, overlapping with the test period. If we train
    on this sample, we're using information from the future (label leakage).

    PurgedKFold fixes this two ways:
    1. Purging: Remove training samples whose event ends after the test period starts
    2. Embargo: Remove training samples within N bars after test end (prevents correlation
       between recent training samples and test samples)

    The embargo is critical: even after purging, recent training samples just before the
    test set may be correlated with test samples. The embargo creates a buffer zone.

    Typical usage: embargo_bars=96 (about 2 days for 5m bars). For small test datasets,
    use embargo_bars=0 to preserve samples.
    """

    def __init__(self, n_splits: int = 5, embargo_bars: int = 96, **kwargs) -> None:
        """Initialize purged k-fold cross-validator."""
        super().__init__(n_splits=n_splits, **kwargs)
        self.embargo_bars = embargo_bars

    def split(
        self,
        X: Any,
        y: Any = None,
        groups: Any = None,
        event_end_idx: np.ndarray | None = None,
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """Generate purged train/test splits.

        Purging removes training samples whose event end index
        overlaps with test period start index.
        Embargo removes training samples within embargo_bars after test end.
        """
        if event_end_idx is None:
            for train_idx, test_idx in super().split(X, y, groups):
                yield train_idx, test_idx
            return

        for train_idx, test_idx in super().split(X, y, groups):
            test_start_idx = test_idx.min()
            test_end_idx = test_idx.max()

            train_event_ends = event_end_idx[train_idx]
            purge_mask = train_event_ends < test_start_idx

            purged_train_idx = train_idx[purge_mask]

            if self.embargo_bars > 0:
                embargo_start = test_end_idx
                embargo_end = test_end_idx + self.embargo_bars
                embargo_mask = (purged_train_idx <= embargo_start) | (
                    purged_train_idx > embargo_end
                )
                purged_train_idx = purged_train_idx[embargo_mask]

            if len(purged_train_idx) == 0:
                purged_train_idx = train_idx

            yield purged_train_idx, test_idx


def compute_features(
    df: pd.DataFrame, fracdiff_d: float = 0.4, fracdiff_window: int = 256
) -> pd.DataFrame:
    """Compute features from candle data.

    Uses fractional differentiation on log prices for stationarity
    as recommended by Lopez de Prado.

    Args:
        df: DataFrame with at minimum 'timestamp' and 'close' columns
        fracdiff_d: Order of fractional differentiation (default 0.4)
        fracdiff_window: Window size for fracdiff computation (default 256)
    """
    df = df.copy()

    if df.empty:
        return df

    df = df.sort_values("timestamp").reset_index(drop=True)

    close = df["close"].astype(float)
    high = df["high"].astype(float) if "high" in df.columns else close
    low = df["low"].astype(float) if "low" in df.columns else close

    log_close = np.log(close)
    actual_window = min(fracdiff_window, len(df))
    fracdiff_log_close = fdiff(
        log_close.values, n=fracdiff_d, window=actual_window, mode="same"
    )
    df["fracdiff_log_close"] = fracdiff_log_close
    df["fracdiff_return"] = pd.Series(fracdiff_log_close).diff()

    df["return"] = close.pct_change()
    df["log_return"] = np.log(close / close.shift(1))

    df["return_ma_5"] = df["return"].rolling(5, min_periods=1).mean()
    df["return_ma_10"] = df["return"].rolling(10, min_periods=1).mean()
    df["return_ma_20"] = df["return"].rolling(20, min_periods=1).mean()

    df["volatility_5"] = df["return"].rolling(5, min_periods=1).std()
    df["volatility_10"] = df["return"].rolling(10, min_periods=1).std()
    df["volatility_20"] = df["return"].rolling(20, min_periods=1).std()

    df["true_range"] = pd.concat(
        [high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    df["atr_14"] = df["true_range"].rolling(14, min_periods=1).mean()

    if "buyNotional" in df.columns and "sellNotional" in df.columns:
        buy_notional = df["buyNotional"].astype(float)
        sell_notional = df["sellNotional"].astype(float)
        df["dollar_imbalance"] = buy_notional - sell_notional
        df["dollar_imbalance_ewma"] = (
            df["dollar_imbalance"].ewm(span=20, adjust=False).mean()
        )
        df["buy_proportion"] = buy_notional / (buy_notional + sell_notional).replace(
            0, np.nan
        )

    if "buyVolume" in df.columns and "sellVolume" in df.columns:
        buy_vol = df["buyVolume"].astype(float)
        sell_vol = df["sellVolume"].astype(float)
        df["ofi"] = buy_vol - sell_vol
        df["ofi_ewma"] = df["ofi"].ewm(span=20, adjust=False).mean()

    if "ticks" in df.columns:
        df["ticks"] = df["ticks"].astype(float)

    if "notional" in df.columns:
        df["notional"] = df["notional"].astype(float)

    if "roundVolumePct" in df.columns:
        df["roundVolumePct"] = df["roundVolumePct"].astype(float)
        df["roundVolumePct_ewma"] = (
            df["roundVolumePct"].ewm(span=20, adjust=False).mean()
        )

    if "roundNotionalPct" in df.columns:
        df["roundNotionalPct"] = df["roundNotionalPct"].astype(float)
        df["roundNotionalPct_ewma"] = (
            df["roundNotionalPct"].ewm(span=20, adjust=False).mean()
        )

    if "roundBuyVolumePct" in df.columns:
        df["roundBuyVolumePct"] = df["roundBuyVolumePct"].astype(float)
        df["roundBuyVolumePct_ma_5"] = (
            df["roundBuyVolumePct"].rolling(5, min_periods=1).mean()
        )
        df["roundBuyVolumePct_ma_20"] = (
            df["roundBuyVolumePct"].rolling(20, min_periods=1).mean()
        )

    if "roundSellVolumePct" in df.columns:
        df["roundSellVolumePct"] = df["roundSellVolumePct"].astype(float)
        df["roundSellVolumePct_ma_5"] = (
            df["roundSellVolumePct"].rolling(5, min_periods=1).mean()
        )
        df["roundSellVolumePct_ma_20"] = (
            df["roundSellVolumePct"].rolling(20, min_periods=1).mean()
        )

    if "roundBuyNotionalPct" in df.columns:
        df["roundBuyNotionalPct"] = df["roundBuyNotionalPct"].astype(float)
        df["roundBuyNotionalPct_ma_5"] = (
            df["roundBuyNotionalPct"].rolling(5, min_periods=1).mean()
        )
        df["roundBuyNotionalPct_ma_20"] = (
            df["roundBuyNotionalPct"].rolling(20, min_periods=1).mean()
        )

    if "roundSellNotionalPct" in df.columns:
        df["roundSellNotionalPct"] = df["roundSellNotionalPct"].astype(float)
        df["roundSellNotionalPct_ma_5"] = (
            df["roundSellNotionalPct"].rolling(5, min_periods=1).mean()
        )
        df["roundSellNotionalPct_ma_20"] = (
            df["roundSellNotionalPct"].rolling(20, min_periods=1).mean()
        )

    df["bar_duration"] = df["timestamp"].diff().dt.total_seconds()

    df = df.replace([np.inf, -np.inf], np.nan)

    return df


def compute_ewma_volatility(df: pd.DataFrame, span: int = 20) -> pd.Series:
    """Compute EWMA volatility from returns."""
    if "return" not in df.columns:
        close = df["close"].astype(float)
        df["return"] = close.pct_change()

    vol = df["return"].ewm(span=span, adjust=False).std()
    return vol.fillna(0.01)


def cusum_events(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Detect significant events using CUSUM filter.

    Returns indices where cumulative sum of price changes exceeds threshold.
    Based on López de Prado AFML Chapter 2.5.

    Args:
        df: DataFrame with 'close' and 'timestamp' columns
        threshold: CUSUM threshold (typically 0.01-0.05 for 1%-5% moves)

    Returns:
        Array of event start indices

    Note:
        Optimized with NumPy arrays for ~4× speedup over pandas iteration.
    """
    if df.empty or "close" not in df.columns:
        return np.array([], dtype=int)

    close = df["close"].astype(float).values
    log_returns = np.diff(np.log(close))
    log_returns = np.insert(log_returns, 0, 0.0)

    events = []
    s_pos = 0.0
    s_neg = 0.0

    for i in range(len(log_returns)):
        ret = log_returns[i]

        s_pos = max(0.0, s_pos + ret)
        s_neg = min(0.0, s_neg + ret)

        if s_pos > threshold:
            events.append(i)
            s_pos = 0.0
            s_neg = 0.0
        elif s_neg < -threshold:
            events.append(i)
            s_pos = 0.0
            s_neg = 0.0

    return np.array(events, dtype=int)


def apply_triple_barrier(
    df: pd.DataFrame,
    pt_mult: float,
    sl_mult: float,
    max_holding: int,
    vol_span: int = 20,
    event_idx: np.ndarray | None = None,
) -> pd.DataFrame:
    """Label events using three simultaneous exit conditions: profit-target, stop-loss, and time.

    For each event (usually detected by CUSUM), we set three barriers:
    1. Profit-target: Exit at +pt_mult × volatility (e.g., 2× recent volatility)
    2. Stop-loss: Exit at -sl_mult × volatility
    3. Time limit: Exit after max_holding bars if neither barrier hit

    Whichever barrier gets hit first determines the label:
    - Profit-target hit first → label = +1 (buy signal)
    - Stop-loss hit first → label = -1 (sell signal)
    - Time limit hit first → label = 0 (neutral, no clear direction)

    This is better than fixed-horizon labeling (which ignores when the move actually
    happened) and better than using only returns (which ignores the path taken).

    Args:
        df: DataFrame with OHLCV data including 'close' and 'timestamp'
        pt_mult: Profit-target as multiple of volatility (e.g., 2.0 = 2σ)
        sl_mult: Stop-loss as multiple of volatility (e.g., 2.0 = 2σ)
        max_holding: Maximum bars to hold before forced exit
        vol_span: EWMA span for volatility calculation
        event_idx: Array of event start indices (from CUSUM). None = label all bars

    Returns:
        DataFrame with added columns:
        - label: -1 (sell), 0 (neutral), +1 (buy)
        - event_end_idx: When the event ended (for purging in CV)
        - event_end_time: Timestamp when event ended
    """
    df = df.copy()

    if df.empty or "close" not in df.columns:
        return df

    df = df.sort_values("timestamp").reset_index(drop=True)

    vol = compute_ewma_volatility(df, span=vol_span)

    if event_idx is None:
        event_idx = np.arange(len(df))

    prices = df["close"].astype(float).values
    vol_values = vol.values
    timestamps = df["timestamp"].values

    labels = np.zeros(len(df))
    event_end_idx_arr = np.arange(len(df))
    event_end_times = timestamps.copy()

    for i in event_idx:
        if i >= len(df):
            continue

        entry_price = prices[i]

        if np.isnan(entry_price) or entry_price == 0:
            continue

        volatility = vol_values[i]

        pt = entry_price * (1 + pt_mult * volatility)
        sl = entry_price * (1 - sl_mult * volatility)

        end_idx = i
        label = 0

        for j in range(i + 1, min(i + 1 + max_holding, len(prices))):
            price = prices[j]

            if np.isnan(price):
                continue

            if price >= pt:
                label = 1
                end_idx = j
                break
            elif price <= sl:
                label = -1
                end_idx = j
                break

            end_idx = j

        labels[i] = label
        event_end_idx_arr[i] = end_idx
        event_end_times[i] = timestamps[end_idx]

    df["label"] = labels
    df["event_end_idx"] = event_end_idx_arr
    df["event_end_time"] = event_end_times

    return df


def compute_sample_weights(df: pd.DataFrame) -> np.ndarray:
    """Compute sample weights based on event uniqueness using difference array.

    Uses formal concurrency matrix from AFML Chapter 4.
    Weight for event i is average of (1 / concurrency) over event i's bar span.

    Args:
        df: DataFrame with event_end_idx column

    Returns:
        DataFrame with sample_weight column

    Note:
        O(n) time via difference array + prefix sum.
        Normalizes backwards intervals by clipping event_end_idx >= start_idx.
    """
    if "event_end_idx" not in df.columns:
        df = df.copy()
        df["sample_weight"] = 1.0
        return df

    df = df.copy()

    n = len(df)
    if n == 0:
        return df

    if n == 1:
        df["sample_weight"] = 1.0
        return df

    event_ends = df["event_end_idx"].values.copy().astype(int)
    event_ends = np.maximum(event_ends, np.arange(n))

    max_idx = int(event_ends.max())
    diff = np.zeros(max_idx + 2, dtype=int)

    for i in range(n):
        diff[i] += 1
        if event_ends[i] + 1 < len(diff):
            diff[event_ends[i] + 1] -= 1

    concurrency = np.cumsum(diff)

    inv_concurrency = np.where(concurrency > 0, 1.0 / concurrency, 0.0)
    cumsum_inv = np.cumsum(inv_concurrency)

    weights = np.zeros(n)
    for i in range(n):
        start = i
        end = int(event_ends[i])
        span_length = end - start + 1

        sum_inv = cumsum_inv[end] - (cumsum_inv[start - 1] if start > 0 else 0.0)
        weights[i] = sum_inv / span_length

    df["sample_weight"] = weights

    return df


def sequential_bootstrap(
    event_spans: list[tuple[int, int]], n_samples: int, random_state: int = 42
) -> np.ndarray:
    """Sequential bootstrap sampling using AFML Chapter 4.5 bar-level uniqueness.

    Implements AFML Algorithm 4.3: samples with probability proportional to static
    average uniqueness Θᵢ = (1/Lᵢ) ∑_{t∈eventᵢ} 1/C_t.

    Args:
        event_spans: List of (start_idx, end_idx) tuples
        n_samples: Number of samples to draw
        random_state: Random seed

    Returns:
        Array of sampled event indices

    Note:
        Builds static indicator matrix once, computes fixed uniqueness per event.
        O(n_events × avg_bars_per_event + n_samples) complexity.
    """
    from collections import defaultdict

    rng = np.random.RandomState(random_state)
    n_events = len(event_spans)

    if n_events == 0 or n_samples == 0:
        return np.array([])

    bar_concurrency = defaultdict(int)
    event_bars = []

    for event_id, (start, end) in enumerate(event_spans):
        bars = list(range(start, end + 1))
        event_bars.append(bars)
        for bar in bars:
            bar_concurrency[bar] += 1

    uniqueness = np.zeros(n_events)
    for event_id in range(n_events):
        bars = event_bars[event_id]
        if len(bars) == 0:
            uniqueness[event_id] = 0.0
            continue

        sum_inv_c = sum(1.0 / bar_concurrency[bar] for bar in bars)
        uniqueness[event_id] = sum_inv_c / len(bars)

    if uniqueness.sum() == 0:
        probs = np.ones(n_events) / n_events
    else:
        probs = uniqueness / uniqueness.sum()

    phi = np.zeros(n_events, dtype=float)
    sampled_indices = []

    for _ in range(n_samples):
        sampled_idx = rng.choice(n_events, p=probs)
        sampled_indices.append(sampled_idx)
        phi[sampled_idx] += 1

    return np.array(sampled_indices)


def compute_mda_importance(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    metric_func: Any,
    n_repeats: int = 1,
    random_state: int = 42,
) -> dict:
    """Compute Mean Decrease Accuracy feature importance via permutation.

    For each feature, permute its values and measure the drop in performance.
    """
    rng = np.random.RandomState(random_state)
    baseline_score = metric_func(model, X_test, y_test)

    importances = np.zeros(X_test.shape[1])

    for feature_idx in range(X_test.shape[1]):
        drops = []
        for _ in range(n_repeats):
            X_permuted = X_test.copy()
            rng.shuffle(X_permuted[:, feature_idx])
            permuted_score = metric_func(model, X_permuted, y_test)
            drop = baseline_score - permuted_score
            drops.append(drop)
        importances[feature_idx] = np.mean(drops)

    return importances


def _determine_calibration_class_idx(
    model: Any, calibration_class: int | None = None
) -> tuple[int, int]:
    """Determine which class index to use for calibration metrics/curves.

    For multi-class problems, calibration is applied in one-vs-rest fashion.
    This helper selects which class to use as the "positive" class for generating
    calibration curves and metrics.

    Args:
        model: Trained classifier with .classes_ attribute
        calibration_class: Target class value (e.g., 1 for long signals), or None for auto-detect

    Returns:
        Tuple of (class_idx, class_value) where class_idx is the index in model.classes_
    """
    if calibration_class is None:
        # Auto-detect: prefer +1 class if present (typical AFML long signal)
        if 1 in model.classes_:
            idx = np.where(model.classes_ == 1)[0][0]
            return idx, 1
        # Otherwise use last class
        idx = len(model.classes_) - 1
        return idx, model.classes_[idx]

    # User specified a class
    idx = np.where(model.classes_ == calibration_class)[0][0]
    return idx, calibration_class


def train_model(
    data: pd.DataFrame,
    n_estimators: int = 500,
    max_features: str | int | float = "sqrt",
    min_samples_leaf: int = 50,
    max_depth: int | None = None,
    n_splits: int = 5,
    random_state: int = 42,
    embargo_bars: int = 96,
    importance_method: str = "mdi",
    importance_metric: str = "auc",
    mda_n_repeats: int = 1,
    use_sequential_bootstrap: bool = False,
    n_bootstrap_samples: int | None = None,
    max_samples: float | None = None,
    run_diagnostics: bool = False,
    generate_interpretations: bool = False,
    calibrate_probabilities: bool = False,
    calibration_method: str = "sigmoid",
    enable_early_stopping: bool = False,
    early_stopping_max_estimators: int = 1000,
    early_stopping_min_estimators: int = 100,
    early_stopping_step: int = 50,
    early_stopping_epsilon: float = 0.001,
    early_stopping_patience: int = 2,
) -> dict:
    """Train Random Forest model with purged CV.

    Args:
        importance_method: "mdi" for impurity-based or "mda" for permutation-based
        importance_metric: "auc" or "f1" for MDA scoring
        mda_n_repeats: number of permutation repeats for MDA
        use_sequential_bootstrap: if True, use sequential bootstrap for final model training
        n_bootstrap_samples: number of bootstrap samples (default: len(data))
        max_samples: fraction of samples per tree (for OOB-like validation)
        run_diagnostics: if True, run RF diagnostics (permutation importance, pruning, OOB)
        generate_interpretations: if True, generate PDP/ICE plots and SHAP summary
    """
    import sys

    import sklearn

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
    event_ends = (
        data["event_end_idx"].values if "event_end_idx" in data.columns else None
    )

    cv = PurgedKFold(n_splits=n_splits, embargo_bars=embargo_bars, shuffle=False)

    metadata = {}

    if enable_early_stopping:
        from quant_tick.lib.rf_early_stopping import find_optimal_n_estimators

        logger.info(
            f"Early stopping enabled: max={early_stopping_max_estimators}, "
            f"min={early_stopping_min_estimators}, step={early_stopping_step}, "
            f"epsilon={early_stopping_epsilon}, patience={early_stopping_patience}"
        )

        optimal_n, convergence_history = find_optimal_n_estimators(
            X=X,
            y=y,
            sample_weight=weights,
            kfold=cv,
            max_estimators=early_stopping_max_estimators,
            min_estimators=early_stopping_min_estimators,
            step=early_stopping_step,
            epsilon=early_stopping_epsilon,
            patience=early_stopping_patience,
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            max_samples=max_samples,
            random_state=random_state,
        )

        logger.info(f"Early stopping selected n_estimators={optimal_n}")
        n_estimators = optimal_n
        metadata["early_stopping"] = {
            "optimal_n_estimators": optimal_n,
            "convergence_history": convergence_history,
        }

    fold_metrics = []
    mdi_importances = []
    mda_importances = []
    oof_proba = None
    # Will be initialized after first fold when we know n_classes

    if importance_metric == "auc":

        def metric_func(model: Any, X: np.ndarray, y: np.ndarray) -> float:
            y_proba = model.predict_proba(X)
            unique_y = np.unique(y)
            if len(unique_y) > 1:
                try:
                    if len(model.classes_) > 2:
                        return roc_auc_score(
                            y,
                            y_proba,
                            multi_class="ovr",
                            average="weighted",
                            labels=model.classes_,
                        )
                    else:
                        return roc_auc_score(y, y_proba[:, 1])
                except ValueError:
                    return 0.5
            return 0.5

    else:

        def metric_func(model: Any, X: np.ndarray, y: np.ndarray) -> float:
            y_pred = model.predict(X)
            return f1_score(y, y_pred, average="weighted", zero_division=0)

    for fold, (train_idx, test_idx) in enumerate(
        cv.split(X, y, event_end_idx=event_ends)
    ):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        w_train = weights[train_idx] if weights is not None else None

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
            max_depth=max_depth,
            max_samples=max_samples,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
            oob_score=True if max_samples is not None else False,
        )

        model.fit(X_train, y_train, sample_weight=w_train)

        # Initialize oof_proba after first fold when we know n_classes
        if oof_proba is None and (calibrate_probabilities or generate_interpretations):
            n_classes = len(model.classes_)
            oof_proba = np.zeros((len(X), n_classes))

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        if oof_proba is not None:
            oof_proba[test_idx] = y_proba

        unique_test = np.unique(y_test)
        if len(unique_test) > 1:
            try:
                if len(model.classes_) > 2:
                    auc = roc_auc_score(
                        y_test,
                        y_proba,
                        multi_class="ovr",
                        average="weighted",
                        labels=model.classes_,
                    )
                else:
                    auc = roc_auc_score(y_test, y_proba[:, 1])
            except ValueError:
                auc = 0.5
        else:
            auc = 0.5

        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)

        fold_metrics.append(
            {
                "fold": fold,
                "auc": float(auc),
                "f1": float(f1),
                "precision": float(precision),
                "recall": float(recall),
            }
        )

        mdi_importances.append(model.feature_importances_)

        if importance_method == "mda":
            mda_imp = compute_mda_importance(
                model,
                X_test,
                y_test,
                metric_func,
                n_repeats=mda_n_repeats,
                random_state=random_state,
            )
            mda_importances.append(mda_imp)

    final_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        max_depth=max_depth,
        max_samples=max_samples,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
        oob_score=True if max_samples is not None else False,
    )

    if use_sequential_bootstrap and event_ends is not None:
        n_samples = n_bootstrap_samples if n_bootstrap_samples else len(X)
        event_spans = [(i, event_ends[i]) for i in range(len(X))]
        sampled_indices = sequential_bootstrap(
            event_spans, n_samples, random_state=random_state
        )

        X_boot = X[sampled_indices]
        y_boot = y[sampled_indices]
        w_boot = weights[sampled_indices] if weights is not None else None

        final_model.fit(X_boot, y_boot, sample_weight=w_boot)
    else:
        final_model.fit(X, y, sample_weight=weights)

    avg_metrics = {
        "auc": float(pd.DataFrame(fold_metrics)["auc"].mean()),
        "f1": float(pd.DataFrame(fold_metrics)["f1"].mean()),
        "precision": float(pd.DataFrame(fold_metrics)["precision"].mean()),
        "recall": float(pd.DataFrame(fold_metrics)["recall"].mean()),
        "folds": fold_metrics,
    }

    avg_mdi = pd.DataFrame(mdi_importances).mean(axis=0)
    mdi_dict = {col: float(avg_mdi[i]) for i, col in enumerate(feature_cols)}
    sorted_mdi = dict(sorted(mdi_dict.items(), key=lambda x: x[1], reverse=True)[:20])

    importances_output = {"mdi": sorted_mdi}

    if importance_method == "mda" and mda_importances:
        avg_mda = pd.DataFrame(mda_importances).mean(axis=0)
        mda_dict = {col: float(avg_mda[i]) for i, col in enumerate(feature_cols)}
        sorted_mda = dict(
            sorted(mda_dict.items(), key=lambda x: x[1], reverse=True)[:20]
        )
        importances_output["mda"] = sorted_mda

    final_metadata = {
        "random_state": random_state,
        "n_estimators": n_estimators,
        "max_features": max_features,
        "min_samples_leaf": min_samples_leaf,
        "max_depth": max_depth,
        "max_samples": max_samples,
        "n_splits": n_splits,
        "embargo_bars": embargo_bars,
        "importance_method": importance_method,
        "importance_metric": importance_metric if importance_method == "mda" else None,
        "mda_n_repeats": mda_n_repeats if importance_method == "mda" else None,
        "use_sequential_bootstrap": use_sequential_bootstrap,
        "n_bootstrap_samples": (
            n_bootstrap_samples if use_sequential_bootstrap else None
        ),
        "sklearn_version": sklearn.__version__,
        "python_version": sys.version,
        "n_features": len(feature_cols),
        "n_samples": len(X),
        "feature_columns": sorted(feature_cols),
    }

    if metadata:
        final_metadata.update(metadata)

    metadata = final_metadata

    if run_diagnostics:
        from quant_tick.lib.rf_diagnostics import (
            compute_oob_metrics,
            compute_permutation_importances,
            generate_diagnostics_report,
            iterative_feature_pruning,
        )

        perm_importances = compute_permutation_importances(
            final_model, X, y, n_repeats=10, random_state=random_state
        )

        oob_metrics = None
        if max_samples is not None:
            oob_metrics = compute_oob_metrics(
                X,
                y,
                n_estimators=n_estimators,
                max_features=max_features,
                min_samples_leaf=min_samples_leaf,
                max_samples=max_samples,
                random_state=random_state,
            )

        best_features, pruning_history = iterative_feature_pruning(
            X,
            y,
            feature_cols,
            n_estimators=n_estimators,
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
            max_samples=max_samples,
            n_splits=n_splits,
            min_features=5,
            prune_fraction=0.25,
            random_state=random_state,
        )

        diagnostics_report = generate_diagnostics_report(
            perm_importances, feature_cols, pruning_history, oob_metrics
        )

        metadata["rf_diagnostics"] = diagnostics_report
        metadata["best_features_from_pruning"] = best_features

    if generate_interpretations:
        from quant_tick.lib.rf_interpretation import (
            generate_pdp_plots,
            generate_shap_summary,
        )

        pdp_plots = generate_pdp_plots(
            final_model,
            X,
            feature_cols,
            top_k_features=10,
            kind="both",
        )

        shap_summary = generate_shap_summary(
            final_model,
            X,
            feature_cols,
            max_samples=100,
        )

        interpretation_plots = {"pdp": pdp_plots}
        if shap_summary:
            interpretation_plots["shap"] = shap_summary

        metadata["interpretation_plots_generated"] = list(interpretation_plots.keys())
        metadata["interpretation_plots"] = interpretation_plots

    calibrated_model = None
    if calibrate_probabilities and oof_proba is not None:
        from quant_tick.lib.rf_calibration import (
            calibrate_classifier,
            compute_calibration_metrics,
            generate_calibration_curve,
        )

        # Calibration is applied to all classes by sklearn internally (one-vs-rest)
        calibrated_model = calibrate_classifier(
            final_model, X, y, method=calibration_method, sample_weight=weights
        )

        # For calibration curves/metrics, select which class to analyze
        # (sklearn calibrates all classes, but curves/metrics are per-class)
        cal_class_idx, cal_class_value = _determine_calibration_class_idx(final_model)

        # Guard: check if calibration class is present in labels
        if cal_class_value not in y:
            logger.warning(
                f"Calibration class {cal_class_value} not present in labels, "
                f"skipping calibration curves/metrics"
            )
        else:
            # Convert to binary (selected class vs rest) for calibration metrics
            y_binary = (y == cal_class_value).astype(int)
            oof_proba_binary = oof_proba[:, cal_class_idx]

            cal_curve = generate_calibration_curve(
                y_binary, oof_proba_binary, n_bins=10
            )
            cal_metrics = compute_calibration_metrics(y_binary, oof_proba_binary)

            metadata["calibration_method"] = calibration_method
            metadata["calibration_class"] = int(cal_class_value)
            metadata["calibration_curve"] = cal_curve
            metadata["calibration_metrics"] = cal_metrics

            logger.info(
                f"Calibration ({calibration_method}, class={cal_class_value}): "
                f"Brier={cal_metrics['brier_score']:.4f}, ECE={cal_metrics['ece']:.4f}"
            )

    return final_model, avg_metrics, importances_output, metadata, calibrated_model


def generate_signals(
    model: Any,
    df: pd.DataFrame,
    prob_threshold: float = 0.6,
    expected_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Generate signals from model predictions.

    Aligns df columns to expected schema if provided.
    Sorts feature columns by name for consistent ordering.
    """
    df = df.reset_index(drop=True)

    drop_cols = [
        "timestamp",
        "label",
        "event_end_idx",
        "event_end_time",
        "sample_weight",
    ]

    if expected_columns is not None:
        feature_cols = sorted(expected_columns)
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0.0
    else:
        feature_cols = [c for c in df.columns if c not in drop_cols]
        feature_cols = sorted(feature_cols)

    X = df[feature_cols].fillna(0).values

    proba = model.predict_proba(X)
    predictions = model.predict(X)

    signals = []
    for i, pred in enumerate(predictions):
        prob_class = proba[i][list(model.classes_).index(pred)]
        if prob_class >= prob_threshold:
            signals.append(
                {
                    "timestamp": df.loc[i, "timestamp"],
                    "prediction": int(pred),
                    "probability": float(prob_class),
                }
            )

    return signals


def align_features_to_schema(
    df: pd.DataFrame, expected_columns: list[str], drop_cols: list[str] | None = None
) -> pd.DataFrame:
    """Align feature DataFrame to expected schema.

    Handles schema drift by:
    - Dropping extra columns not in training schema
    - Adding missing columns filled with zeros
    - Sorting columns by name for consistent ordering

    Returns aligned DataFrame and drift statistics.
    """
    if drop_cols is None:
        drop_cols = [
            "timestamp",
            "label",
            "event_end_idx",
            "event_end_time",
            "sample_weight",
        ]

    current_cols = [c for c in df.columns if c not in drop_cols]
    current_set = set(current_cols)
    expected_set = set(expected_columns)

    extra_cols = current_set - expected_set
    missing_cols = expected_set - current_set

    df_aligned = df.copy()

    if extra_cols:
        logger.warning(
            f"Dropping {len(extra_cols)} extra columns not in training schema: {sorted(extra_cols)}"
        )
        df_aligned = df_aligned.drop(columns=list(extra_cols))

    if missing_cols:
        logger.warning(
            f"Adding {len(missing_cols)} missing columns with zeros: {sorted(missing_cols)}"
        )
        for col in missing_cols:
            df_aligned[col] = 0.0

    drift_stats = {
        "extra_columns": sorted(extra_cols),
        "missing_columns": sorted(missing_cols),
        "n_extra": len(extra_cols),
        "n_missing": len(missing_cols),
    }

    return df_aligned, drift_stats


def features_from_candles(
    candle: Any, timestamp_from: Any, timestamp_to: Any, lookback_bars: int = 50
) -> pd.DataFrame:
    """Build features from CandleData for live inference.

    Includes lookback_bars prior to timestamp_from for indicator computation.
    """
    from quant_tick.models import CandleData

    queryset = CandleData.objects.filter(
        candle=candle, timestamp__lt=timestamp_to
    ).order_by("timestamp")

    if lookback_bars > 0:
        all_data = list(queryset)
        target_idx = None
        for i, cd in enumerate(all_data):
            if cd.timestamp >= timestamp_from:
                target_idx = i
                break

        if target_idx is not None:
            start_idx = max(0, target_idx - lookback_bars)
            candle_data = all_data[start_idx:]
        else:
            candle_data = all_data
    else:
        candle_data = list(queryset.filter(timestamp__gte=timestamp_from))

    if not candle_data:
        return pd.DataFrame()

    rows = []
    for cd in candle_data:
        row = {"timestamp": cd.timestamp, **cd.json_data}
        rows.append(row)

    data_frame = pd.DataFrame(rows)

    if "volume" in data_frame.columns and "buyVolume" in data_frame.columns:
        data_frame["sellVolume"] = data_frame["volume"] - data_frame["buyVolume"]

    if "notional" in data_frame.columns and "buyNotional" in data_frame.columns:
        data_frame["sellNotional"] = data_frame["notional"] - data_frame["buyNotional"]

    if "roundVolume" in data_frame.columns and "roundBuyVolume" in data_frame.columns:
        data_frame["roundSellVolume"] = (
            data_frame["roundVolume"] - data_frame["roundBuyVolume"]
        )

        vol = data_frame["volume"].replace(0, np.nan)
        buy_vol = data_frame["buyVolume"].replace(0, np.nan)
        sell_vol = data_frame["sellVolume"].replace(0, np.nan)

        data_frame["roundVolumePct"] = (data_frame["roundVolume"] / vol).fillna(0.0)
        data_frame["roundBuyVolumePct"] = (
            data_frame["roundBuyVolume"] / buy_vol
        ).fillna(0.0)
        data_frame["roundSellVolumePct"] = (
            data_frame["roundSellVolume"] / sell_vol
        ).fillna(0.0)

    if (
        "roundNotional" in data_frame.columns
        and "roundBuyNotional" in data_frame.columns
    ):
        data_frame["roundSellNotional"] = (
            data_frame["roundNotional"] - data_frame["roundBuyNotional"]
        )

        notional = data_frame["notional"].replace(0, np.nan)
        buy_not = data_frame["buyNotional"].replace(0, np.nan)
        sell_not = data_frame["sellNotional"].replace(0, np.nan)

        data_frame["roundNotionalPct"] = (
            data_frame["roundNotional"] / notional
        ).fillna(0.0)
        data_frame["roundBuyNotionalPct"] = (
            data_frame["roundBuyNotional"] / buy_not
        ).fillna(0.0)
        data_frame["roundSellNotionalPct"] = (
            data_frame["roundSellNotional"] / sell_not
        ).fillna(0.0)

    features = compute_features(data_frame)

    result = features[features["timestamp"] >= timestamp_from].copy()
    result = result.sort_values("timestamp").reset_index(drop=True)

    return result


def trigger_ml_inference(candle: Any, timestamp_from: Any, timestamp_to: Any) -> list:
    """Trigger ML inference for a candle after new data arrives.

    Two-stage meta-labeling:
    1. Primary model predicts side (-1, 0, 1) and probability
    2. Meta model (if available) predicts whether to take the bet (meta_prob)

    Guards against model/feature schema drift by aligning features to
    the training schema stored in MLRun.metadata.

    Returns list of created MLSignal objects.
    """
    from quant_tick.models import MLConfig, MLSignal

    cfg = MLConfig.objects.filter(candle=candle, is_active=True).first()
    if not cfg:
        return []

    cfg_json = cfg.json_data
    prob_threshold = cfg_json.get("prob_threshold", 0.6)
    meta_prob_threshold = cfg_json.get("meta_prob_threshold", 0.5)
    lookback_bars = cfg_json.get("lookback_bars", 50)

    latest_run = cfg.ml_runs.filter(status="completed").order_by("-created_at").first()
    if not latest_run:
        return []

    calibrated_artifact = latest_run.ml_artifacts.filter(
        artifact_type="calibrated_model"
    ).first()
    if calibrated_artifact and calibrated_artifact.artifact:
        try:
            model = joblib.load(calibrated_artifact.artifact.open())
            artifact = calibrated_artifact
            logger.info(f"{candle}: using calibrated model from run {latest_run.id}")
        except Exception as e:
            logger.warning(
                f"{candle}: failed to load calibrated model: {e}, falling back to primary"
            )
            artifact = latest_run.ml_artifacts.filter(
                artifact_type="primary_model"
            ).first()
            if not artifact or not artifact.artifact:
                return []
            model = joblib.load(artifact.artifact.open())
    else:
        artifact = latest_run.ml_artifacts.filter(artifact_type="primary_model").first()
        if not artifact or not artifact.artifact:
            return []
        model = joblib.load(artifact.artifact.open())

    meta_artifact = latest_run.ml_artifacts.filter(artifact_type="meta_model").first()
    meta_model = None
    if meta_artifact and meta_artifact.artifact:
        meta_model = joblib.load(meta_artifact.artifact.open())
        logger.info(f"{candle}: loaded meta model for two-stage inference")

    df = features_from_candles(candle, timestamp_from, timestamp_to, lookback_bars)

    if df.empty:
        return []

    metadata = latest_run.metadata or {}
    expected_columns = metadata.get("feature_columns")

    if expected_columns:
        df, drift_stats = align_features_to_schema(df, expected_columns)

        if drift_stats["n_extra"] > 0 or drift_stats["n_missing"] > 0:
            logger.info(
                f"Schema drift detected for {candle}: "
                f"{drift_stats['n_extra']} extra, {drift_stats['n_missing']} missing columns"
            )
    else:
        logger.warning(
            f"No feature schema found in MLRun metadata for {candle}. "
            "Using features as-is with sorted column order."
        )

    signals = generate_signals(model, df, prob_threshold, expected_columns)

    created_signals = []
    for sig in signals:
        primary_side = sig["prediction"]
        primary_prob = Decimal(str(sig["probability"]))
        meta_prob = None
        meta_label = None

        if meta_model and primary_side != 0:
            row_idx = df[df["timestamp"] == sig["timestamp"]].index
            if len(row_idx) > 0:
                row_idx = row_idx[0]
                meta_features = df.iloc[row_idx : row_idx + 1].copy()
                meta_features["primary_side"] = primary_side

                meta_cols = sorted(
                    [c for c in meta_features.columns if c not in ["timestamp"]]
                )
                X_meta = meta_features[meta_cols].fillna(0).values

                try:
                    meta_pred_proba = meta_model.predict_proba(X_meta)[0]
                    meta_prob = Decimal(str(meta_pred_proba[1]))
                    meta_label = (
                        1 if meta_prob >= Decimal(str(meta_prob_threshold)) else 0
                    )

                    if meta_label == 0:
                        logger.info(
                            f"{candle} {sig['timestamp']}: meta model filtered signal "
                            f"(primary_side={primary_side}, meta_prob={meta_prob:.4f})"
                        )
                        continue

                except Exception as e:
                    logger.error(f"{candle}: meta model inference failed: {e}")
                    meta_prob = None
                    meta_label = None

        obj, created = MLSignal.objects.update_or_create(
            candle=candle,
            timestamp=sig["timestamp"],
            defaults={
                "ml_artifact": artifact,
                "probability": primary_prob,
                "side": primary_side,
                "meta_label": meta_label,
                "meta_prob": meta_prob,
                "size": None,
                "json_data": {
                    "prob_threshold": prob_threshold,
                    "meta_prob_threshold": meta_prob_threshold,
                },
            },
        )
        if created:
            created_signals.append(obj)

    return created_signals
