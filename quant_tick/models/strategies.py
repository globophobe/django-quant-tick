import logging
from datetime import datetime
from decimal import Decimal

import joblib
import numpy as np
import pandas as pd
from django.db import models
from django.db.models import QuerySet
from django.utils import timezone
from pandas import DataFrame
from polymorphic.models import PolymorphicModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from quant_tick.constants import Decision
from quant_tick.lib.calibration import apply_calibration, calibrate_probabilities
from quant_tick.lib.cross_validation import PurgedKFold
from quant_tick.utils import gettext_lazy as _

from .base import AbstractCodeName, JSONField
from .candles import CandleData
from .meta_labelling import MLArtifact

logger = logging.getLogger(__name__)


class BaseStrategy(models.Model):
    """Base strategy."""

    json_data = JSONField(_("json data"), null=True)
    is_active = models.BooleanField(_("is active"), default=True)

    class Meta:
        abstract = True


class Strategy(BaseStrategy, AbstractCodeName, PolymorphicModel):
    """Strategy."""

    candle = models.ForeignKey(
        "quant_tick.Candle",
        on_delete=models.CASCADE,
        verbose_name=_("candle"),
    )
    symbol = models.ForeignKey(
        "quant_tick.Symbol",
        on_delete=models.CASCADE,
        related_name="strategies",
        null=False,
        blank=False,
        verbose_name=_("symbol"),
        help_text=_("Target symbol for feature selection and trading."),
    )
    last_candle_data = models.ForeignKey(
        "quant_tick.CandleData",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        verbose_name=_("last candle data"),
    )

    def get_data_frame(self, queryset: QuerySet) -> DataFrame:
        """Get data frame."""
        data = []
        for idx, obj in enumerate(queryset):
            is_incomplete = bool(obj.json_data.get("incomplete", False))
            data.append(
                {
                    "timestamp": obj.timestamp,
                    "obj": obj,
                    "bar_idx": idx,
                    **obj.json_data,
                    **{"incomplete": is_incomplete},
                }
            )
        return DataFrame(data)

    def get_events(
        self,
        *,
        timestamp_from: datetime,
        timestamp_to: datetime,
        include_incomplete: bool = False,
    ) -> DataFrame:
        """Get events."""
        raise NotImplementedError

    def get_event_time_column(self) -> str:
        """Event timestamp column name."""
        return "timestamp_entry"

    def get_event_exit_column(self) -> str:
        """Event exit timestamp column name."""
        return "timestamp_exit"

    def get_label_column(self) -> str:
        """Label column name."""
        return "label"

    @property
    def cost(self) -> Decimal:
        """Return per-strategy transaction cost."""
        if self.json_data and self.json_data.get("cost") is not None:
            return Decimal(str(self.json_data["cost"]))
        return Decimal("0")

    def get_feature_columns(self, events: DataFrame) -> list[str]:
        """Select feature columns for modeling."""
        base_cols = [
            "direction",
            "run_length_prev",
            "run_duration_prev_seconds",
            "exch_dispersion_close",
            "exch_count",
        ]
        feature_cols = [c for c in base_cols if c in events.columns]
        feature_cols += [c for c in events.columns if c.startswith("feat_")]
        return feature_cols

    def _clean_event_json(self, event: dict) -> dict:
        """Return event data without NaN/NaT values for JSON validity."""
        cleaned: dict[str, object] = {}
        for key, value in event.items():
            if key == "obj":
                continue
            if pd.isna(value):
                cleaned[key] = None
            else:
                cleaned[key] = value
        return cleaned

    def get_warmup_bars(self) -> int:
        """Return warmup bars required for feature computation."""
        if self.json_data and self.json_data.get("warmup_bars") is not None:
            return int(self.json_data["warmup_bars"])
        return self.compute_max_warmup_bars()

    @staticmethod
    def compute_max_warmup_bars() -> int:
        """Compute maximum warmup bars required for default features."""
        warmup_requirements = {
            "realized_vol": 20,  # rolling(20).std()
            "realized_vol5": 5,  # rolling(5).std()
            "vol_ratio": 20,  # max(realizedVol5, realizedVol)
            "vol_zscore": 120,  # vol_slow.rolling(60).std() needs 60 + vol_slow needs 60
            "vol_percentile": 100,  # rolling(100).rank()
            "rolling_sharpe20": 20,  # rolling(20) for mean and std
            "ofi_ma5": 5,  # rolling(5).mean()
            "ofi_ma20": 20,  # rolling(20).mean()
        }
        return max(warmup_requirements.values())

    def compute_features(self, df: DataFrame) -> DataFrame:
        """Compute default features for a strategy."""
        if self._is_multi_exchange(df):
            return self._compute_multi_exchange_features(df)
        return self._compute_single_exchange_features(df)

    def _is_multi_exchange(self, df: DataFrame) -> bool:
        """Return True when per-exchange close columns are present."""
        return any(c.endswith("Close") and c != "close" for c in df.columns)

    def _compute_single_exchange_features(self, data_frame: DataFrame) -> DataFrame:
        """Compute single exchange features."""
        df = data_frame.copy()
        returns = df["close"].pct_change(fill_method=None)

        max_warmup = self.get_warmup_bars()
        df["has_full_warmup"] = (np.arange(len(df)) >= max_warmup).astype(int)

        df["realizedVol"] = returns.rolling(20).std()
        df["realizedVol5"] = returns.rolling(5).std()

        vol_slow = returns.rolling(60).std()
        df["volRatio"] = df["realizedVol5"] / df["realizedVol"]
        df["volZScore"] = (df["realizedVol"] - vol_slow) / vol_slow.rolling(60).std()
        df["volPercentile"] = df["realizedVol"].rolling(100).rank(pct=True)
        df["isHighVol"] = (df["volPercentile"] > 0.75).astype(int)
        df["isLowVol"] = (df["volPercentile"] < 0.25).astype(int)

        rolling_mean = returns.rolling(20).mean()
        rolling_std = returns.rolling(20).std()
        df["rollingSharpe20"] = rolling_mean / (rolling_std + 1e-8)

        buy_vol_col = "buyVolume"
        vol_col = "volume"
        if buy_vol_col in df.columns and vol_col in df.columns:
            ofi = df[buy_vol_col] / df[vol_col].replace(0, np.nan) - 0.5
            df["aggregateOfi"] = ofi
            df["aggregateOfiMa5"] = ofi.rolling(5).mean()
            df["aggregateOfiMa20"] = ofi.rolling(20).mean()

        return df

    def _compute_multi_exchange_features(self, data_frame: DataFrame) -> DataFrame:
        """Compute multi exchange features."""
        df = data_frame.copy()

        close_cols = [
            c for c in data_frame.columns if c.endswith("Close") and c != "close"
        ]

        canonical_exchange = self.symbol.exchange

        preferred_col = f"{canonical_exchange}Close"
        if preferred_col not in close_cols:
            raise ValueError(
                f"Canonical exchange '{canonical_exchange}' not found in candle data. "
                f"Available exchanges: {[c.replace('Close', '') for c in close_cols]}"
            )

        canonical_col = preferred_col
        canonical_name = canonical_col.replace("Close", "")
        canonical_close = data_frame[canonical_col]

        if "close" not in df.columns:
            df["close"] = canonical_close
        if "low" not in df.columns and f"{canonical_name}Low" in df.columns:
            df["low"] = df[f"{canonical_name}Low"]
        if "high" not in df.columns and f"{canonical_name}High" in df.columns:
            df["high"] = df[f"{canonical_name}High"]
        if "open" not in df.columns and f"{canonical_name}Open" in df.columns:
            df["open"] = df[f"{canonical_name}Open"]

        canonical_returns = canonical_close.pct_change(fill_method=None)
        df[f"{canonical_name}Ret"] = canonical_returns

        max_warmup = self.get_warmup_bars()
        df["has_full_warmup"] = (np.arange(len(df)) >= max_warmup).astype(int)

        for close_col in [c for c in close_cols if c != canonical_col]:
            other_name = close_col.replace("Close", "")
            other_close = df[close_col]

            df[f"{other_name}Missing"] = other_close.isna().astype(int)
            df[f"basis{other_name.title()}"] = other_close - canonical_close
            df[f"basisPct{other_name.title()}"] = (
                other_close - canonical_close
            ) / canonical_close

            other_returns = other_close.pct_change(fill_method=None)
            df[f"{other_name}Ret"] = other_returns
            df[f"retDivergence{other_name.title()}"] = other_returns - canonical_returns

            for lag in [1, 2, 3, 5]:
                df[f"{other_name}RetLag{lag}"] = other_returns.shift(lag)

            other_vol_col = f"{other_name}Volume"
            canon_vol_col = f"{canonical_name}Volume"
            if other_vol_col in df.columns and canon_vol_col in df.columns:
                df[f"volRatio{other_name.title()}"] = df[other_vol_col] / df[
                    canon_vol_col
                ].replace(0, np.nan)

        df["realizedVol"] = canonical_returns.rolling(20).std()
        df["realizedVol5"] = canonical_returns.rolling(5).std()

        vol_slow = canonical_returns.rolling(60).std()
        df["volRatio"] = df["realizedVol5"] / df["realizedVol"]
        df["volZScore"] = (df["realizedVol"] - vol_slow) / vol_slow.rolling(60).std()
        df["volPercentile"] = df["realizedVol"].rolling(100).rank(pct=True)
        df["isHighVol"] = (df["volPercentile"] > 0.75).astype(int)
        df["isLowVol"] = (df["volPercentile"] < 0.25).astype(int)

        rolling_mean = canonical_returns.rolling(20).mean()
        rolling_std = canonical_returns.rolling(20).std()
        df["rollingSharpe20"] = rolling_mean / (rolling_std + 1e-8)

        exchanges = [c.replace("Close", "") for c in close_cols]
        ofi_series = {}
        notional_series = {}

        for exchange in exchanges:
            buy_vol_col = f"{exchange}BuyVolume"
            vol_col = f"{exchange}Volume"
            notional_col = f"{exchange}Notional"

            if buy_vol_col in df.columns and vol_col in df.columns:
                ofi = df[buy_vol_col] / df[vol_col].replace(0, np.nan) - 0.5
                df[f"{exchange}Ofi"] = ofi
                df[f"{exchange}OfiMa5"] = ofi.rolling(5).mean()
                df[f"{exchange}OfiMa20"] = ofi.rolling(20).mean()
                ofi_series[exchange] = ofi

            if notional_col in df.columns:
                notional_series[exchange] = df[notional_col]

        if ofi_series and notional_series:
            common_exchanges = set(ofi_series.keys()) & set(notional_series.keys())
            if common_exchanges:
                total_notional = sum(notional_series[ex] for ex in common_exchanges)
                weighted_ofi = sum(
                    ofi_series[ex]
                    * notional_series[ex]
                    / total_notional.replace(0, np.nan)
                    for ex in common_exchanges
                )
                df["aggregateOfi"] = weighted_ofi
                df["aggregateOfiMa5"] = weighted_ofi.rolling(5).mean()
                df["aggregateOfiMa20"] = weighted_ofi.rolling(20).mean()

        return df

    def get_threshold(
        self,
        metadata: dict | None = None,
        artifact: object | None = None,
    ) -> float:
        """Get decision threshold for take/pass."""
        if metadata and "threshold" in metadata:
            return float(metadata["threshold"])
        if artifact is not None:
            threshold = getattr(artifact, "json_data", {}).get("threshold")
            if threshold is not None:
                return float(threshold)
        return 0.5

    def _get_time_bounds(
        self, timestamp_to: datetime | None = None
    ) -> tuple[datetime, datetime] | None:
        """Get candle data bounds for this strategy."""
        queryset = CandleData.objects.filter(candle=self.candle)
        if timestamp_to is not None:
            queryset = queryset.filter(timestamp__lte=timestamp_to)
        first = queryset.order_by("timestamp").first()
        last = queryset.order_by("-timestamp").first()
        if not first or not last:
            return None
        return first.timestamp, last.timestamp + pd.Timedelta("1us")

    def _load_active_artifact(self) -> MLArtifact | None:
        artifacts = [
            artifact for artifact in self.artifacts.all() if artifact.is_active
        ]
        if not artifacts:
            return None
        artifacts.sort(key=lambda artifact: artifact.created_at, reverse=True)
        return artifacts[0]

    def _load_bundle(self, artifact: MLArtifact) -> dict:
        return joblib.load(artifact.file_data.path)

    def _build_model(self, *, penalty: str, c_value: float) -> Pipeline:
        """Build a logistic regression model pipeline."""
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=200,
                        penalty=penalty,
                        C=c_value,
                        solver="liblinear",
                    ),
                ),
            ]
        )

    def _get_bar_idx(self, events: DataFrame) -> np.ndarray:
        """Return bar indices for purging/embargo logic."""
        if "bar_idx" not in events.columns:
            raise ValueError("bar_idx is required for purged CV")
        return events["bar_idx"].to_numpy()

    def _get_event_end_exclusive_idx(self, events: DataFrame) -> np.ndarray | None:
        """Return exclusive end indices aligned to the event timeline."""
        exit_col = self.get_event_exit_column()
        if exit_col not in events.columns:
            return None

        event_time_col = self.get_event_time_column()
        event_times = pd.to_datetime(events[event_time_col])
        exit_times = pd.to_datetime(events[exit_col])
        if exit_times.isna().all():
            return None

        event_index = pd.Index(event_times)
        positions = event_index.searchsorted(
            exit_times.fillna(event_times), side="right"
        )

        bar_idx = self._get_bar_idx(events)
        max_bar = bar_idx[-1] + 1 if len(bar_idx) else 0
        end_idx = np.where(
            positions >= len(bar_idx),
            max_bar,
            bar_idx[np.clip(positions, 0, len(bar_idx) - 1)],
        )
        return end_idx

    def _train_model_with_cv(
        self,
        events: DataFrame,
        *,
        feature_cols: list[str] | None = None,
        label_col: str | None = None,
        cv_splits: int = 5,
        embargo_bars: int = 10,
        penalties: list[str] | None = None,
        c_values: list[float] | None = None,
        calibrate: bool = True,
        calibration_method: str = "auto",
    ) -> tuple[Pipeline | None, dict, object | None]:
        """Train a model with optional PurgedKFold tuning."""
        if events.empty:
            return None, {}, None

        event_time_col = self.get_event_time_column()
        events = events.sort_values(event_time_col)

        label_col = label_col or self.get_label_column()
        feature_cols = feature_cols or self.get_feature_columns(events)

        train_events = events.dropna(subset=[label_col])
        if train_events.empty:
            return None, {"feature_cols": feature_cols}, None

        X = train_events.reindex(columns=feature_cols, fill_value=0)
        y = train_events[label_col].fillna(0).astype(int)

        if y.nunique() < 2:
            logger.warning(
                "Skipping training for %s: only one class present", self.code_name
            )
            return None, {"feature_cols": feature_cols}, None

        penalties = [p.strip() for p in (penalties or ["l1", "l2"]) if p.strip()]
        penalties = [p for p in penalties if p in {"l1", "l2"}]
        c_values = [float(v) for v in (c_values or [0.1, 1.0, 10.0])]
        if not penalties or not c_values:
            return None, {"feature_cols": feature_cols}, None

        best_params: dict[str, object] | None = None
        best_score = float("inf")
        best_folds = 0

        event_end_idx = self._get_event_end_exclusive_idx(train_events)
        bar_idx = self._get_bar_idx(train_events)
        if cv_splits < 2:
            model = self._build_model(penalty="l2", c_value=1.0)
            model.fit(X, y)
            return model, {"feature_cols": feature_cols, "cv_score": None}, None

        cv = PurgedKFold(n_splits=cv_splits, embargo_bars=embargo_bars)
        try:
            folds = list(
                cv.split(X, y, event_end_exclusive_idx=event_end_idx, bar_idx=bar_idx)
            )
        except ValueError:
            folds = []

        if not folds:
            model = self._build_model(penalty="l2", c_value=1.0)
            model.fit(X, y)
            return model, {"feature_cols": feature_cols, "cv_score": None}, None

        for penalty in penalties:
            for c_value in c_values:
                scores: list[float] = []
                for train_idx, val_idx in folds:
                    X_train = X.iloc[train_idx]
                    y_train = y.iloc[train_idx]
                    if y_train.nunique() < 2:
                        continue
                    model = self._build_model(penalty=penalty, c_value=c_value)
                    model.fit(X_train, y_train)
                    X_val = X.iloc[val_idx]
                    y_val = y.iloc[val_idx]
                    proba = model.predict_proba(X_val)[:, 1]
                    scores.append(log_loss(y_val, proba, labels=[0, 1]))
                if scores:
                    mean_score = float(np.mean(scores))
                    if mean_score < best_score:
                        best_score = mean_score
                        best_params = {"penalty": penalty, "C": c_value}
                        best_folds = len(scores)

        if best_params is None:
            model = self._build_model(penalty="l2", c_value=1.0)
            model.fit(X, y)
            return model, {"feature_cols": feature_cols, "cv_score": None}, None

        calibrator = None
        calibrator_method = "none"
        if calibrate and folds:
            oof_probs: list[float] = []
            oof_true: list[int] = []
            for train_idx, val_idx in folds:
                X_train = X.iloc[train_idx]
                y_train = y.iloc[train_idx]
                if y_train.nunique() < 2:
                    continue
                model = self._build_model(
                    penalty=str(best_params["penalty"]),
                    c_value=float(best_params["C"]),
                )
                model.fit(X_train, y_train)
                X_val = X.iloc[val_idx]
                oof_probs.extend(model.predict_proba(X_val)[:, 1].tolist())
                oof_true.extend(y.iloc[val_idx].tolist())

            if oof_probs and oof_true:
                calibrator, calibrator_method = calibrate_probabilities(
                    np.array(oof_true),
                    np.array(oof_probs),
                    method=calibration_method,
                )

        model = self._build_model(
            penalty=str(best_params["penalty"]),
            c_value=float(best_params["C"]),
        )
        model.fit(X, y)
        return (
            model,
            {
                "feature_cols": feature_cols,
                "cv_score": best_score,
                "cv_folds": best_folds,
                "best_params": best_params,
                "calibration_method": calibrator_method,
            },
            calibrator,
        )

    def inference(self, candle_data: CandleData) -> "Signal | None":
        """Inference."""
        bounds = self._get_time_bounds(timestamp_to=candle_data.timestamp)
        if not bounds:
            return None
        timestamp_from, timestamp_to = bounds

        events = self.get_events(
            timestamp_from=timestamp_from,
            timestamp_to=timestamp_to,
            include_incomplete=True,
        )
        if events.empty:
            return None

        event_time_col = self.get_event_time_column()
        events = events.sort_values(event_time_col)
        latest = events.iloc[-1]
        event_time = latest[event_time_col]
        event_candle_data = latest.get("obj")
        if (
            self.last_candle_data_id
            and event_candle_data
            and event_candle_data.id == self.last_candle_data_id
        ):
            return None

        artifact = self._load_active_artifact()
        proba = None
        decision = Decision.TAKE
        feature_cols = []
        metadata = {}

        if artifact:
            if artifact.next_retrain_at and artifact.next_retrain_at <= timezone.now():
                logger.warning(
                    "MLArtifact %s for strategy %s is past next_retrain_at %s",
                    artifact.id,
                    self.code_name,
                    artifact.next_retrain_at,
                )
            bundle = self._load_bundle(artifact)
            model = bundle["model"]
            calibrator = bundle.get("calibrator")
            metadata = bundle.get("metadata", {})
            feature_cols = metadata.get("feature_cols") or self.get_feature_columns(
                events
            )
            threshold = self.get_threshold(metadata=metadata, artifact=artifact)

            feature_dict = {col: latest.get(col) for col in feature_cols}
            df_feat = pd.DataFrame([feature_dict]).fillna(0)
            proba = float(model.predict_proba(df_feat)[0, 1])
            calibration_method = metadata.get("calibration_method", "none")
            if calibrator and calibration_method != "none":
                proba = float(
                    apply_calibration(
                        np.array([proba]), calibrator, calibration_method
                    )[0]
                )
            decision = Decision.TAKE if proba >= threshold else Decision.PASS
        else:
            decision = Decision.TAKE

        json_data = {
            "event": self._clean_event_json(latest.to_dict()),
            "metadata": metadata,
        }
        signal = Signal.objects.create(
            strategy=self,
            timestamp=event_time,
            probability=proba,
            decision=decision,
            json_data=json_data,
        )

        if event_candle_data:
            self.last_candle_data = event_candle_data
            self.save(update_fields=["last_candle_data"])
        return signal

    def backtest(
        self,
        *,
        train_months: int = 24,
        test_months: int = 3,
        step_months: int = 3,
        lookback_hours: int | None = None,
        cv_splits: int = 5,
        embargo_bars: int = 10,
        penalties: list[str] | None = None,
        c_values: list[float] | None = None,
        calibration_method: str = "auto",
    ) -> dict | None:
        """Walk-forward backtest with rolling windows."""

        def init_stats() -> dict:
            return {
                "total": 0,
                "taken": 0,
                "wins": 0,
                "cum_net_return": Decimal("0"),
            }

        def update_stats(stats: dict, net_return, label, take: bool) -> None:
            stats["total"] += 1
            if not take:
                return
            stats["taken"] += 1
            if label == 1:
                stats["wins"] += 1
            if net_return is None or pd.isna(net_return):
                return
            if isinstance(net_return, Decimal):
                stats["cum_net_return"] += net_return
            else:
                stats["cum_net_return"] += Decimal(str(net_return))

        def finalize_stats(stats: dict) -> dict:
            total = stats["total"]
            taken = stats["taken"]
            wins = stats["wins"]
            cum_net = stats["cum_net_return"]
            take_rate = float(taken / total) if total else 0.0
            win_rate = float(wins / taken) if taken else 0.0
            avg_net = float(cum_net / taken) if taken else 0.0
            return {
                "total": total,
                "taken": taken,
                "take_rate": take_rate,
                "win_rate": win_rate,
                "avg_net_return": avg_net,
                "cum_net_return": float(cum_net),
            }

        bounds = self._get_time_bounds()
        if not bounds:
            return
        timestamp_from, timestamp_to = bounds
        if lookback_hours:
            timestamp_from = timestamp_to - pd.Timedelta(hours=lookback_hours)

        events = self.get_events(
            timestamp_from=timestamp_from,
            timestamp_to=timestamp_to,
            include_incomplete=False,
        )
        if events.empty:
            return

        baseline_stats = init_stats()
        meta_stats = init_stats()
        windows = {"evaluated": 0, "skipped": 0}

        event_time_col = self.get_event_time_column()
        label_col = self.get_label_column()
        events = events.sort_values(event_time_col)
        window_start = events[event_time_col].min()
        window_end = events[event_time_col].max()

        train_start = window_start
        train_end = train_start + pd.DateOffset(months=train_months)

        while train_end < window_end:
            test_end = train_end + pd.DateOffset(months=test_months)
            train_events = events[
                (events[event_time_col] >= train_start)
                & (events[event_time_col] < train_end)
            ]
            test_events = events[
                (events[event_time_col] >= train_end)
                & (events[event_time_col] < test_end)
            ]

            train_events = train_events.dropna(subset=[label_col])
            if train_events.empty or test_events.empty:
                windows["skipped"] += 1
                train_start += pd.DateOffset(months=step_months)
                train_end += pd.DateOffset(months=step_months)
                continue

            model, metadata, calibrator = self._train_model_with_cv(
                train_events,
                feature_cols=self.get_feature_columns(train_events),
                label_col=label_col,
                cv_splits=cv_splits,
                embargo_bars=embargo_bars,
                penalties=penalties,
                c_values=c_values,
                calibrate=calibration_method != "none",
                calibration_method=calibration_method,
            )
            if model is None:
                windows["skipped"] += 1
                train_start += pd.DateOffset(months=step_months)
                train_end += pd.DateOffset(months=step_months)
                continue

            windows["evaluated"] += 1

            feature_cols = metadata.get("feature_cols") or self.get_feature_columns(
                train_events
            )
            X_test = test_events.reindex(columns=feature_cols, fill_value=0)
            probs = model.predict_proba(X_test)[:, 1]
            calibration_method = metadata.get("calibration_method", "none")
            if calibrator and calibration_method != "none":
                probs = apply_calibration(probs, calibrator, calibration_method)
            threshold = self.get_threshold()

            signals = []
            for idx, row in test_events.iterrows():
                proba = float(probs[test_events.index.get_loc(idx)])
                decision = Decision.TAKE if proba >= threshold else Decision.PASS
                net_return = row.get("net_return")
                label = row.get(label_col)
                update_stats(baseline_stats, net_return, label, True)
                update_stats(meta_stats, net_return, label, decision == Decision.TAKE)
                signals.append(
                    Signal(
                        strategy=self,
                        timestamp=row[event_time_col],
                        probability=proba,
                        decision=decision,
                        json_data={"event": self._clean_event_json(row.to_dict())},
                    )
                )

            if signals:
                Signal.objects.bulk_create(signals)

            train_start += pd.DateOffset(months=step_months)
            train_end += pd.DateOffset(months=step_months)

        summary = {
            "baseline": finalize_stats(baseline_stats),
            "meta": finalize_stats(meta_stats),
            "windows": windows,
        }
        json_data = self.json_data or {}
        json_data["backtest_summary"] = summary
        self.json_data = json_data
        self.save(update_fields=["json_data"])
        return summary

    def on_signal(self, candle_data: CandleData, **kwargs) -> None:
        """On signal."""
        raise NotImplementedError

    def __str__(self) -> str:
        """str."""
        return self.code_name

    class Meta:
        db_table = "quant_tick_strategy"
        verbose_name = _("strategy")
        verbose_name_plural = _("strategies")


class Signal(models.Model):
    """Signal."""

    strategy = models.ForeignKey(
        "quant_tick.Strategy",
        related_name="signals",
        on_delete=models.CASCADE,
        verbose_name=_("strategy"),
    )
    timestamp = models.DateTimeField(_("timestamp"), db_index=True)
    probability = models.FloatField(_("probability"), null=True, blank=True)
    decision = models.CharField(
        _("decision"), max_length=16, choices=Decision.choices, db_index=True
    )
    json_data = JSONField(_("json data"), default=dict)

    def __str__(self) -> str:
        """str."""
        code_name = self.strategy.code_name
        timestamp = self.timestamp.isoformat()
        return f"{code_name} - signal: {timestamp}"

    class Meta:
        db_table = "quant_tick_signal"
        verbose_name = _("signal")
        verbose_name_plural = _("signals")


class Position(models.Model):
    """Position."""

    strategy = models.ForeignKey(
        "quant_tick.Strategy",
        related_name="executions",
        on_delete=models.CASCADE,
        verbose_name=_("strategy"),
    )
    signal = models.OneToOneField(
        "quant_tick.Signal",
        related_name="position",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        verbose_name=_("signal"),
    )
    open_candle_data = models.ForeignKey(
        "quant_tick.CandleData",
        on_delete=models.CASCADE,
        related_name="open_positions",
        verbose_name=_("candle data"),
    )
    close_candle_data = models.ForeignKey(
        "quant_tick.CandleData",
        on_delete=models.CASCADE,
        related_name="close_positions",
        null=True,
        blank=True,
        verbose_name=_("candle data"),
    )
    json_data = JSONField(_("json data"), null=True)

    def __str__(self) -> str:
        """str."""
        code_name = self.strategy.code_name
        open_timestamp = self.open_candle_data.timestamp.isoformat()
        if self.close_candle_data:
            close_timestamp = self.close_candle_data.timestamp.isoformat()
            timestamp = f"{open_timestamp} - {close_timestamp}"
        else:
            timestamp = open_timestamp
        return f"{code_name} - position: {timestamp}"

    class Meta:
        db_table = "quant_tick_position"
        verbose_name = _("position")
        verbose_name_plural = _("positions")
