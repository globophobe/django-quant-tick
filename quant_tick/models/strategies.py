from __future__ import annotations

import logging
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING

import joblib
import numpy as np
import pandas as pd
from django.db import models
from django.utils import timezone
from pandas import DataFrame
from polymorphic.models import PolymorphicModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from quant_tick.constants import Decision, Direction
from quant_tick.lib.calibration import apply_calibration, calibrate_probabilities
from quant_tick.lib.cross_validation import PurgedKFold
from quant_tick.utils import gettext_lazy as _

from .base import AbstractCodeName, JSONField
from .candles import CandleData
from .meta_labelling import MLArtifact

if TYPE_CHECKING:
    from .symbols import Symbol

logger = logging.getLogger(__name__)


class Strategy(AbstractCodeName, PolymorphicModel):
    """Strategy."""

    candle = models.ForeignKey(
        "quant_tick.Candle",
        on_delete=models.CASCADE,
        verbose_name=_("candle"),
    )
    last_candle_data = models.ForeignKey(
        "quant_tick.CandleData",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        verbose_name=_("last candle data"),
    )
    json_data = JSONField(_("json data"), null=True)
    is_active = models.BooleanField(_("is active"), default=True)

    @property
    def symbol(self) -> Symbol:
        """Symbol from candle."""
        return self.candle.symbol

    def get_data_frame(self, df: DataFrame) -> DataFrame:
        """Process dataframe."""
        if df.empty:
            return df
        return self.convert_types(df)

    def convert_types(self, df: DataFrame) -> DataFrame:
        """Convert Decimal columns to float for pandas operations."""
        numeric_cols = [
            col
            for col in [
                "open",
                "high",
                "low",
                "close",
                "volume",
                "buyVolume",
                "notional",
                "buyNotional",
                "realizedVariance",
                "roundVolume",
                "roundBuyVolume",
                "roundNotional",
                "roundBuyNotional",
            ]
            if col in df.columns
        ]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def get_events(
        self,
        *,
        timestamp_from: datetime | None = None,
        timestamp_to: datetime | None = None,
        data_frame: DataFrame | None = None,
        include_incomplete: bool = False,
        progress: bool = False,
    ) -> DataFrame:
        """Get events.

        Args:
            timestamp_from: Start timestamp
            timestamp_to: End timestamp
            data_frame: Candle dataFrame
            include_incomplete: Include incomplete events
            progress: Show progress bar

        Returns:
            DataFrame with events and features
        """
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

    @property
    def leverage(self) -> Decimal:
        """Position leverage multiplier."""
        if self.json_data and self.json_data.get("leverage") is not None:
            return Decimal(str(self.json_data["leverage"]))
        return Decimal("1")

    def get_feature_columns(self, events: DataFrame) -> list[str]:
        """Select feature columns for modeling."""
        return ["run_length_prev", "run_duration_prev_seconds"]

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

    def compute_features(self, df: DataFrame) -> DataFrame:
        """Compute default features for a strategy."""
        df = self.convert_types(df)
        return self._compute_features(df)

    def _compute_features(self, data_frame: DataFrame) -> DataFrame:
        """Compute features."""
        df = data_frame.copy()
        returns = df["close"].pct_change(fill_method=None)

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

            # Delta features (order flow)
            delta = 2 * df[buy_vol_col] - df[vol_col]  # buyVol - sellVol
            df["cumDelta20"] = delta.rolling(20).sum()
            df["cumDelta50"] = delta.rolling(50).sum()
            vol_sum = df[vol_col].rolling(20).sum()
            df["cumDeltaNorm"] = delta.rolling(20).sum() / vol_sum.replace(0, np.nan)
            df["deltaAccel"] = df["cumDelta20"].diff(5)

        # Tick features
        if "ticks" in df.columns and "volume" in df.columns:
            ticks_safe = df["ticks"].replace(0, 1)
            df["avgTradeSize"] = df["volume"] / ticks_safe
            df["avgTradeSizeMa20"] = df["avgTradeSize"].rolling(20).mean()
            avg_std = df["avgTradeSize"].rolling(20).std()
            df["avgTradeSizeZScore"] = (df["avgTradeSize"] - df["avgTradeSizeMa20"]) / (
                avg_std + 1e-8
            )

            df["tickRate"] = df["ticks"]
            tick_mean = df["tickRate"].rolling(20).mean()
            tick_std = df["tickRate"].rolling(20).std()
            df["tickRateZScore"] = (df["tickRate"] - tick_mean) / (tick_std + 1e-8)

        if "buyTicks" in df.columns and "ticks" in df.columns:
            df["buyTickRatio"] = df["buyTicks"] / df["ticks"].replace(0, np.nan)

        # Notional features
        if "notional" in df.columns and "ticks" in df.columns:
            ticks_safe = df["ticks"].replace(0, 1)
            df["avgNotionalPerTick"] = df["notional"] / ticks_safe
            df["avgNotionalPerTickMa20"] = df["avgNotionalPerTick"].rolling(20).mean()
            notional_std = df["avgNotionalPerTick"].rolling(20).std()
            df["avgNotionalPerTickZScore"] = (
                df["avgNotionalPerTick"] - df["avgNotionalPerTickMa20"]
            ) / (notional_std + 1e-8)

        if "buyNotional" in df.columns and "notional" in df.columns:
            df["buyNotionalRatio"] = df["buyNotional"] / df["notional"].replace(
                0, np.nan
            )

        if "notional" in df.columns and "volume" in df.columns:
            df["notionalVsVolume"] = df["notional"] / df["volume"].replace(0, np.nan)

        # Round lot features (institutional activity)
        if "roundVolume" in df.columns and "volume" in df.columns:
            df["roundVolumeRatio"] = df["roundVolume"] / df["volume"].replace(0, np.nan)

        if "roundBuyVolume" in df.columns and "roundVolume" in df.columns:
            df["roundBuyRatio"] = df["roundBuyVolume"] / df["roundVolume"].replace(
                0, np.nan
            )

        # Price action features
        if all(c in df.columns for c in ["high", "low", "open", "close"]):
            bar_range = df["high"] - df["low"]
            df["barRange"] = bar_range / df["close"]
            df["bodyRatio"] = (df["close"] - df["open"]).abs() / bar_range.replace(
                0, np.nan
            )
            body_top = df[["open", "close"]].max(axis=1)
            body_bottom = df[["open", "close"]].min(axis=1)
            df["upperWickRatio"] = (df["high"] - body_top) / bar_range.replace(
                0, np.nan
            )
            df["lowerWickRatio"] = (body_bottom - df["low"]) / bar_range.replace(
                0, np.nan
            )

        # Volatility features (realizedVariance)
        if "realizedVariance" in df.columns:
            var_mean = df["realizedVariance"].rolling(20).mean()
            var_std = df["realizedVariance"].rolling(20).std()
            df["varianceRatio"] = df["realizedVariance"] / var_mean.replace(0, np.nan)
            df["varianceZScore"] = (df["realizedVariance"] - var_mean) / (
                var_std + 1e-8
            )

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

    def _get_bar_index(self, events: DataFrame) -> np.ndarray:
        """Return bar indices for purging/embargo logic."""
        if "bar_index" not in events.columns:
            raise ValueError("bar_index is required for purged CV")
        return events["bar_index"].to_numpy()

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

        bar_index = self._get_bar_index(events)
        max_bar = bar_index[-1] + 1 if len(bar_index) else 0
        end_idx = np.where(
            positions >= len(bar_index),
            max_bar,
            bar_index[np.clip(positions, 0, len(bar_index) - 1)],
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
        nan_mask = X.isna().any(axis=1)
        if nan_mask.any():
            train_events = train_events.loc[~nan_mask].copy()
            if train_events.empty:
                return None, {"feature_cols": feature_cols}, None
            X = X.loc[~nan_mask]
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
        bar_index = self._get_bar_index(train_events)
        if cv_splits < 2:
            model = self._build_model(penalty="l2", c_value=1.0)
            model.fit(X, y)
            return model, {"feature_cols": feature_cols, "cv_score": None}, None

        cv = PurgedKFold(n_splits=cv_splits, embargo_bars=embargo_bars)
        try:
            folds = list(
                cv.split(
                    X, y, event_end_exclusive_idx=event_end_idx, bar_index=bar_index
                )
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
        oof_probs: list[float] = []
        oof_true: list[int] = []
        oof_idx: list[int] = []

        # Always collect OOF predictions when folds exist
        if folds:
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
                oof_idx.extend(val_idx.tolist())
                oof_probs.extend(model.predict_proba(X_val)[:, 1].tolist())
                oof_true.extend(y.iloc[val_idx].tolist())

            # Only calibrate if requested
            if calibrate and oof_probs and oof_true:
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
                "oof_idx": oof_idx if oof_idx else None,
                "oof_probs": oof_probs if oof_probs else None,
                "oof_true": oof_true if oof_true else None,
            },
            calibrator,
        )

    def inference(self, candle_data: CandleData) -> Signal | None:
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
        event_candle_data_id = latest.get("candle_data_id")
        if event_candle_data_id == self.last_candle_data_id:
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

        if event_candle_data_id:
            self.last_candle_data_id = event_candle_data_id
            self.save(update_fields=["last_candle_data"])
        return signal

    def on_signal(
        self,
        candle_data: CandleData,
        direction: Direction,
        position: Position | None = None,
        data: dict | None = None,
    ) -> Position | None:
        """On signal."""
        data = data or {}
        if position:
            if position.json_data["direction"] != direction.value:
                position.close_candle_data = candle_data
                position.save()
                position = Position.objects.create(
                    strategy=self,
                    open_candle_data=candle_data,
                    close_candle_data=None,
                    json_data={"direction": direction.value, **data},
                )
        else:
            position = Position.objects.create(
                strategy=self,
                open_candle_data=candle_data,
                close_candle_data=None,
                json_data={"direction": direction.value, **data},
            )
        return position

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
