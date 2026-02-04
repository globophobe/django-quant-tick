from datetime import datetime
from decimal import Decimal

import pandas as pd
from pandas import DataFrame

from quant_tick.utils import gettext_lazy as _

from ..strategies import Strategy


class MACrossoverStrategy(Strategy):
    """MA crossover strategy."""

    @property
    def confirm_windows(self) -> list[int]:
        """Optional additional MA windows used for confirmation."""
        windows = []
        if self.json_data:
            if "confirm_windows" in self.json_data:
                windows = self.json_data.get("confirm_windows") or []
            elif "mid_window" in self.json_data:
                windows = [self.json_data.get("mid_window")]
        result = []
        for window in windows:
            if window is None:
                continue
            try:
                window = int(window)
            except (TypeError, ValueError):
                continue
            if window > 0:
                result.append(window)
        return sorted(set(result))

    @property
    def hysteresis_k(self) -> float:
        """Volatility multiplier for hysteresis threshold (0 = disabled)."""
        if self.json_data and self.json_data.get("hysteresis_k") is not None:
            return float(self.json_data["hysteresis_k"])
        return 0.0

    def get_data_frame(self, df: DataFrame) -> DataFrame:
        """Add MA columns to dataframe."""
        df = super().get_data_frame(df)

        moving_average_type = self.json_data["moving_average_type"]
        confirm_windows = self.confirm_windows
        if moving_average_type == "sma":
            df["fast_ma"] = df["close"].rolling(self.json_data["fast_window"]).mean()
            df["slow_ma"] = df["close"].rolling(self.json_data["slow_window"]).mean()
            for window in confirm_windows:
                df[f"confirm_ma_{window}"] = df["close"].rolling(window).mean()
        elif moving_average_type == "ema":
            df["fast_ma"] = df["close"].ewm(span=self.json_data["fast_window"]).mean()
            df["slow_ma"] = df["close"].ewm(span=self.json_data["slow_window"]).mean()
            for window in confirm_windows:
                df[f"confirm_ma_{window}"] = df["close"].ewm(span=window).mean()

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
        """Build MA crossover events with labels."""
        if data_frame is not None:
            df = data_frame.copy()
            # Compute MAs on provided dataframe
            df = self.get_data_frame(df)
        else:
            df = DataFrame(self.candle.get_candle_data(timestamp_from, timestamp_to))
            df = self.get_data_frame(df)
        if df.empty:
            return DataFrame()

        df["ma_diff"] = df["fast_ma"] - df["slow_ma"]

        # Add price distance from MAs (percentage)
        df["close_distance_fast_ma"] = (df["close"] - df["fast_ma"]) / df["close"]
        df["close_distance_slow_ma"] = (df["close"] - df["slow_ma"]) / df["close"]

        # Add MA slope (change over last 5 periods)
        df["fast_ma_slope"] = df["fast_ma"].diff(5) / df["fast_ma"].shift(5)
        df["slow_ma_slope"] = df["slow_ma"].diff(5) / df["slow_ma"].shift(5)

        feature_frame = self.compute_features(df)

        confirm_windows = self.confirm_windows
        ma_cols = ["fast_ma", "slow_ma"]
        for window in confirm_windows:
            ma_cols.append(f"confirm_ma_{window}")
        df = df.dropna(subset=ma_cols)
        if df.empty:
            return DataFrame()
        feature_frame = feature_frame.loc[df.index]
        extra_cols = [col for col in feature_frame.columns if col not in df]

        confirm_long = None
        confirm_short = None
        if confirm_windows:
            ma_windows = [
                (self.json_data["fast_window"], "fast_ma"),
                (self.json_data["slow_window"], "slow_ma"),
            ]
            ma_windows += [(window, f"confirm_ma_{window}") for window in confirm_windows]
            ma_windows = sorted(ma_windows, key=lambda x: x[0])
            ma_order = [col for _, col in ma_windows]
            confirm_long = pd.Series(True, index=df.index)
            confirm_short = pd.Series(True, index=df.index)
            for idx in range(len(ma_order) - 1):
                left = df[ma_order[idx]]
                right = df[ma_order[idx + 1]]
                confirm_long &= left > right
                confirm_short &= left < right

        if self.hysteresis_k == 0:
            # Original behavior: simple crossover
            directions = df["fast_ma"] > df["slow_ma"]
            base_direction = directions.astype(int) * 2 - 1
            if confirm_long is not None and confirm_short is not None:
                confirmed = pd.Series(index=df.index, dtype=float)
                confirmed[confirm_long] = 1
                confirmed[confirm_short] = -1
                if pd.isna(confirmed.iloc[0]):
                    confirmed.iloc[0] = base_direction.iloc[0]
                direction = confirmed.ffill()
            else:
                direction = base_direction
            df = df.assign(direction=direction.astype(int))
            crossover_mask = df["direction"] != df["direction"].shift(1)
            event_rows = df[crossover_mask]
        else:
            # Hysteresis: require volatility-scaled separation (Schmitt trigger)
            vol = pd.to_numeric(df["realizedVariance"], errors="coerce")
            vol = vol.clip(lower=0).pow(0.5)
            threshold = self.hysteresis_k * vol * df["close"]

            # Schmitt trigger: direction changes only when crossing INTO a zone
            direction = pd.Series(index=df.index, dtype=float)

            prev_threshold = threshold.shift(1)
            prev_ma_diff = df["ma_diff"].shift(1)
            crossed_into_long = (df["ma_diff"] > threshold) & (
                prev_ma_diff <= prev_threshold
            )
            crossed_into_short = (df["ma_diff"] < -threshold) & (
                prev_ma_diff >= -prev_threshold
            )
            if confirm_long is not None and confirm_short is not None:
                crossed_into_long &= confirm_long
                crossed_into_short &= confirm_short

            direction[crossed_into_long] = 1
            direction[crossed_into_short] = -1

            # Check if first bar is already beyond threshold (startup in trend)
            first_idx = df.index[0]
            if pd.isna(direction.iloc[0]):
                if df.loc[first_idx, "ma_diff"] > threshold.iloc[0] and (
                    confirm_long is None or confirm_long.loc[first_idx]
                ):
                    direction.iloc[0] = 1
                elif df.loc[first_idx, "ma_diff"] < -threshold.iloc[0] and (
                    confirm_short is None or confirm_short.loc[first_idx]
                ):
                    direction.iloc[0] = -1

            # Forward-fill maintains state; fill remaining with 0 (no position)
            direction = direction.ffill().fillna(0)

            if (direction == 0).all():
                # No crossings at all - no events with hysteresis
                return DataFrame()

            df = df.assign(direction=direction.astype(int))

            # Use fill_value to avoid spurious row-0 event when direction is 0
            crossover_mask = df["direction"] != df["direction"].shift(1, fill_value=0)
            event_rows = df[crossover_mask]

        if event_rows.empty:
            return DataFrame()

        events: list[dict] = []
        event_indices = list(event_rows.index)
        cost_decimal = self.cost

        for i, idx in enumerate(event_indices):
            row = df.loc[idx]
            entry_ts = row["timestamp"]
            entry_price = Decimal(str(row["close"]))
            direction = int(row["direction"])

            next_idx = event_indices[i + 1] if i + 1 < len(event_indices) else None
            if next_idx is not None:
                exit_row = df.loc[next_idx]
                exit_price = Decimal(str(exit_row["close"]))
                gross_ret = direction * (exit_price / entry_price - 1)
                net_ret = gross_ret - cost_decimal
                exit_ts = exit_row["timestamp"]
            else:
                exit_price = None
                gross_ret = None
                net_ret = None
                exit_ts = None
                if not include_incomplete:
                    continue

            prev_idx = event_indices[i - 1] if i > 0 else None
            if prev_idx is not None:
                run_length_prev = idx - prev_idx
                run_duration_prev = (
                    row["timestamp"] - df.loc[prev_idx, "timestamp"]
                ).total_seconds()
            else:
                run_length_prev = None
                run_duration_prev = None

            events.append(
                {
                    "timestamp_event": entry_ts,
                    "timestamp_entry": entry_ts,
                    "timestamp_exit": exit_ts,
                    "direction": direction,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "gross_return": gross_ret,
                    "net_return": net_ret,
                    "label": int(net_ret > 0) if net_ret is not None else None,
                    "run_length_prev": run_length_prev,
                    "run_duration_prev_seconds": run_duration_prev,
                    "fast_ma": row["fast_ma"],
                    "slow_ma": row["slow_ma"],
                    "ma_diff": row["ma_diff"],
                    "close_distance_fast_ma": row["close_distance_fast_ma"],
                    "close_distance_slow_ma": row["close_distance_slow_ma"],
                    "fast_ma_slope": row.get("fast_ma_slope"),
                    "slow_ma_slope": row.get("slow_ma_slope"),
                    "candle_data_id": row.get("candle_data_id"),
                    "bar_index": row["bar_index"],
                    **{k: feature_frame.at[idx, k] for k in extra_cols},
                }
            )

        return DataFrame(events)

    class Meta:
        proxy = True
        verbose_name = _("ma crossover strategy")
        verbose_name_plural = _("ma crossover strategies")
