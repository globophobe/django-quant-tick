from datetime import datetime
from decimal import Decimal

import pandas as pd
from pandas import DataFrame

from quant_tick.utils import gettext_lazy as _

from ..strategies import Strategy


class DailyClusterTrendStrategy(Strategy):
    """Daily aggregated trend-following strategy.

    Aggregates cluster candles to daily bars, then applies price > MA filter.
    Long when daily close > MA, flat otherwise. Never short.

    EDA findings (sections 20-21 of EDA_FINDINGS.txt):
    - Daily aggregation reduces switches from ~213/yr to ~15/yr (14x less whipsaw)
    - Daily close > 30d SMA: +35,032% vs buy-and-hold +28,151%, maxDD 59%
    - Adding rate confirmation (14d/60d > 1.2) further reduces time in market

    json_data config:
    - ma_window: MA lookback in days (default 30)
    - ma_type: "sma" or "ema" (default "sma")
    - use_rate_confirmation: require rate fast/slow > threshold (default False)
    - rate_fast_window: fast rate lookback in days (default 14)
    - rate_slow_window: slow rate lookback in days (default 60)
    - rate_threshold: ratio threshold for confirmation (default 1.2)
    - cost: transaction cost as decimal

    Signal logic:
    1. Aggregate cluster candles to daily: count candles, last close, first open
    2. Compute rolling MA of daily closes
    3. Compute rolling candle rate ratio (if rate confirmation enabled)
    4. Long when: close > MA AND (rate_fast/rate_slow > threshold OR no confirmation)
    5. Flat otherwise (direction=0, not short)
    6. Entry/exit at next day's open
    """

    @property
    def ma_window(self) -> int:
        """MA lookback in days."""
        if self.json_data and self.json_data.get("ma_window") is not None:
            return int(self.json_data["ma_window"])
        return 30

    @property
    def ma_type(self) -> str:
        """MA type: 'sma' or 'ema'."""
        if self.json_data and self.json_data.get("ma_type"):
            return str(self.json_data["ma_type"]).lower()
        return "sma"

    @property
    def use_rate_confirmation(self) -> bool:
        """Whether to require rate confirmation."""
        if self.json_data and self.json_data.get("use_rate_confirmation") is not None:
            return bool(self.json_data["use_rate_confirmation"])
        return False

    @property
    def rate_fast_window(self) -> int:
        """Fast rate lookback in days."""
        if self.json_data and self.json_data.get("rate_fast_window") is not None:
            return int(self.json_data["rate_fast_window"])
        return 14

    @property
    def rate_slow_window(self) -> int:
        """Slow rate lookback in days."""
        if self.json_data and self.json_data.get("rate_slow_window") is not None:
            return int(self.json_data["rate_slow_window"])
        return 60

    @property
    def rate_threshold(self) -> float:
        """Rate ratio threshold for confirmation."""
        if self.json_data and self.json_data.get("rate_threshold") is not None:
            return float(self.json_data["rate_threshold"])
        return 1.2

    def _aggregate_to_daily(self, df: DataFrame) -> DataFrame:
        """Aggregate cluster candles to daily bars."""
        if df.empty:
            return DataFrame()

        df = df.copy()
        df["date"] = pd.to_datetime(df["timestamp"]).dt.date

        # Aggregate OHLCV
        daily = (
            df.groupby("date")
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "timestamp": "last",
                }
            )
            .reset_index()
        )

        # Add volume if present
        if "volume" in df.columns:
            vol_agg = df.groupby("date")["volume"].sum().reset_index()
            daily = daily.merge(vol_agg, on="date")

        # Candle count per day
        counts = df.groupby("date").size().reset_index(name="candle_count")
        daily = daily.merge(counts, on="date")

        # Sort by date
        daily = daily.sort_values("date").reset_index(drop=True)

        return daily

    def get_feature_columns(self, events: DataFrame) -> list[str]:
        """Feature columns for modeling."""
        base = super().get_feature_columns(events)
        return base + ["daily_candle_count", "ma_distance", "rate_ratio"]

    def get_events(
        self,
        *,
        timestamp_from: datetime | None = None,
        timestamp_to: datetime | None = None,
        data_frame: DataFrame | None = None,
        include_incomplete: bool = False,
        progress: bool = False,
    ) -> DataFrame:
        """Build daily trend events.

        Events are generated when position changes (flat→long or long→flat).
        Entry/exit at next day's open. Direction is always 1 (long only).
        """
        if data_frame is not None:
            df = data_frame.copy()
            df = self.get_data_frame(df)
        else:
            df = DataFrame(self.candle.get_candle_data(timestamp_from, timestamp_to))
            df = self.get_data_frame(df)

        if df.empty:
            return DataFrame()

        # Aggregate to daily
        daily = self._aggregate_to_daily(df)
        if daily.empty or len(daily) < self.ma_window + 1:
            return DataFrame()

        # Compute MA
        if self.ma_type == "ema":
            daily["ma"] = daily["close"].ewm(span=self.ma_window).mean()
        else:
            daily["ma"] = daily["close"].rolling(self.ma_window).mean()

        # Compute MA distance (for features)
        daily["ma_distance"] = (daily["close"] - daily["ma"]) / daily["close"]

        # Compute rate ratio (if enabled)
        daily["rate_ratio"] = None
        if self.use_rate_confirmation:
            daily["rate_fast"] = (
                daily["candle_count"].rolling(self.rate_fast_window).mean()
            )
            daily["rate_slow"] = (
                daily["candle_count"].rolling(self.rate_slow_window).mean()
            )
            daily["rate_ratio"] = daily["rate_fast"] / daily["rate_slow"].replace(0, 1)
            daily["signal"] = (daily["close"] > daily["ma"]) & (
                daily["rate_ratio"] > self.rate_threshold
            )
        else:
            daily["signal"] = daily["close"] > daily["ma"]

        # Position: 1 = long, 0 = flat
        daily["position"] = daily["signal"].fillna(False).astype(int)

        # Drop rows where MA not yet available
        daily = daily.dropna(subset=["ma"])
        if daily.empty:
            return DataFrame()

        daily = daily.reset_index(drop=True)
        n = len(daily)

        # Find position changes
        daily["prev_position"] = daily["position"].shift(1).fillna(0).astype(int)
        daily["position_change"] = daily["position"] != daily["prev_position"]

        # Build events
        events: list[dict] = []
        cost_decimal = self.cost
        i = 0
        prev_event_idx = None

        while i < n:
            if not daily.loc[i, "position_change"]:
                i += 1
                continue

            new_position = daily.loc[i, "position"]

            if new_position == 1:
                # Entering long position
                signal_idx = i
                entry_idx = i + 1
                if entry_idx >= n:
                    break

                entry_price = Decimal(str(daily.loc[entry_idx, "open"]))
                entry_ts = daily.loc[entry_idx, "timestamp"]
                signal_ts = daily.loc[signal_idx, "timestamp"]

                # Find exit: when position changes back to 0
                exit_signal_idx = None
                for j in range(signal_idx + 1, n):
                    if daily.loc[j, "position"] == 0:
                        exit_signal_idx = j
                        break

                exit_price = None
                exit_ts = None
                gross_ret = None
                net_ret = None

                if exit_signal_idx is not None:
                    exit_idx = exit_signal_idx + 1
                    if exit_idx < n:
                        exit_price = Decimal(str(daily.loc[exit_idx, "open"]))
                        exit_ts = daily.loc[exit_idx, "timestamp"]
                        gross_ret = exit_price / entry_price - 1
                        net_ret = gross_ret - cost_decimal
                    else:
                        exit_signal_idx = None

                if exit_signal_idx is None and not include_incomplete:
                    i += 1
                    continue

                run_length_prev = None
                run_duration_prev = None
                if prev_event_idx is not None:
                    run_length_prev = entry_idx - prev_event_idx
                    run_duration_prev = (
                        pd.Timestamp(entry_ts) - pd.Timestamp(daily.loc[prev_event_idx, "timestamp"])
                    ).total_seconds()

                event = {
                    "timestamp_event": signal_ts,
                    "timestamp_entry": entry_ts,
                    "timestamp_exit": exit_ts,
                    "direction": 1,  # Always long
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "gross_return": gross_ret,
                    "net_return": net_ret,
                    "label": int(net_ret > 0) if net_ret is not None else None,
                    "run_length_prev": run_length_prev,
                    "run_duration_prev_seconds": run_duration_prev,
                    "daily_candle_count": daily.loc[signal_idx, "candle_count"],
                    "ma_distance": daily.loc[signal_idx, "ma_distance"],
                    "rate_ratio": daily.loc[signal_idx, "rate_ratio"],
                    "bar_index": signal_idx,
                }
                events.append(event)
                prev_event_idx = entry_idx

                if exit_signal_idx is not None:
                    i = exit_signal_idx + 1
                else:
                    break
            else:
                # Position changed to flat, but we track events on entry, not exit
                i += 1

        return DataFrame(events)

    class Meta:
        proxy = True
        verbose_name = _("daily cluster trend strategy")
        verbose_name_plural = _("daily cluster trend strategies")
