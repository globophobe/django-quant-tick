from datetime import datetime
from decimal import Decimal

from django.db.models import QuerySet
from pandas import DataFrame

from quant_tick.constants import Direction
from quant_tick.utils import gettext_lazy as _

from ..candles import CandleData
from ..strategies import Position, Strategy


class MACrossoverStrategy(Strategy):
    """MA crossover strategy."""

    def get_data_frame(self, queryset: QuerySet) -> DataFrame:
        """Get data frame."""
        data_frame = super().get_data_frame(queryset)
        moving_average_type = self.json_data["moving_average_type"]
        if moving_average_type == "sma":
            data_frame["fast_ma"] = (
                data_frame["close"].rolling(self.json_data["fast_window"]).mean()
            )
            data_frame["slow_ma"] = (
                data_frame["close"].rolling(self.json_data["slow_window"]).mean()
            )
        elif moving_average_type == "ema":
            data_frame["fast_ma"] = (
                data_frame["close"].ewm(span=self.json_data["fast_window"]).mean()
            )
            data_frame["slow_ma"] = (
                data_frame["close"].ewm(span=self.json_data["slow_window"]).mean()
            )
        return data_frame

    def get_events(
        self,
        *,
        timestamp_from: datetime,
        timestamp_to: datetime,
        include_incomplete: bool = False,
    ) -> DataFrame:
        """Build MA crossover events with labels."""
        queryset = CandleData.objects.filter(
            candle=self.candle, timestamp__gte=timestamp_from, timestamp__lt=timestamp_to
        )
        data_frame = self.get_data_frame(queryset)
        if data_frame.empty:
            return DataFrame()

        data_frame = data_frame.dropna(subset=["fast_ma", "slow_ma"])
        if data_frame.empty:
            return DataFrame()

        directions = data_frame["fast_ma"] > data_frame["slow_ma"]
        data_frame = data_frame.assign(direction=directions.astype(int) * 2 - 1)

        crossover_mask = data_frame["direction"] != data_frame["direction"].shift(1)
        event_rows = data_frame[crossover_mask]
        if event_rows.empty:
            return DataFrame()

        events: list[dict] = []
        event_indices = list(event_rows.index)
        cost_decimal = self.cost

        for i, idx in enumerate(event_indices):
            row = data_frame.loc[idx]
            entry_ts = row["timestamp"]
            entry_price = Decimal(str(row["close"]))
            direction = int(row["direction"])

            next_idx = event_indices[i + 1] if i + 1 < len(event_indices) else None
            if next_idx is not None:
                exit_row = data_frame.loc[next_idx]
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
            run_length_prev = None
            run_duration_prev = None
            if prev_idx is not None:
                run_length_prev = idx - prev_idx
                run_duration_prev = (
                    row["timestamp"] - data_frame.loc[prev_idx, "timestamp"]
                ).total_seconds()

            candle_data = row.get("obj")
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
                    "feat_fast_ma": row["fast_ma"],
                    "feat_slow_ma": row["slow_ma"],
                    "feat_ma_diff": row["fast_ma"] - row["slow_ma"],
                    "feat_close": row["close"],
                    "obj": candle_data,
                    "bar_idx": row["bar_idx"],
                }
            )

        return DataFrame(events)

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

    class Meta:
        proxy = True
        verbose_name = _("ma crossover strategy")
        verbose_name_plural = _("ma crossover strategies")
