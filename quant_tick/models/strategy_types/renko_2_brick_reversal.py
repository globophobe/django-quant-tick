from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

import pandas as pd
from pandas import DataFrame

from quant_tick.constants import Direction, RenkoKind
from quant_tick.utils import gettext_lazy as _

from ..candles import CandleData
from ..strategies import Position, Strategy


@dataclass(frozen=True)
class BodyRow:
    """Parsed Renko body row."""

    timestamp: datetime
    close: Decimal
    direction: int
    sequence: int


def _iter_bodies(df: DataFrame) -> list[BodyRow]:
    """Return ordered body rows from a Renko DataFrame."""
    bodies: list[BodyRow] = []
    for index, row in df.iterrows():
        if row.get("renko_kind") != RenkoKind.BODY:
            continue
        sequence = row.get("renko_sequence")
        if sequence is None:
            continue
        try:
            close = Decimal(str(row["close"]))
        except Exception:
            continue
        bodies.append(
            BodyRow(
                timestamp=row["timestamp"],
                close=close,
                direction=int(row["renko_direction"]),
                sequence=int(sequence),
            )
        )
    return bodies


@dataclass(frozen=True)
class EventBodyRow:
    """Parsed body brick row for event datasets."""

    timestamp: datetime
    close: Decimal
    direction: int
    run_idx: int
    idx_in_run: int
    raw: dict
    feature_row: dict


def _iter_event_bodies(df: DataFrame) -> list[EventBodyRow]:
    """Return ordered body rows parsed from a Renko DataFrame."""
    bodies: list[EventBodyRow] = []
    run_idx = -1
    last_dir: int | None = None

    for index, row in df.iterrows():
        if row.get("renko_kind") != RenkoKind.BODY:
            continue

        direction = int(row["renko_direction"])
        row_dict = row.to_dict()
        if last_dir != direction:
            run_idx += 1
            idx_in_run = 0
            last_dir = direction
        else:
            idx_in_run = bodies[-1].idx_in_run + 1 if bodies else 0

        bodies.append(
            EventBodyRow(
                timestamp=row["timestamp"],
                close=Decimal(str(row["close"])),
                direction=direction,
                run_idx=run_idx,
                idx_in_run=idx_in_run,
                raw=row_dict,
                feature_row=row.get("_feature_row", {}),
            )
        )

    return bodies


def _flatten_exchange_data(exchanges: dict) -> dict:
    """Flatten nested exchange data to columns like binanceClose, coinbaseVolume."""
    if not isinstance(exchanges, dict):
        return {}

    flat: dict[str, object] = {}
    for exchange, data in exchanges.items():
        if not isinstance(data, dict):
            continue
        for key, value in data.items():
            col_name = f"{exchange}{key[0].upper()}{key[1:]}" if key else exchange
            flat[col_name] = value
    return flat


def _exchange_features(entry_row: EventBodyRow) -> dict:
    """Compute simple exchange dispersion + flatten per-exchange fields."""
    exchanges = (entry_row.raw or {}).get("exchanges")
    if not exchanges:
        return {}

    flat = _flatten_exchange_data(exchanges)

    closes: list[Decimal] = []
    for data in exchanges.values():
        if isinstance(data, dict) and data.get("close") is not None:
            try:
                closes.append(Decimal(str(data["close"])))
            except Exception:
                continue

    dispersion = None
    if closes:
        dispersion = max(closes) - min(closes)

    return {
        "exch_count": len(exchanges),
        "exch_dispersion_close": dispersion,
        **flat,
    }


def _build_runs(bodies: Sequence[EventBodyRow]) -> list[list[EventBodyRow]]:
    """Group consecutive bodies with the same direction."""
    runs: list[list[EventBodyRow]] = []
    for body in bodies:
        if not runs or runs[-1][0].direction != body.direction:
            runs.append([body])
        else:
            runs[-1].append(body)
    return runs


def build_event_dataset_for_candle(
    candle,
    *,
    timestamp_from: datetime,
    timestamp_to: datetime,
    cost: Decimal = Decimal("0"),
    include_incomplete: bool = False,
    compute_features: Callable[[DataFrame], DataFrame] | None = None,
) -> DataFrame:
    """Build an in-memory event dataset for Renko 2-brick reversals."""
    df_raw = candle.get_candle_data(
        timestamp_from=timestamp_from,
        timestamp_to=timestamp_to,
        is_complete=False,
    )
    if df_raw.empty:
        return pd.DataFrame()

    flat_rows: list[dict] = []
    for row in df_raw.to_dict(orient="records"):
        exchanges = row.get("exchanges")
        if isinstance(exchanges, dict) and exchanges:
            row = {**row, **_flatten_exchange_data(exchanges)}
        flat_rows.append(row)

    flat_df = pd.DataFrame(flat_rows)
    if compute_features is None:
        features_df = flat_df.copy()
    else:
        try:
            features_df = compute_features(flat_df.copy())
        except Exception:
            features_df = flat_df.copy()

    flat_df["_feature_row"] = features_df.to_dict(orient="records")

    bodies = _iter_event_bodies(flat_df)
    if not bodies:
        return pd.DataFrame()

    runs = _build_runs(bodies)
    events: list[dict] = []
    cost_decimal = Decimal(str(cost))

    for i, run in enumerate(runs):
        if len(run) < 2:
            continue

        prev_run = runs[i - 1] if i > 0 else None
        next_run = runs[i + 1] if i + 1 < len(runs) else None

        entry_body = run[1]
        event_ts = run[0].timestamp
        entry_price = entry_body.close

        if next_run:
            exit_body = next_run[0]
            exit_price = exit_body.close
            has_exit = True
        else:
            exit_body = None
            exit_price = None
            has_exit = False

        if not has_exit and not include_incomplete:
            continue

        direction = entry_body.direction
        if has_exit and exit_price is not None:
            gross_ret = direction * (Decimal(exit_price) / Decimal(entry_price) - 1)
            net_ret = gross_ret - cost_decimal
        else:
            gross_ret = None
            net_ret = None

        run_length_prev = len(prev_run) if prev_run else None
        run_duration_prev = (
            (run[0].timestamp - prev_run[0].timestamp).total_seconds()
            if prev_run
            else None
        )

        events.append(
            {
                "timestamp_event": event_ts,
                "timestamp_entry": entry_body.timestamp,
                "timestamp_exit": exit_body.timestamp if exit_body else None,
                "direction": direction,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "gross_return": gross_ret,
                "net_return": net_ret,
                "label": int(net_ret > 0) if net_ret is not None else None,
                "run_length_prev": run_length_prev,
                "run_duration_prev_seconds": run_duration_prev,
                "obj": entry_body.raw.get("obj") if entry_body.raw else None,
                "bar_idx": entry_body.raw.get("bar_idx") if entry_body.raw else None,
                **_exchange_features(entry_body),
                **{
                    f"feat_{k}": v
                    for k, v in entry_body.feature_row.items()
                    if k not in {"timestamp", "obj"}
                },
            }
        )

    return pd.DataFrame(events)


class Renko2BrickReversalStrategy(Strategy):
    """Renko 2-brick reversal strategy."""

    def get_data_frame(
        self, timestamp_from: datetime, timestamp_to: datetime
    ) -> DataFrame:
        """Get data frame."""
        return self.candle.get_candle_data(
            timestamp_from=timestamp_from,
            timestamp_to=timestamp_to,
            is_complete=True,
        )

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

    def _map_candle_data(self, sequences: set[int]) -> dict[int, CandleData]:
        """Map Renko sequences to CandleData rows."""
        if not sequences:
            return {}
        queryset = CandleData.objects.filter(
            candle=self.candle, renko_data__sequence__in=sequences
        ).select_related("renko_data")
        mapping: dict[int, CandleData] = {}
        for candle_data in queryset:
            try:
                sequence = candle_data.renko_data.sequence
            except Exception:
                continue
            mapping[int(sequence)] = candle_data
        return mapping

    def _run(self, timestamp_from: datetime, timestamp_to: datetime) -> None:
        """Run strategy logic for a time window."""
        data_frame = self.get_data_frame(timestamp_from, timestamp_to)
        if data_frame.empty:
            return

        bodies = _iter_bodies(data_frame)
        if not bodies:
            return

        candle_map = self._map_candle_data({b.sequence for b in bodies})
        position = (
            Position.objects.filter(strategy=self, close_candle_data__isnull=True)
            .select_related("open_candle_data")
            .first()
        )

        last_direction: int | None = None
        run_length = 0

        for body in bodies:
            if last_direction != body.direction:
                run_length = 1
                last_direction = body.direction
                if position:
                    exit_candle = candle_map.get(body.sequence)
                    if exit_candle:
                        position.close_candle_data = exit_candle
                        position.save()
                        position = None
            else:
                run_length += 1

            if run_length == 2 and position is None:
                entry_candle = candle_map.get(body.sequence)
                if not entry_candle:
                    continue
                direction = Direction.LONG if body.direction == 1 else Direction.SHORT
                position = Position.objects.create(
                    strategy=self,
                    open_candle_data=entry_candle,
                    close_candle_data=None,
                    json_data={"direction": direction.value},
                )

    def get_events(
        self,
        *,
        timestamp_from: datetime,
        timestamp_to: datetime,
        include_incomplete: bool = False,
    ) -> DataFrame:
        """Build a Renko 2-brick event dataset for training/simulation."""
        return build_event_dataset_for_candle(
            self.candle,
            timestamp_from=timestamp_from,
            timestamp_to=timestamp_to,
            cost=self.cost,
            include_incomplete=include_incomplete,
            compute_features=self.compute_features,
        )

    class Meta:
        proxy = True
        verbose_name = _("renko 2-brick reversal strategy")
        verbose_name_plural = _("renko 2-brick reversal strategies")
