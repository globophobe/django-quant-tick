from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Protocol

import pandas as pd
from pandas import DataFrame

from quant_tick.lib.ml import _compute_features

if TYPE_CHECKING:  # pragma: no cover
    from quant_tick.models import MetaModel


class CandleLike(Protocol):
    """Candle like."""

    def get_candle_data(
        self,
        timestamp_from: datetime,
        timestamp_to: datetime,
        limit: int | None = None,
        is_complete: bool = True,
    ) -> DataFrame: ...


@dataclass(frozen=True)
class BodyRow:
    """Parsed body brick row."""

    timestamp: datetime
    close: Decimal
    direction: int  # +1 or -1
    run_idx: int
    idx_in_run: int  # 0-based position inside the run
    raw: dict
    feature_row: dict


def _iter_bodies(df: DataFrame) -> list[BodyRow]:
    """Return ordered body rows parsed from a Renko DataFrame."""
    bodies: list[BodyRow] = []
    run_idx = -1
    last_dir: int | None = None

    for idx, row in df.iterrows():
        if row.get("renko_kind") != "body":
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
            BodyRow(
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


def _exchange_features(entry_row: BodyRow) -> dict:
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


def _build_runs(bodies: Sequence[BodyRow]) -> list[list[BodyRow]]:
    """Group consecutive bodies with the same direction."""
    runs: list[list[BodyRow]] = []
    for body in bodies:
        if not runs or runs[-1][0].direction != body.direction:
            runs.append([body])
        else:
            runs[-1].append(body)
    return runs


def build_event_dataset(
    candle: CandleLike,
    timestamp_from: datetime,
    timestamp_to: datetime,
    *,
    cost: float | Decimal = 0,
    include_incomplete: bool = False,
    meta_model: "MetaModel | None" = None,
) -> DataFrame:
    """Build an in-memory event dataset for meta-labelling.

    Args:
        candle: A RenkoBrick (or any Candle-like with get_candle_data)
        timestamp_from: Start time
        timestamp_to: End time
        cost: Per-trade cost to subtract from returns (round-trip), default 0
        include_incomplete: If True, keep the last event even if there is no exit yet

    Returns:
        DataFrame with one row per reversal event (confirmed by 2-body rule):
        columns include timestamp_event, timestamp_entry, direction, entry_price,
        exit_price, gross_return, net_return, run_length_prev, run_duration_prev_seconds.
    """
    df_raw = candle.get_candle_data(
        timestamp_from=timestamp_from,
        timestamp_to=timestamp_to,
        is_complete=False,  # we decide on incompleteness ourselves
    )
    if df_raw.empty:
        return pd.DataFrame()

    # Flatten exchanges so quant_core can compute multi-exchange features
    canonical_exchange: str | None = (
        meta_model.symbol.exchange if meta_model and meta_model.symbol else None
    )
    flat_rows: list[dict] = []
    for row in df_raw.to_dict(orient="records"):
        exchanges = row.get("exchanges")
        if isinstance(exchanges, dict) and exchanges:
            row = {**row, **_flatten_exchange_data(exchanges)}
        flat_rows.append(row)

    flat_df = pd.DataFrame(flat_rows)
    try:
        features_df = _compute_features(
            flat_df.copy(), canonical_exchange=canonical_exchange
        )
    except Exception:
        features_df = flat_df.copy()

    # Attach feature rows back to flat_df for positional alignment
    flat_df["_feature_row"] = features_df.to_dict(orient="records")

    bodies = _iter_bodies(flat_df)
    if not bodies:
        return pd.DataFrame()

    runs = _build_runs(bodies)
    events: list[dict] = []
    cost_decimal = Decimal(str(cost))

    for i, run in enumerate(runs):
        if len(run) < 2:
            continue  # need 2-body confirmation for entry

        prev_run = runs[i - 1] if i > 0 else None
        next_run = runs[i + 1] if i + 1 < len(runs) else None

        # Entry: second body in this run (confirmation)
        entry_body = run[1]
        event_ts = run[0].timestamp  # flip happened at first body of this run
        entry_price = entry_body.close

        # Exit: first body of the next run (next reversal)
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
                **_exchange_features(entry_body),
                **{
                    f"feat_{k}": v
                    for k, v in entry_body.feature_row.items()
                    if k != "timestamp"
                },
            }
        )

    return pd.DataFrame(events)
