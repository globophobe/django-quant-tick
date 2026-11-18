from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pandas as pd
from pandas import DataFrame

from quant_tick.lib import iter_chunks, to_utc_datetime

from ..binance_futures.market_history import HISTORY_EXHAUSTED_ATTR
from ..quicknode.solana import available as solana_available
from .api import get_phoenix_response
from .constants import MARKET_HISTORY_INTERVAL, MARKET_HISTORY_MAX_RESULTS
from .positioning import phoenix_positioning
from .trades import has_trades

MARKET_HISTORY_COLUMNS = [
    "open_interest",
    "open_interest_value",
    "open_interest_unit",
    "mark_price",
    "spot_price",
    "slot",
    "market_history_interval",
    "market_history_source",
    "market_history_timestamp_convention",
    "single_open_interest",
    "open_interest_basis",
    "mark_spot_basis_bps",
    "cumulative_maker_fees",
    "cumulative_taker_fees",
]


def phoenix_market_history(
    api_symbol: str,
    timestamp_from: datetime,
    timestamp_to: datetime,
    *,
    position_history=None,
) -> DataFrame:
    """Fetch completed hourly open-interest buckets, retaining gaps.

    Phoenix reports base-asset quantity; quote value is quantity times the
    accompanying mark price. Its bucket-start timestamp can precede the sampled
    state, so an observation is available only after that hour closes. Preserve
    the venue's slot and prices alongside the interval-start convention. An
    optional Solana endpoint enriches each hour with validated trade flows;
    incomplete RPC evidence raises and leaves existing observations untouched.
    The collector supplies position_history to reconstruct hour-end cohorts and
    reuse bulk transactions, including separate backstop and ADL quantities.
    Fill-only requests cannot establish their absence. The collector requests
    one hour at a time, newest first.
    """
    symbol = api_symbol.strip().upper()
    interval = pd.Timedelta(MARKET_HISTORY_INTERVAL)
    start = pd.Timestamp(to_utc_datetime(timestamp_from)).ceil(MARKET_HISTORY_INTERVAL)
    end = pd.Timestamp(to_utc_datetime(timestamp_to)).floor(MARKET_HISTORY_INTERVAL)
    rows = []
    previous_timestamp = None
    for chunk_from, chunk_to in iter_chunks(
        start, end, value=interval * (MARKET_HISTORY_MAX_RESULTS - 1)
    ):
        result = get_phoenix_response(
            f"/v1/market/{symbol}/stats",
            {
                "start_time": chunk_from.isoformat(),
                "end_time": (chunk_to - timedelta(microseconds=1)).isoformat(),
                "timeframe": MARKET_HISTORY_INTERVAL,
                "limit": MARKET_HISTORY_MAX_RESULTS,
            },
        )
        if result["symbol"] != symbol or result["timeframe"] != MARKET_HISTORY_INTERVAL:
            raise ValueError(
                "Phoenix market stats returned an unexpected market or timeframe"
            )
        if len(result["stats"]) >= MARKET_HISTORY_MAX_RESULTS:
            raise ValueError("Phoenix market stats window reached the response limit")
        for point in result["stats"]:
            timestamp = pd.Timestamp(to_utc_datetime(point["timestamp"]))
            if not chunk_from <= timestamp < chunk_to:
                raise ValueError(
                    "Phoenix market stats are outside the requested window"
                )
            if timestamp != timestamp.floor(MARKET_HISTORY_INTERVAL) or (
                previous_timestamp is not None and timestamp <= previous_timestamp
            ):
                raise ValueError(
                    "Phoenix market stats timestamps are not ordered hourly buckets"
                )
            previous_timestamp = timestamp
            quantities = {
                field: Decimal(str(point[field]))
                for field in ("open_interest", "mark_price", "spot_price")
            }
            if (
                not all(value.is_finite() for value in quantities.values())
                or quantities["open_interest"] < 0
                or quantities["mark_price"] <= 0
                or quantities["spot_price"] <= 0
            ):
                raise ValueError(
                    "Phoenix market stats have invalid quantities or prices"
                )
            row = {
                "timestamp": timestamp,
                **quantities,
                "open_interest_value": quantities["open_interest"]
                * quantities["mark_price"],
                "open_interest_unit": "base_asset",
                "single_open_interest": quantities["open_interest"],
                "open_interest_basis": "one_sided_positions",
                "mark_spot_basis_bps": (
                    quantities["mark_price"] / quantities["spot_price"] - 1
                )
                * 10000,
                "slot": point["slot"],
                "market_history_interval": MARKET_HISTORY_INTERVAL,
                "market_history_source": "rest",
                "market_history_timestamp_convention": "interval_start",
            }
            for side in ("maker", "taker"):
                raw = point.get("total_" + side + "_fees")
                if raw is not None:
                    value = Decimal(str(raw))
                    if not value.is_finite():
                        raise ValueError("Phoenix returned invalid cumulative fees")
                    row["cumulative_" + side + "_fees"] = value
            if solana_available():
                row.update(
                    position_history.enrich(row)
                    if position_history is not None
                    else phoenix_positioning(symbol, timestamp, timestamp + interval)
                )
            rows.append(row)
    frame = (
        DataFrame(rows)
        if rows
        else DataFrame(columns=["timestamp", *MARKET_HISTORY_COLUMNS])
    ).set_index("timestamp")
    if solana_available() and not rows and end > start:
        frame.attrs[HISTORY_EXHAUSTED_ATTR] = not has_trades(
            symbol, datetime.fromtimestamp(0, UTC), end
        )
    return frame
