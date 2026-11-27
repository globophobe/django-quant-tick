"""Sampled marks from node state, kept at their actual observation time.

The snapshot's integer price has 6 - szDecimals decimal places. The table
does not contain historical oracle or mid prices; daily_px is not an oracle.
Only snapshots within each completed hour contribute, without carrying prices
across missing hours or treating sampled extrema as continuous mark OHLC.
"""

from decimal import Decimal
from functools import lru_cache

from quant_tick.lib import get_current_time

from .api import coin_literal, number, query, time_literal, timestamp


@lru_cache(maxsize=128)
def mark_history_start(coin, day):
    rows = query(
        "SELECT snapshot_time FROM hyperliquid_oracle_prices "
        f"WHERE coin={coin_literal(coin)} ORDER BY block_number LIMIT 1"
    )
    return timestamp(rows[0]["snapshot_time"]).floor("h") if rows else None


@lru_cache(maxsize=128)
def price_scale(coin):
    rows = query(
        "SELECT coin,sz_decimals FROM hyperliquid_perpetual_markets FINAL "
        f"WHERE coin={coin_literal(coin)} LIMIT 2"
    )
    if len(rows) != 1 or rows[0]["coin"] != coin:
        raise ValueError("QuickNode omitted the perpetual price precision")
    decimals = number(rows[0]["sz_decimals"])
    if decimals != int(decimals) or not 0 <= decimals <= 6:
        raise ValueError("QuickNode returned invalid perpetual price precision")
    return Decimal(10) ** (decimals - 6)


def mark_prices(coin, start, end):
    first = mark_history_start(coin, get_current_time().date())
    if first is None or end <= first:
        return {}
    start = max(start, first)
    rows = query(
        "SELECT block_number,snapshot_time,mark_px "
        "FROM hyperliquid_oracle_prices FINAL "
        f"WHERE coin={coin_literal(coin)} "
        f"AND snapshot_time >= {time_literal(start)} "
        f"AND snapshot_time < {time_literal(end)} "
        "ORDER BY block_number LIMIT 1001"
    )
    if len(rows) >= 1001:
        raise ValueError("QuickNode mark snapshot limit reached")
    if not rows:
        return {}
    scale = price_scale(coin)
    result = {}
    previous_block, previous_time = 0, None
    for row in rows:
        moment = timestamp(row["snapshot_time"])
        block = int(row["block_number"])
        raw = number(row["mark_px"], positive=True)
        if (
            not start <= moment < end
            or block <= previous_block
            or (previous_time is not None and moment < previous_time)
            or raw != raw.to_integral_value()
        ):
            raise ValueError("QuickNode returned invalid mark snapshots")
        previous_block, previous_time = block, moment
        result[moment.floor("h")] = {
            "mark_price": raw * scale,
            "oracle_price": None,
            "mid_price": None,
            "price_timestamp": moment,
            "price_block": block,
        }
    return result
