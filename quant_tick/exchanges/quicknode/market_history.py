"""Hourly positioning anchored independently by funding settlement records.

Rows describe completed intervals. Cohorts are positions just before the funding
block near hour start; OI and OI counts describe the next settlement boundary.
Trades between those blocks must reconcile both position sizes and account counts.
The rate is the funding settled at the interval's end. Prices use actual execution
order; trade IDs do not define execution order.

Funding coverage, not trade or node-snapshot coverage, bounds this collector.
Old stored history remains usable. Missing anchors, prices or flows leave hours retryable;
provider errors and reconciliation failures propagate. Intrahour OI high/low is
unavailable here, and is explicitly null rather than inferred from block sums.
"""

from functools import lru_cache
from itertools import pairwise

import pandas as pd

from quant_tick.lib import get_current_time, iter_chunks

from ..binance_futures.market_history import HISTORY_EXHAUSTED_ATTR
from ..hyperliquid.api import normalize_coin
from .api import block_range, query, timestamp
from .flows import (
    COUNT_FIELDS,
    FLOW_FIELDS,
    SIGNED_FLOW_FIELDS,
    position_flows,
    validate_flow,
)
from .positions import cohort_tiers, cohort_values, funding_points, preceding_prices

BATCH = pd.Timedelta(hours=6)


@lru_cache(maxsize=2)
def funding_history_start(day):
    rows = query(
        "SELECT time FROM hyperliquid_funding ORDER BY block_number,user LIMIT 1"
    )
    return timestamp(rows[0]["time"]).floor("h") if rows else None


def market_window(coin, start, end):
    query_end = end + pd.Timedelta(minutes=1)
    price_start = start - pd.Timedelta(hours=1)
    bounds = block_range(price_start, query_end)
    points = funding_points(coin, start, query_end, bounds)
    if len(points) < 2:
        return []
    prices = preceding_prices(coin, points, price_start, query_end, bounds)
    tiers = cohort_tiers(coin, points, prices, start, query_end)
    flows = position_flows(coin, points, start, query_end)
    rows = []
    for index, ((hour, before), (following, after)) in enumerate(
        pairwise(points.items())
    ):
        if (
            following - hour != pd.Timedelta(hours=1)
            or index not in flows
            or before["block_number"] not in prices
            or after["block_number"] not in prices
        ):
            continue
        flow = flows[index]
        validate_flow(before, after, flow)
        price = prices[after["block_number"]]
        oi = after["long_size"] * 2
        rows.append(
            {
                "timestamp": hour,
                "open_interest": oi,
                "single_open_interest": oi / 2,
                "open_interest_value": oi * price,
                "open_interest_unit": "base_asset",
                "open_interest_open": before["long_size"] * 2,
                "open_interest_high": None,
                "open_interest_low": None,
                "last_price": price,
                "oi_long_accounts": after["long_accounts"],
                "oi_short_accounts": after["short_accounts"],
                "oi_timestamp": after["time"],
                "oi_block": after["block_number"],
                "funding_rate": after["rate"],
                "trades": int(flow["trades"]),
                **{
                    key: flow[key]
                    for key in (*FLOW_FIELDS, *SIGNED_FLOW_FIELDS, *COUNT_FIELDS)
                },
                "long_account_delta": int(flow["long_account_delta"]),
                "short_account_delta": int(flow["short_account_delta"]),
                "fee_currency": flow["fee_currency"],
                "taker_long_short_volume_ratio": flow["taker_buy_size"]
                / flow["taker_sell_size"]
                if flow["taker_sell_size"]
                else None,
                **cohort_values(
                    before,
                    prices[before["block_number"]],
                    tiers.get(before["block_number"], []),
                ),
                "market_history_interval": "1h",
                "market_history_source": "quicknode",
                "market_history_timestamp_convention": "interval_start",
                "open_interest_basis": "sum_absolute_positions",
                "position_source": "funding_settlement",
                "position_flow_validated": True,
            }
        )
    return rows


def quicknode_market_history(api_symbol, timestamp_from, timestamp_to):
    coin = normalize_coin(api_symbol)
    if coin.startswith("@"):
        raise ValueError("Positioning history requires a perpetual market")
    start = timestamp(timestamp_from).ceil("h")
    end = min(timestamp(timestamp_to), timestamp(get_current_time())).floor("h")
    empty = pd.DataFrame(columns=["timestamp", "open_interest"]).set_index("timestamp")
    if end <= start:
        return empty
    first = funding_history_start(timestamp(get_current_time()).date())
    if first is not None:
        empty.attrs[HISTORY_EXHAUSTED_ATTR] = end <= first
    if first is None or end <= first:
        return empty
    start = max(start, first)
    rows = []
    for lower, upper in iter_chunks(start, end, value=BATCH):
        rows.extend(market_window(coin, lower, upper))
    return pd.DataFrame(rows).set_index("timestamp") if rows else empty
