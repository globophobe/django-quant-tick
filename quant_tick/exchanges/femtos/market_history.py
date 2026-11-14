"""Hourly OI/flows and funding-block position cohorts with distinct timestamps.

Rows are available after their hour closes. OI and its account counts describe
the hour end; cohort counts/ratios and sizes describe the funding block near
the hour start, retaining its exact timestamp. OI counts both sides together.
The $1M+ tiers and top-ten positions are not Binance's top-trader population.

Before cohort history begins, OI-only observations are complete. Afterwards,
a missing cohort leaves that hour absent and retryable. Missing flow evidence
stays null; it is never inferred from changes in OI or treated as zero.
"""

from decimal import Decimal

import pandas as pd
from pandas import DataFrame

from .api import (
    COHORT_START,
    PAGE_SIZE,
    get_femtos_response,
    normalize_coin,
    number,
    oi_pages,
    ordered_points,
)

FLOW_FIELDS = {
    "longOpen": "long_open_size",
    "longClose": "long_close_size",
    "shortOpen": "short_open_size",
    "shortClose": "short_close_size",
    "takerLongOpen": "taker_long_open_size",
    "takerShortOpen": "taker_short_open_size",
    "liq": "liquidated_size",
}


def cohort_values(point):
    cohorts = point["cohorts"]
    if len({c["tier"] for c in cohorts}) != len(cohorts):
        raise ValueError("Femtos cohort tiers are duplicated")
    sizes, counts = {}, {}
    for side in ("long", "short"):
        sizes[side] = sum(number(c[side + "Sz"]) for c in cohorts)
        counts[side] = number(point[side + "Accounts"])
        if sum(number(c[side + "Accounts"]) for c in cohorts) != counts[side]:
            raise ValueError("Femtos cohort account counts do not reconcile")
    if sizes["long"] != sizes["short"] or sum(sizes.values()) != number(point["oi"]):
        raise ValueError("Femtos cohort quantities do not reconcile")
    total = sum(counts.values())
    return {
        "long_accounts": counts["long"],
        "short_accounts": counts["short"],
        "long_account_ratio": counts["long"] / total if total else None,
        "short_account_ratio": counts["short"] / total if total else None,
        "long_short_account_ratio": (
            counts["long"] / counts["short"] if counts["short"] else None
        ),
        "cohort_timestamp": pd.Timestamp(point["t"], unit="ms", tz="UTC"),
        "cohort_block": point["block"],
        "cohort_price": number(point["px"], nullable=True),
        "cohort_open_interest": number(point["oi"]),
        "cohort_top10_long_size": number(point["top10Long"]),
        "cohort_top10_short_size": number(point["top10Short"]),
        "cohorts": cohorts,
        **{
            "large_" + side + "_size": sum(
                (number(c[side + "Sz"]) for c in cohorts if c["tier"] >= 4),
                Decimal(0),
            )
            for side in ("long", "short")
        },
    }


def femtos_market_history(api_symbol, timestamp_from, timestamp_to):
    coin = normalize_coin(api_symbol)
    rows = []
    for start, end, points in oi_pages(coin, timestamp_from, timestamp_to):
        cohorts = {}
        if points and end > COHORT_START:
            cohort_start = max(start, COHORT_START)
            result = get_femtos_response(
                "/hypercore/oi/cohorts",
                {
                    "coin": coin,
                    "every": "1h",
                    "from": int(cohort_start.timestamp() * 1000),
                    "to": int(end.timestamp() * 1000),
                    "limit": PAGE_SIZE,
                },
            )
            if result["coin"] != coin:
                raise ValueError("Femtos returned unexpected cohort coin")
            cohorts = {
                timestamp: cohort_values(point)
                for timestamp, point in ordered_points(
                    result["points"], cohort_start, end, 60, snapshots=True
                )
            }
        for timestamp, point in points:
            if timestamp >= COHORT_START and timestamp not in cohorts:
                continue
            o, h, low, close = (number(point[key]) for key in ("o", "h", "l", "c"))
            if not low <= min(o, close) <= max(o, close) <= h:
                raise ValueError("Femtos OI has invalid OHLC bounds")
            rows.append(
                {
                    "timestamp": timestamp,
                    "open_interest": close,
                    "single_open_interest": close / 2,
                    "open_interest_value": number(point["usd"], nullable=True),
                    "open_interest_unit": "base_asset",
                    "open_interest_open": o,
                    "open_interest_high": h,
                    "open_interest_low": low,
                    "last_price": number(point["px"], nullable=True),
                    "oi_long_accounts": number(point["longAccounts"], nullable=True),
                    "oi_short_accounts": number(point["shortAccounts"], nullable=True),
                    "trades": number(point["trades"], nullable=True),
                    "funding_rate": number(
                        point["funding"], nullable=True, nonnegative=False
                    ),
                    **{
                        field: number(point[key], nullable=True)
                        for key, field in FLOW_FIELDS.items()
                    },
                    **cohorts.get(timestamp, {}),
                    "market_history_interval": "1h",
                    "market_history_source": "femtos",
                    "market_history_timestamp_convention": "interval_start",
                    "open_interest_basis": "sum_absolute_positions",
                }
            )
    if not rows:
        return DataFrame(columns=["timestamp", "open_interest"]).set_index("timestamp")
    return DataFrame(rows).set_index("timestamp")
