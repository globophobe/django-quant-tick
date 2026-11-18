"""Bounded public-market SQL requests.

SQL requests follow the response quota headers and honor Retry-After. Scan limits fail
closed; they never truncate a successful result or silently switch providers.
"""

import os
import re
from decimal import Decimal
from functools import lru_cache

import pandas as pd
from django.core.exceptions import ImproperlyConfigured

from quant_tick.lib import get_current_time

from ..public import get_public_json

QUERY_SETTINGS = (
    " SETTINGS max_execution_time=10, max_bytes_to_read={scan_limit}, "
    "max_rows_to_read=60000000, read_overflow_mode='throw', "
    "result_overflow_mode='throw', timeout_overflow_mode='throw'"
)


def available():
    return bool(os.environ.get("QUICKNODE_API_KEY", "").strip())


def coin_literal(coin):
    if not re.fullmatch(r"(?:@[0-9]+|[A-Za-z0-9._+-]+(?::[A-Za-z0-9._+-]+)?)", coin):
        raise ValueError("QuickNode requires a native Hyperliquid coin or @spot ID")
    return "'" + coin + "'"


def timestamp(value):
    result = pd.Timestamp(value)
    return (
        result.tz_localize("UTC") if result.tzinfo is None else result.tz_convert("UTC")
    )


def time_literal(value):
    return "'" + timestamp(value).strftime("%Y-%m-%d %H:%M:%S.%f") + "'"


def number(value, *, positive=False):
    result = Decimal(str(value))
    if not result.is_finite() or (positive and result <= 0):
        raise ValueError("QuickNode returned an invalid numeric value")
    return result


def query(sql, *, scan_limit=536870912):
    key = os.environ.get("QUICKNODE_API_KEY", "").strip()
    if not key:
        raise ImproperlyConfigured("Set QUICKNODE_API_KEY to collect QuickNode history")
    payload = get_public_json(
        "https://api.quicknode.com",
        "/sql/rest/v1/query",
        {},
        namespace="QUICKNODE",
        max_requests=None,
        headers={"x-api-key": key},
        json_body={
            "clusterId": "hyperliquid-core-mainnet",
            "query": sql + QUERY_SETTINGS.format(scan_limit=scan_limit),
        },
    )
    if (
        not isinstance(payload, dict)
        or "error" in payload
        or not isinstance(payload.get("data"), list)
        or payload.get("rows") != len(payload["data"])
    ):
        raise ValueError("QuickNode returned an invalid or incomplete SQL response")
    return payload["data"]


def history_start():
    return _history_start(get_current_time().date())


@lru_cache(maxsize=2)
def _history_start(day):
    # Read the raw table: summary and snapshot tables have different coverage.
    rows = query(
        "SELECT block_time FROM hyperliquid_trades "
        "ORDER BY block_number, trade_id LIMIT 1"
    )
    return timestamp(rows[0]["block_time"]) if rows else None


def block_range(start, end):
    lower = timestamp(start) - pd.Timedelta(minutes=1)
    stop = timestamp(end) + pd.Timedelta(minutes=1)
    rows = []
    # The time filter prunes monthly partitions but can still scan every block
    # in each month. Keep month crossings within the same per-request scan limits.
    while lower < stop:
        upper = min(stop, lower.normalize() + pd.offsets.MonthBegin(1))
        from_literal, to_literal = time_literal(lower), time_literal(upper)
        rows.extend(
            query(
                "SELECT min(block_number) AS first, max(block_number) AS last, "
                "count() AS blocks, min(block_time) AS start, max(block_time) AS end "
                "FROM hyperliquid_blocks "
                f"WHERE block_time >= {from_literal} "
                f"AND block_time < {to_literal} LIMIT 1"
            )
        )
        lower = upper
    rows = [row for row in rows if row["blocks"]]
    if not rows:
        return None
    # The index is only a scan hint; lag or missing boundary coverage must not
    # clip trade queries to a partial window.
    if min(timestamp(row["start"]) for row in rows) > timestamp(start) or max(
        timestamp(row["end"]) for row in rows
    ) < timestamp(end):
        return None
    bounds = [(int(row["first"]), int(row["last"])) for row in rows]
    if any(not 0 < first <= last for first, last in bounds):
        raise ValueError("QuickNode returned invalid block bounds")
    return min(first for first, _ in bounds), max(last for _, last in bounds)


def trade_where(coin, start, end, bounds):
    market, lower, upper = coin_literal(coin), time_literal(start), time_literal(end)
    block_from = time_literal(timestamp(start) - pd.Timedelta(minutes=1))
    block_to = time_literal(timestamp(end) + pd.Timedelta(minutes=1))
    block_filter = ""
    if bounds is not None:
        first, last = bounds
        block_filter = f"AND block_number BETWEEN {first} AND {last} "
    return (
        f"coin={market} {block_filter}"
        f"AND block_time >= {block_from} AND block_time < {block_to} "
        f"AND timestamp >= {lower} AND timestamp < {upper}"
    )
