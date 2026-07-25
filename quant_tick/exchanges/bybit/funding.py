from datetime import datetime
from decimal import Decimal

import pandas as pd
from pandas import DataFrame

from quant_tick.constants import SymbolType
from quant_tick.exchanges.funding import ExchangeFunding

from .api import get_bybit_result, to_millis
from .candles import get_bybit_category
from .constants import (
    FUNDING_MAX_RESULTS,
    MARKET_HISTORY_INTERVAL,
    MARKET_HISTORY_MAX_RESULTS,
)


class BybitFunding(ExchangeFunding):
    interval = pd.Timedelta("8h")
    timestamp_anomaly_tolerance = pd.Timedelta("1min")


def _category(api_symbol: str) -> str:
    return get_bybit_category(api_symbol, SymbolType.PERPETUAL)


def get_bybit_funding_response(
    api_symbol: str,
    start_ms: int,
    end_ms: int,
    *,
    limit: int = FUNDING_MAX_RESULTS,
) -> dict:
    return get_bybit_result(
        "/v5/market/funding/history",
        {
            "category": _category(api_symbol),
            "symbol": str(api_symbol).strip().upper(),
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": limit,
        },
    )


def _fetch_funding_rows(
    api_symbol: str,
    timestamp_from: datetime,
    timestamp_to: datetime,
) -> list[dict]:
    start_ms = to_millis(timestamp_from)
    cursor_end_ms = to_millis(timestamp_to)
    rows = []
    while start_ms <= cursor_end_ms:
        result = get_bybit_funding_response(
            api_symbol,
            start_ms,
            cursor_end_ms,
        )
        page = result.get("list", [])
        if not page:
            break
        rows.extend(page)
        oldest_ms = min(int(item["fundingRateTimestamp"]) for item in page)
        if len(page) < FUNDING_MAX_RESULTS or oldest_ms <= start_ms:
            break
        next_end_ms = oldest_ms - 1
        if next_end_ms >= cursor_end_ms:
            raise ValueError("Bybit funding pagination did not move backward")
        cursor_end_ms = next_end_ms
    return rows


def _fetch_cursor_rows(path: str, params: dict) -> list[dict]:
    rows = []
    cursor = ""
    seen_cursors = set()
    while True:
        page_params = dict(params)
        if cursor:
            page_params["cursor"] = cursor
        result = get_bybit_result(path, page_params)
        rows.extend(result.get("list", []))
        next_cursor = result.get("nextPageCursor") or ""
        if not next_cursor:
            break
        if next_cursor in seen_cursors:
            raise ValueError(f"Bybit {path} pagination did not move backward")
        seen_cursors.add(next_cursor)
        cursor = next_cursor
    return rows


def bybit_open_interest(
    api_symbol: str,
    timestamp_from: datetime,
    timestamp_to: datetime,
) -> DataFrame:
    columns = ["timestamp", "open_interest", "single_open_interest"]
    rows = _fetch_cursor_rows(
        "/v5/market/open-interest",
        {
            "category": _category(api_symbol),
            "symbol": str(api_symbol).strip().upper(),
            "intervalTime": MARKET_HISTORY_INTERVAL,
            "startTime": to_millis(timestamp_from),
            "endTime": to_millis(timestamp_to),
            "limit": MARKET_HISTORY_MAX_RESULTS,
        },
    )
    if not rows:
        return DataFrame(columns=columns).set_index("timestamp")
    return DataFrame(
        {
            "timestamp": pd.to_datetime(
                [int(item["timestamp"]) for item in rows],
                unit="ms",
                utc=True,
            ),
            "open_interest": [
                Decimal(str(item["openInterest"])) for item in rows
            ],
            "single_open_interest": [
                Decimal(str(item["singleOpenInterest"])) for item in rows
            ],
        }
    ).set_index("timestamp")


def bybit_account_ratio(
    api_symbol: str,
    timestamp_from: datetime,
    timestamp_to: datetime,
) -> DataFrame:
    columns = [
        "timestamp",
        "long_account_ratio",
        "short_account_ratio",
        "long_short_account_ratio",
    ]
    rows = _fetch_cursor_rows(
        "/v5/market/account-ratio",
        {
            "category": _category(api_symbol),
            "symbol": str(api_symbol).strip().upper(),
            "period": MARKET_HISTORY_INTERVAL,
            "startTime": to_millis(timestamp_from),
            "endTime": to_millis(timestamp_to),
            "limit": MARKET_HISTORY_MAX_RESULTS,
        },
    )
    if not rows:
        return DataFrame(columns=columns).set_index("timestamp")
    long_ratios = [Decimal(str(item["buyRatio"])) for item in rows]
    short_ratios = [Decimal(str(item["sellRatio"])) for item in rows]
    return DataFrame(
        {
            "timestamp": pd.to_datetime(
                [int(item["timestamp"]) for item in rows],
                unit="ms",
                utc=True,
            ),
            "long_account_ratio": long_ratios,
            "short_account_ratio": short_ratios,
            "long_short_account_ratio": [
                long_ratio / short_ratio if short_ratio else None
                for long_ratio, short_ratio in zip(
                    long_ratios,
                    short_ratios,
                    strict=True,
                )
            ],
        }
    ).set_index("timestamp")


def bybit_funding(
    api_symbol: str,
    timestamp_from: datetime,
    timestamp_to: datetime,
    *,
    funding_interval: str | pd.Timedelta | None = None,
) -> DataFrame:
    """Fetch Bybit funding with aligned OI and account positioning."""
    columns = [
        "funding_rate",
        "open_interest",
        "single_open_interest",
        "open_interest_unit",
        "long_account_ratio",
        "short_account_ratio",
        "long_short_account_ratio",
        "market_history_interval",
    ]
    if timestamp_to <= timestamp_from:
        return BybitFunding.empty_frame(columns)

    rows = _fetch_funding_rows(api_symbol, timestamp_from, timestamp_to)
    if not rows:
        return BybitFunding.empty_frame(columns)

    df = DataFrame(
        {
            "timestamp": pd.to_datetime(
                [int(item["fundingRateTimestamp"]) for item in rows],
                unit="ms",
                utc=True,
            ),
            "funding_rate": [
                Decimal(str(item["fundingRate"])) for item in rows
            ],
        }
    ).set_index("timestamp")
    df = df.join(
        bybit_open_interest(api_symbol, timestamp_from, timestamp_to),
        how="left",
    ).join(
        bybit_account_ratio(api_symbol, timestamp_from, timestamp_to),
        how="left",
    )
    category = _category(api_symbol)
    df["open_interest_unit"] = (
        "base_asset" if category == "linear" else "quote_asset"
    )
    df["market_history_interval"] = MARKET_HISTORY_INTERVAL
    normalized = BybitFunding.normalize_frame(
        df.reset_index(),
        timestamp_from,
        timestamp_to,
        interval=funding_interval,
    )
    return normalized.sort_index(kind="stable")
