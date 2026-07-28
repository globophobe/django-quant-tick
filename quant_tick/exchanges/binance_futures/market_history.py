from datetime import date, datetime, timedelta
from decimal import Decimal

import pandas as pd
from pandas import DataFrame

from quant_tick.exchanges.binance.api import get_binance_api_response
from quant_tick.lib import filter_by_timestamp, get_current_time, zip_downloader
from .constants import (
    DATA_API_URL,
    METRICS_S3_URL,
    MARKET_HISTORY_INTERVAL,
    MARKET_HISTORY_MAX_RESULTS,
    MARKET_HISTORY_REST_RETENTION,
)

ARCHIVE_COLUMNS = [
    "create_time",
    "symbol",
    "sum_open_interest",
    "sum_open_interest_value",
    "count_toptrader_long_short_ratio",
    "sum_toptrader_long_short_ratio",
    "count_long_short_ratio",
    "sum_taker_long_short_vol_ratio",
]
MARKET_HISTORY_COLUMNS = [
    "open_interest",
    "open_interest_value",
    "top_trader_long_short_account_ratio",
    "top_trader_long_short_position_ratio",
    "long_short_account_ratio",
    "taker_long_short_volume_ratio",
    "open_interest_unit",
    "market_history_interval",
    "market_history_source",
]
REST_SERIES = {
    "openInterestHist": {
        "sumOpenInterest": "open_interest",
        "sumOpenInterestValue": "open_interest_value",
    },
    "topLongShortAccountRatio": {
        "longShortRatio": "top_trader_long_short_account_ratio",
    },
    "topLongShortPositionRatio": {
        "longShortRatio": "top_trader_long_short_position_ratio",
    },
    "globalLongShortAccountRatio": {
        "longShortRatio": "long_short_account_ratio",
    },
    "takerlongshortRatio": {
        "buySellRatio": "taker_long_short_volume_ratio",
    },
}


def empty_market_history() -> DataFrame:
    return DataFrame(columns=["timestamp", *MARKET_HISTORY_COLUMNS]).set_index(
        "timestamp"
    )


def get_binance_metrics_archive_url(api_symbol: str, archive_date: date) -> str:
    symbol = str(api_symbol).strip().upper()
    date_str = archive_date.isoformat()
    return f"{METRICS_S3_URL}/{symbol}/{symbol}-metrics-{date_str}.zip"


def get_binance_metrics_archive(
    api_symbol: str,
    archive_date: date,
) -> DataFrame | None:
    df = zip_downloader(
        get_binance_metrics_archive_url(api_symbol, archive_date),
        ARCHIVE_COLUMNS,
    )
    if df is None:
        return None

    df = df[df["create_time"].str.lower() != "create_time"].copy()
    if df.empty:
        return empty_market_history()

    df["timestamp"] = pd.to_datetime(df["create_time"], utc=True)
    rename = {
        "sum_open_interest": "open_interest",
        "sum_open_interest_value": "open_interest_value",
        "count_toptrader_long_short_ratio": ("top_trader_long_short_account_ratio"),
        "sum_toptrader_long_short_ratio": ("top_trader_long_short_position_ratio"),
        "count_long_short_ratio": "long_short_account_ratio",
        "sum_taker_long_short_vol_ratio": "taker_long_short_volume_ratio",
    }
    df = df.rename(columns=rename)
    for column in rename.values():
        df[column] = [Decimal(str(value)) for value in df[column]]

    # Binance's early metrics archives contain exact duplicate rows.
    df = df.drop_duplicates("timestamp", keep="last")
    df["open_interest_unit"] = "base_asset"
    df["market_history_interval"] = MARKET_HISTORY_INTERVAL
    df["market_history_source"] = "data_vision"
    return df.set_index("timestamp")[MARKET_HISTORY_COLUMNS]


def binance_market_history_archives(
    api_symbol: str,
    timestamp_from: datetime,
    timestamp_to: datetime,
) -> DataFrame:
    if timestamp_to <= timestamp_from:
        return empty_market_history()

    archive_date = (pd.Timestamp(timestamp_to) - pd.Timedelta("1ns")).date()
    first_date = pd.Timestamp(timestamp_from).date()
    frames = []
    while archive_date >= first_date:
        frame = get_binance_metrics_archive(api_symbol, archive_date)
        if frame is None:
            break
        frames.append(frame)
        archive_date -= timedelta(days=1)

    if not frames:
        return empty_market_history()
    df = pd.concat(frames)
    df = filter_by_timestamp(df.reset_index(), timestamp_from, timestamp_to)
    return df.set_index("timestamp").sort_index(kind="stable")


def get_binance_market_history_response(base_url: str) -> list[dict]:
    return get_binance_api_response(
        lambda url, **kwargs: url,
        base_url,
        reverse=False,
    )


def _fetch_rest_series(
    api_symbol: str,
    endpoint: str,
    timestamp_from: datetime,
    timestamp_to: datetime,
) -> list[dict]:
    cursor = timestamp_to
    rows = []
    while cursor > timestamp_from:
        url = (
            f"{DATA_API_URL}/{endpoint}"
            f"?symbol={str(api_symbol).strip().upper()}"
            f"&period={MARKET_HISTORY_INTERVAL}"
            f"&startTime={int(timestamp_from.timestamp() * 1000)}"
            f"&endTime={int(cursor.timestamp() * 1000)}"
            f"&limit={MARKET_HISTORY_MAX_RESULTS}"
        )
        page = get_binance_market_history_response(url)
        if not page:
            break
        rows.extend(page)
        first_timestamp = min(int(item["timestamp"]) for item in page)
        next_cursor = pd.to_datetime(
            first_timestamp - 1,
            unit="ms",
            utc=True,
        ).to_pydatetime()
        if next_cursor >= cursor:
            raise ValueError(f"Binance {endpoint} pagination did not move backward")
        cursor = next_cursor
        if len(page) < MARKET_HISTORY_MAX_RESULTS:
            break
    return rows


def _rest_series_frame(rows: list[dict], fields: dict[str, str]) -> DataFrame:
    columns = ["timestamp", *fields.values()]
    if not rows:
        return DataFrame(columns=columns).set_index("timestamp")
    return DataFrame(
        {
            "timestamp": pd.to_datetime(
                [int(item["timestamp"]) for item in rows],
                unit="ms",
                utc=True,
            ),
            **{
                target: [Decimal(str(item[source])) for item in rows]
                for source, target in fields.items()
            },
        }
    ).set_index("timestamp")


def binance_market_history_rest(
    api_symbol: str,
    timestamp_from: datetime,
    timestamp_to: datetime,
) -> DataFrame:
    if timestamp_to <= timestamp_from:
        return empty_market_history()

    df = None
    for endpoint, fields in REST_SERIES.items():
        frame = _rest_series_frame(
            _fetch_rest_series(api_symbol, endpoint, timestamp_from, timestamp_to),
            fields,
        )
        df = frame if df is None else df.join(frame, how="outer")

    if df is None or df.empty:
        return empty_market_history()
    df["open_interest_unit"] = "base_asset"
    df["market_history_interval"] = MARKET_HISTORY_INTERVAL
    df["market_history_source"] = "rest"
    df = filter_by_timestamp(df.reset_index(), timestamp_from, timestamp_to)
    return df.set_index("timestamp").sort_index(kind="stable")


def binance_market_history(
    api_symbol: str,
    timestamp_from: datetime,
    timestamp_to: datetime,
) -> DataFrame:
    """Fetch Binance Futures market history."""
    if timestamp_to <= timestamp_from:
        return empty_market_history()

    archive_date = (pd.Timestamp(timestamp_to) - pd.Timedelta("1ns")).date()
    first_date = pd.Timestamp(timestamp_from).date()
    rest_cutoff = get_current_time() - MARKET_HISTORY_REST_RETENTION
    frames = []
    while archive_date >= first_date:
        frame = get_binance_metrics_archive(api_symbol, archive_date)
        if frame is None:
            day_from = datetime.combine(
                archive_date,
                datetime.min.time(),
                tzinfo=timestamp_from.tzinfo,
            )
            day_to = day_from + timedelta(days=1)
            rest_from = max(timestamp_from, day_from, rest_cutoff)
            rest_to = min(timestamp_to, day_to)
            if rest_from >= rest_to:
                break
            frame = binance_market_history_rest(
                api_symbol,
                rest_from,
                rest_to,
            )
        if not frame.empty:
            frames.append(frame)
        archive_date -= timedelta(days=1)

    if not frames:
        return empty_market_history()
    df = pd.concat(frames)
    df = filter_by_timestamp(df.reset_index(), timestamp_from, timestamp_to)
    df = df.set_index("timestamp")
    return df[~df.index.duplicated(keep="last")].sort_index(kind="stable")
