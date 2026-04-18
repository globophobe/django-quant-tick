from datetime import UTC, datetime, timedelta
from decimal import Decimal
from functools import partial

from pandas import DataFrame

from quant_tick.controllers import iter_api
from quant_tick.lib import (
    candles_to_data_frame,
    get_interval_inclusive_end,
    parse_fixed_resolution_minutes,
    resample_candles,
)

from .api import get_coinbase_api_response
from .constants import API_URL, CANDLE_MAX_RESULTS, MIN_ELAPSED_PER_REQUEST

COINBASE_GRANULARITY_BY_MINUTES = {
    1: 60,
    5: 300,
    15: 900,
    60: 3600,
    360: 21600,
    1440: 86400,
}


def get_coinbase_candle_url(
    url: str, timestamp_from: datetime, pagination_id: str | None
) -> str:
    start = timestamp_from.replace(tzinfo=None).isoformat()
    url += f"&start={start}"
    if pagination_id:
        url += f"&end={pagination_id}"
    return url


def get_coinbase_candle_timestamp(candle: list) -> datetime:
    return datetime.fromtimestamp(candle[0]).replace(tzinfo=UTC)


def get_coinbase_candle_pagination_id(
    timestamp: datetime,
    last_data: list | None = None,
    data: list | None = None,
) -> None:
    """Coinbase candle windows are paged explicitly by the caller."""


def get_coinbase_fetch_granularity(
    resolution: str | int | None,
) -> tuple[int, int]:
    """Get the largest Coinbase granularity that can be resampled."""
    target_minutes = parse_fixed_resolution_minutes(resolution)
    if target_minutes in COINBASE_GRANULARITY_BY_MINUTES:
        return target_minutes, COINBASE_GRANULARITY_BY_MINUTES[target_minutes]
    for minutes in sorted(COINBASE_GRANULARITY_BY_MINUTES, reverse=True):
        if target_minutes % minutes == 0:
            return target_minutes, COINBASE_GRANULARITY_BY_MINUTES[minutes]
    return target_minutes, COINBASE_GRANULARITY_BY_MINUTES[1]


def fetch_coinbase_candles(
    api_symbol: str,
    timestamp_from: datetime,
    timestamp_to: datetime,
    *,
    granularity: int,
    log_format: str | None = None,
) -> DataFrame:
    """Fetch Coinbase candles."""
    url = f"{API_URL}/products/{api_symbol}/candles?granularity={granularity}"
    window = timedelta(seconds=granularity * CANDLE_MAX_RESULTS)
    candles_by_timestamp = {}
    current = timestamp_from
    while current < timestamp_to:
        window_to = min(current + window, timestamp_to)
        ts_to = get_interval_inclusive_end(current, window_to, f"{granularity}s")
        pagination_id = ts_to.replace(tzinfo=None).isoformat()
        candles, _, _ = iter_api(
            url,
            get_coinbase_candle_pagination_id,
            get_coinbase_candle_timestamp,
            partial(get_coinbase_api_response, get_coinbase_candle_url),
            CANDLE_MAX_RESULTS,
            MIN_ELAPSED_PER_REQUEST,
            timestamp_from=current,
            pagination_id=pagination_id,
            log_format=log_format,
        )
        for candle in candles:
            candle_timestamp = get_coinbase_candle_timestamp(candle)
            if current <= candle_timestamp < window_to:
                candles_by_timestamp[candle_timestamp] = {
                    "timestamp": candle_timestamp,
                    "open": Decimal(str(candle[3])),
                    "high": Decimal(str(candle[2])),
                    "low": Decimal(str(candle[1])),
                    "close": Decimal(str(candle[4])),
                    "notional": Decimal(str(candle[5])),
                }
        current = window_to
    parsed = sorted(candles_by_timestamp.values(), key=lambda candle: candle["timestamp"])
    return candles_to_data_frame(
        timestamp_from,
        timestamp_to,
        parsed,
        reverse=False,
    )


def coinbase_candles(
    api_symbol: str,
    timestamp_from: datetime,
    timestamp_to: datetime,
    granularity: int = 60,
    resolution: str | int | None = None,
    log_format: str | None = None,
) -> DataFrame:
    if resolution is None:
        target_minutes = granularity // 60
        fetch_granularity = granularity
    else:
        target_minutes, fetch_granularity = get_coinbase_fetch_granularity(resolution)
    data_frame = fetch_coinbase_candles(
        api_symbol,
        timestamp_from,
        timestamp_to,
        granularity=fetch_granularity,
        log_format=log_format,
    )
    if (fetch_granularity // 60) == target_minutes:
        return data_frame
    return resample_candles(
        data_frame,
        timestamp_from=timestamp_from,
        timestamp_to=timestamp_to,
        resolution_minutes=target_minutes,
    )
