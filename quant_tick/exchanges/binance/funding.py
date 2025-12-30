import logging
import time
from datetime import datetime
from decimal import Decimal

import httpx
from decouple import config

from quant_tick.controllers import HTTPX_ERRORS
from quant_tick.lib import parse_datetime
from quant_tick.models import FundingRate, Symbol

from .constants import BINANCE_API_KEY, FUTURES_API_URL

logger = logging.getLogger(__name__)


def get_funding_rates(
    api_symbol: str,
    start_time: datetime | None = None,
    end_time: datetime | None = None,
    limit: int = 1000,
    retry: int = 30,
) -> list[dict]:
    """Fetch funding rate history from Binance Futures API.

    API: GET /fapi/v1/fundingRate
    Rate limit: 500/5min/IP shared with /fapi/v1/fundingInfo

    Returns list of:
        {symbol, fundingRate, fundingTime, markPrice}
    """
    url = f"{FUTURES_API_URL}/fapi/v1/fundingRate?symbol={api_symbol}&limit={limit}"
    if start_time:
        url += f"&startTime={int(start_time.timestamp() * 1000)}"
    if end_time:
        url += f"&endTime={int(end_time.timestamp() * 1000)}"

    try:
        headers = {"X-MBX-APIKEY": config(BINANCE_API_KEY)}
        response = httpx.get(url, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            response.raise_for_status()
    except HTTPX_ERRORS:
        if retry > 0:
            time.sleep(1)
            return get_funding_rates(api_symbol, start_time, end_time, limit, retry - 1)
        raise


def collect_funding_rates(
    symbol: Symbol,
    timestamp_from: datetime | None = None,
    timestamp_to: datetime | None = None,
) -> dict:
    """Collect funding rates for a symbol.

    Args:
        symbol: Symbol instance
        timestamp_from: Start time (inclusive)
        timestamp_to: End time (inclusive)

    Returns:
        dict with keys: created, skipped
    """
    created = 0
    skipped = 0

    # If no date range, get latest (last 200)
    if timestamp_from is None and timestamp_to is None:
        data = get_funding_rates(symbol.api_symbol)
    else:
        # Paginate through date range
        data = []
        current_start = timestamp_from
        while True:
            batch = get_funding_rates(
                symbol.api_symbol,
                start_time=current_start,
                end_time=timestamp_to,
                limit=1000,
            )
            if not batch:
                break
            data.extend(batch)
            # API returns in ascending order, get last timestamp for next page
            last_ts = parse_datetime(batch[-1]["fundingTime"], unit="ms")
            if len(batch) < 1000 or (timestamp_to and last_ts >= timestamp_to):
                break
            current_start = last_ts

    for item in data:
        funding_time = parse_datetime(item["fundingTime"], unit="ms")
        rate = Decimal(item["fundingRate"])

        _, was_created = FundingRate.objects.get_or_create(
            symbol=symbol,
            timestamp=funding_time,
            defaults={"rate": rate},
        )
        if was_created:
            created += 1
        else:
            skipped += 1

    logger.info(f"{symbol}: Created {created} FundingRate, skipped {skipped}")
    return {"created": created, "skipped": skipped}
