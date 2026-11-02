"""Settled Aster funding, including the predecessor APX/Astherus history."""

from datetime import datetime, timedelta
from decimal import Decimal

import pandas as pd
from pandas import DataFrame

from quant_tick.exchanges.funding import ExchangeFunding

from .api import get_aster_response

FUNDING_MAX_RESULTS = 1000


class AsterFunding(ExchangeFunding):
    interval = pd.Timedelta("8h")
    timestamp_anomaly_tolerance = pd.Timedelta("1min")


def get_aster_funding_interval(api_symbol: str) -> timedelta:
    symbol = api_symbol.strip().upper()
    rows = get_aster_response("/fundingInfo", {"symbol": symbol})
    if len(rows) != 1 or rows[0]["symbol"] != symbol:
        raise ValueError("Aster funding config returned an unexpected market")
    hours = Decimal(str(rows[0]["fundingIntervalHours"]))
    if not hours.is_finite() or hours <= 0 or hours != int(hours):
        raise ValueError("Aster funding interval must be positive whole hours")
    return timedelta(hours=int(hours))


def aster_funding(
    api_symbol: str,
    timestamp_from: datetime,
    timestamp_to: datetime,
    *,
    funding_interval: timedelta | None = None,
) -> DataFrame:
    """Page oldest-first; retain millisecond settlement offsets as metadata."""
    symbol = api_symbol.strip().upper()
    cursor = int(timestamp_from.timestamp() * 1000)
    end = int(timestamp_to.timestamp() * 1000)
    rows = []
    while cursor < end:
        page = get_aster_response(
            "/fundingRate",
            {
                "symbol": symbol,
                "startTime": cursor,
                "endTime": end - 1,
                "limit": FUNDING_MAX_RESULTS,
            },
        )
        if not page:
            break
        for point in page:
            timestamp = int(point["fundingTime"])
            if point["symbol"] != symbol or not cursor <= timestamp < end:
                raise ValueError("Aster funding has an unexpected market or timestamp")
            rate = Decimal(point["fundingRate"])
            if not rate.is_finite():
                raise ValueError("Aster funding rate is not finite")
            rows.append(
                {
                    "timestamp": pd.to_datetime(timestamp, unit="ms", utc=True),
                    "funding_rate": rate,
                }
            )
            cursor = timestamp + 1
        if len(page) < FUNDING_MAX_RESULTS:
            break
    if not rows:
        return AsterFunding.empty_frame(["funding_rate"])
    frame = AsterFunding.normalize_frame(
        DataFrame(rows), timestamp_from, timestamp_to, interval=funding_interval
    )
    if frame.timestamp_anomaly.eq(True).any() or not frame.index.is_unique:
        raise ValueError("Aster funding history does not match its funding interval")
    return frame
