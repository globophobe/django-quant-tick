"""Authenticated HyperCore market history with bounded request windows."""

import os
import re
from datetime import timedelta
from decimal import Decimal

import pandas as pd
from django.core.exceptions import ImproperlyConfigured

from quant_tick.lib import get_current_time, iter_chunks, to_utc_datetime

from ..public import get_public_json

PAGE_SIZE = 720
OI_START = pd.Timestamp("2023-05-20", tz="UTC")
COHORT_START = pd.Timestamp("2025-09-28", tz="UTC")


def normalize_coin(api_symbol):
    coin = api_symbol.strip()
    if not re.fullmatch(r"[A-Za-z0-9._+-]+(?::[A-Za-z0-9._+-]+)?", coin):
        raise ValueError("Femtos requires a HyperCore perp coin, e.g. BTC or xyz:TSLA")
    return coin


def get_femtos_response(path, params):
    key = os.environ.get("FEMTOS_API_KEY", "").strip()
    if not key:
        raise ImproperlyConfigured("Set FEMTOS_API_KEY to collect Femtos market data")
    result = get_public_json(
        "https://api.femtos.xyz/v1",
        path,
        params,
        namespace="FEMTOS",
        headers={"Authorization": "Bearer " + key},
        retry=3,
    )
    if not isinstance(result, dict) or "error" in result:
        raise ValueError("Femtos returned an unexpected market data response")
    return result


def number(value, *, nullable=False, nonnegative=True):
    if value is None and nullable:
        return None
    result = Decimal(str(value))
    if not result.is_finite() or (nonnegative and result < 0):
        raise ValueError("Femtos returned an invalid numeric value")
    return result


def completed_windows(timestamp_from, timestamp_to, minutes):
    """Exclude forming bars and keep requests below the heavy-call boundary."""
    interval = pd.Timedelta(minutes=minutes)
    start = pd.Timestamp(to_utc_datetime(timestamp_from)).ceil(interval)
    end = min(
        pd.Timestamp(to_utc_datetime(timestamp_to)), pd.Timestamp(get_current_time())
    ).floor(interval)
    if end <= start:
        return
    yield from iter_chunks(
        start, end, value=min(timedelta(days=30), interval * PAGE_SIZE)
    )


def ordered_points(points, start, end, minutes, *, snapshots=False):
    previous = None
    interval = pd.Timedelta(minutes=minutes)
    if len(points) > PAGE_SIZE:
        raise ValueError("Femtos response exceeds its bounded window")
    for point in points:
        raw = pd.Timestamp(point["t"], unit="ms", tz="UTC")
        timestamp = raw.floor(interval)
        if (
            not start <= raw < end
            or (not snapshots and raw != timestamp)
            or (previous is not None and timestamp <= previous)
        ):
            raise ValueError("Femtos timestamps are unordered or outside the window")
        previous = timestamp
        yield timestamp, point


def oi_pages(api_symbol, timestamp_from, timestamp_to):
    coin = normalize_coin(api_symbol)
    for start, end in completed_windows(
        max(timestamp_from, OI_START), timestamp_to, 60
    ):
        result = get_femtos_response(
            "/hypercore/oi",
            {
                "coin": coin,
                "tf": "1h",
                "from": int(start.timestamp() * 1000),
                "to": int(end.timestamp() * 1000),
                "limit": PAGE_SIZE,
            },
        )
        if result["coin"] != coin or result["tf"] != "1h":
            raise ValueError("Femtos returned an unexpected coin or interval")
        head = result["asOf"]
        if result["bars"] and head is None:
            raise ValueError("Femtos OI response has no indexed head")
        points = [
            (timestamp, point)
            for timestamp, point in ordered_points(result["bars"], start, end, 60)
            if timestamp + pd.Timedelta(hours=1)
            <= pd.Timestamp(head, unit="ms", tz="UTC")
        ]
        yield start, end, points
