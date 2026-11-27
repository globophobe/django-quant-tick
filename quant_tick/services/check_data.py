"""Compare daily quantities without downloading trade files or rebuilding candles.

TradeData summaries are compared with fresh venue candles using collection's
quantity tolerance. Stored CandleData and opening/closing cache remainders are
then reconciled with those summaries. Cache boundaries must match the requested
interval; pending candle timestamps delimit the completed candle rows.
"""

from collections import Counter
from datetime import timedelta
from decimal import Decimal, InvalidOperation

import pandas as pd
from django.db.models import Count, Sum

from quant_tick.constants import Exchange
from quant_tick.exchanges.api import candles_api
from quant_tick.exchanges.hyperliquid.candles import native_hyperliquid_candles
from quant_tick.lib import resample_candles
from quant_tick.lib.dataframe import is_decimal_close
from quant_tick.models import CandleCache, CandleData, TradeData

QUANTITIES = ("volume", "notional")


class AuditUnavailable(ValueError):
    pass


def quantities(data):
    try:
        values = {key: Decimal(str(data[key])) for key in QUANTITIES}
    except (KeyError, TypeError, ValueError, InvalidOperation) as exc:
        raise AuditUnavailable(
            "Missing or invalid volume/notional in saved state"
        ) from exc
    if any(not value.is_finite() or value < 0 for value in values.values()):
        raise AuditUnavailable("Invalid volume/notional in saved state")
    return values


def trade_totals(symbol, start, end):
    totals = dict.fromkeys(QUANTITIES, Decimal(0))
    flags = Counter()
    cursor = start
    rows = TradeData.objects.overlapping(symbol, start, end, (1, 60, 1440)).only(
        "timestamp", "frequency", "json_data", "ok"
    )
    for row in rows:
        upper = row.timestamp + timedelta(minutes=row.frequency)
        if row.timestamp != cursor or upper > end:
            raise AuditUnavailable(
                f"Trade partition gap, overlap or partial boundary at {cursor}"
            )
        if row.json_data is None and row.ok is True:
            values = dict.fromkeys(QUANTITIES, Decimal(0))
        else:
            values = quantities((row.json_data or {}).get("candle"))
        for key in QUANTITIES:
            totals[key] += values[key]
        flags[str(row.ok)] += 1
        cursor = upper
    if cursor != end:
        raise AuditUnavailable(f"Trade partition gap at {cursor} to {end}")
    return totals, dict(flags)


def daily_candles(symbol, start, end):
    if symbol.exchange == Exchange.HYPERLIQUID:
        return native_hyperliquid_candles(symbol.api_symbol, start, end, "1d")
    if symbol.exchange == Exchange.DERIBIT:
        # Native daily candles start at 08:00 UTC; the audit uses midnight.
        hourly = candles_api(symbol, start, end, resolution="1h")
        if not hourly.empty and (
            hourly.index.has_duplicates
            or not hourly.index.equals(hourly.index.floor("h"))
        ):
            raise AuditUnavailable("Duplicate or misaligned Deribit hourly candles")
        return resample_candles(
            hourly,
            timestamp_from=start,
            timestamp_to=end,
            resolution_minutes=1440,
            source_resolution_minutes=60,
        )
    return candles_api(symbol, start, end, resolution="1d")


def check_exchange(symbol, start, totals, reference):
    reference = reference.loc[reference.index == start]
    if len(reference) > 1:
        raise AuditUnavailable(f"Duplicate exchange candles at {start}")
    saved = pd.DataFrame([totals], index=[start])
    if not any(totals.values()):
        saved = saved.iloc[:0]
    ok = TradeData._validate_candles(symbol, saved, reference)
    key = (
        "volume"
        if symbol.exchange == Exchange.BYBIT_INVERSE or "notional" not in reference
        else "notional"
    )
    expected = reference.iloc[0][key] if len(reference) else None
    actual = totals[key]
    return {
        "status": "PASS" if ok is True else "FAIL" if ok is False else "UNKNOWN",
        "quantity": key,
        "trade_data": actual,
        "exchange_quantity": expected,
        "difference": actual - expected if expected is not None else None,
    }


def cache_at(candle, boundary, *, allow_initial=False):
    row = candle.get_candle_cache(boundary, boundary, retry=True)
    if row and row.timestamp + timedelta(minutes=row.frequency) == boundary:
        return row.json_data
    if allow_initial and row is None:
        first = CandleCache.objects.filter(candle=candle).first()
        if (
            first
            and first.timestamp == boundary
            and not CandleData.objects.filter(
                candle=candle, timestamp__lt=boundary
            ).exists()
        ):
            return candle.get_initial_cache(boundary)
    raise AuditUnavailable(f"No cache ending at {boundary}")


def pending_totals(state):
    if not isinstance(state, dict):
        raise AuditUnavailable("Invalid candle cache")
    if "next" not in state:
        return dict.fromkeys(QUANTITIES, Decimal(0))
    return quantities(state["next"])


def check_candle(candle, start, end, totals):
    opening = cache_at(candle, start, allow_initial=True)
    closing = cache_at(candle, end)
    before = pending_totals(opening)
    after = pending_totals(closing)
    lower = candle.get_retry_timestamp_delete_from(start, opening)
    upper = candle.get_retry_timestamp_delete_from(end, closing)
    if lower > upper:
        raise AuditUnavailable("Pending candle boundaries run backwards")
    pending_boundaries = set()
    if "next" in opening:
        pending_boundaries.add(lower)
    if "next" in closing:
        pending_boundaries.add(upper)
    for boundary in pending_boundaries:
        if CandleData.objects.filter(candle=candle, timestamp=boundary).count() > 1:
            raise AuditUnavailable(
                f"Multiple candles share the boundary timestamp {boundary}"
            )
    completed = CandleData.objects.filter(
        candle=candle, timestamp__gte=lower, timestamp__lt=upper
    ).aggregate(count=Count("pk"), **{key: Sum(key) for key in QUANTITIES})
    comparisons = {}
    for key in QUANTITIES:
        amount = completed[key] or Decimal(0)
        accounted = amount + after[key] - before[key]
        comparisons[key] = {
            "trade_data": totals[key],
            "completed": amount,
            "pending_start": before[key],
            "pending_end": after[key],
            "accounted": accounted,
            "difference": accounted - totals[key],
        }
    return {
        "status": "PASS"
        if all(
            is_decimal_close(value["accounted"], value["trade_data"])
            for value in comparisons.values()
        )
        else "FAIL",
        "completed_candles": completed["count"],
        "candle_from": lower,
        "candle_to": upper,
        "quantities": comparisons,
    }
