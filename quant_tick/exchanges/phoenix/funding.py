from datetime import UTC, datetime
from decimal import Decimal

import pandas as pd
from pandas import DataFrame

from quant_tick.exchanges.funding import ExchangeFunding
from quant_tick.lib import iter_chunks

from .api import get_phoenix_response, to_millis
from .constants import FUNDING_MAX_RESULTS


class PhoenixFunding(ExchangeFunding):
    interval = pd.Timedelta("1h")
    timestamp_anomaly_tolerance = pd.Timedelta("1min")


def phoenix_funding(
    api_symbol: str, timestamp_from: datetime, timestamp_to: datetime
) -> DataFrame:
    """Convert hourly funding percentages to fractional rates at the hourly boundary.

    Response timestamps are seconds; request bounds are milliseconds. Preserve
    the raw timestamp when the hourly snapshot lands slightly after the hour.
    """
    symbol = api_symbol.strip().upper()
    rows = []
    for start, end in iter_chunks(
        timestamp_from,
        timestamp_to,
        value=PhoenixFunding.interval * (FUNDING_MAX_RESULTS - 1),
    ):
        result = get_phoenix_response(
            f"/v1/funding/{symbol}/rates",
            {
                "startTime": to_millis(start),
                "endTime": to_millis(end) - 1,
                "limit": FUNDING_MAX_RESULTS,
            },
        )
        if result["symbol"] != symbol:
            raise ValueError("Phoenix funding returned an unexpected market")
        if len(result["rates"]) >= FUNDING_MAX_RESULTS:
            raise ValueError("Phoenix funding window reached the response limit")
        for row in result["rates"]:
            timestamp = datetime.fromtimestamp(int(row["timestamp"]), tz=UTC)
            if not start <= timestamp < end:
                raise ValueError("Phoenix funding is outside the requested window")
            rate = Decimal(row["fundingRatePercentage"]) / 100
            if not rate.is_finite():
                raise ValueError("Phoenix funding rate is not finite")
            rows.append({"timestamp": timestamp, "funding_rate": rate})
    if not rows:
        return PhoenixFunding.empty_frame(["funding_rate"])
    frame = PhoenixFunding.normalize_frame(
        DataFrame(rows), timestamp_from, timestamp_to
    )
    if not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError("Phoenix funding timestamps are not strictly increasing")
    return frame
