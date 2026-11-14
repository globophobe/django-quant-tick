"""Settled hourly funding, labelled by settlement rather than OI bar start.

Before 2023-06-08 the provider spreads eight-hour settlements across hours.
Exclude those synthetic hourly rates from the funding-event collector.
"""

import pandas as pd
from pandas import DataFrame

from quant_tick.exchanges.funding import ExchangeFunding

from .api import number, oi_pages

HOURLY_FUNDING_START = pd.Timestamp("2023-06-08", tz="UTC")


class FemtosFunding(ExchangeFunding):
    interval = pd.Timedelta("1h")

    @classmethod
    def is_known_missing_timestamp(cls, timestamp):
        return timestamp < HOURLY_FUNDING_START


def femtos_funding(api_symbol, timestamp_from, timestamp_to):
    start = max(pd.Timestamp(timestamp_from).ceil("h"), HOURLY_FUNDING_START)
    end = pd.Timestamp(timestamp_to).ceil("h")
    rows = []
    for _, _, points in oi_pages(
        api_symbol, start - FemtosFunding.interval, end - FemtosFunding.interval
    ):
        for timestamp, point in points:
            timestamp += FemtosFunding.interval
            if (
                point["funding"] is None
                or not timestamp_from <= timestamp < timestamp_to
            ):
                continue
            rows.append(
                {
                    "timestamp": timestamp,
                    "funding_rate": number(point["funding"], nonnegative=False),
                    "funding_source": "femtos",
                    "source_bar_start": timestamp - FemtosFunding.interval,
                }
            )
    return DataFrame(
        rows,
        columns=[
            "timestamp",
            "funding_rate",
            "funding_source",
            "source_bar_start",
        ],
    ).set_index("timestamp")
