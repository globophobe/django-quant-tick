"""Optional fixed quote-value brackets for candle trade records.

volume_brackets is a strictly increasing list of positive cutoffs. For example,
[10000, 100000] creates [0, 10000), [10000, 100000), and [100000, infinity).
volumeBrackets.buckets stores minVolume (inclusive), maxVolume (exclusive;
None for the last bracket), and the usual volume/notional/tick buy and total
fields. Sell amounts are total minus buy. Ticks count underlying fills, not
aggregated records. Raw inputs classify fills; aggregated/filtered inputs
classify retained execution aggregates, not account positions or whole orders.

Filtered records retain an execution's own quantities alongside totals that can
include smaller trades of either side. Only the own quantities are classified;
the remainder stays in volumeBrackets.unclassified because its individual sizes
are unavailable. Brackets plus that remainder reconcile to ordinary candle totals.
"""

from bisect import bisect_right
from decimal import Decimal, InvalidOperation

from pandas import DataFrame

AMOUNT_FIELDS = ("volume", "buyVolume", "notional", "buyNotional")
COUNT_FIELDS = ("ticks", "buyTicks")
FIELDS = (*AMOUNT_FIELDS, *COUNT_FIELDS)


def parse_volume_brackets(values: list | None) -> list[Decimal]:
    if values is None:
        return []
    message = "volume_brackets must be a strictly increasing list of positive numbers"
    if not isinstance(values, list):
        raise TypeError(message)
    bounds = []
    for value in values:
        try:
            bound = Decimal(str(value))
        except InvalidOperation as exc:
            raise ValueError(message) from exc
        if not bound.is_finite() or bound <= (bounds[-1] if bounds else 0):
            raise ValueError(message)
        bounds.append(bound)
    return bounds


def empty_totals() -> dict:
    return dict.fromkeys(AMOUNT_FIELDS, Decimal(0)) | dict.fromkeys(COUNT_FIELDS, 0)


def aggregate_volume_brackets(
    df: DataFrame, bounds: list[Decimal], totals: dict
) -> dict:
    if not {"volume", "notional", "tickRule"}.issubset(df.columns):
        raise ValueError("Volume brackets require trade records")
    buckets = [
        {"minVolume": lower, "maxVolume": upper, **empty_totals()}
        for lower, upper in zip([Decimal(0), *bounds], [*bounds, None], strict=True)
    ]
    for row in df[df.volume.notna()].itertuples(index=False):
        try:
            volume, notional = Decimal(str(row.volume)), Decimal(str(row.notional))
            ticks = Decimal(str(getattr(row, "ticks", 1)))
        except InvalidOperation as exc:
            raise ValueError("Invalid trade quantities for volume brackets") from exc
        if (
            any(
                not value.is_finite() or value <= 0
                for value in (volume, notional, ticks)
            )
            or ticks != ticks.to_integral_value()
            or row.tickRule not in (1, -1)
        ):
            raise ValueError("Invalid trade quantities for volume brackets")
        bucket = buckets[bisect_right(bounds, volume)]
        bucket["volume"] += volume
        bucket["notional"] += notional
        bucket["ticks"] += int(ticks)
        if row.tickRule == 1:
            bucket["buyVolume"] += volume
            bucket["buyNotional"] += notional
            bucket["buyTicks"] += int(ticks)
    remainder = {
        key: totals[key] - sum(bucket[key] for bucket in buckets) for key in FIELDS
    }
    if any(value < 0 for value in remainder.values()) or any(
        remainder[buy] > remainder[total]
        for buy, total in zip(FIELDS[1::2], FIELDS[::2], strict=True)
    ):
        raise ValueError("Volume brackets do not reconcile with candle totals")
    return {"buckets": buckets, "unclassified": remainder}


def merge_volume_brackets(previous: dict | None, current: dict | None) -> dict:
    if (
        previous is None
        or current is None
        or [
            (bucket["minVolume"], bucket["maxVolume"]) for bucket in previous["buckets"]
        ]
        != [(bucket["minVolume"], bucket["maxVolume"]) for bucket in current["buckets"]]
    ):
        raise ValueError(
            "volume_brackets changed; use a new candle or rebuild its history"
        )
    return {
        "buckets": [
            {**after, **{key: before[key] + after[key] for key in FIELDS}}
            for before, after in zip(
                previous["buckets"], current["buckets"], strict=True
            )
        ],
        "unclassified": {
            key: previous["unclassified"][key] + current["unclassified"][key]
            for key in FIELDS
        },
    }
