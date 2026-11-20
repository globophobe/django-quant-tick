"""Comparable execution-size brackets and minute-sampled position summaries.

Flow tiers value the taker's absolute position immediately before each fill at
that fill's price, including market liquidations. They are not trade-size or
wallet-wealth brackets. A reversal allocates closing and opening quantities to
the same pre-fill bracket. All quantities use the market's base asset.
min_position_value is an inclusive quote-value boundary; the next tier supplies
the exclusive upper boundary, and the final tier is unbounded.

Minute OI uses one side of matched positions on both venues, independently of
their native OI convention. Extrema include the opening and closing anchors
and completed minute samples; they are not tick-level intraminute extrema.
Increases/decreases sum positive/negative minute changes, not gross openings
and closings. Missing minutes carry state only inside a fully validated replay.
"""

from decimal import Decimal

POSITION_BOUNDS = (0, 10_000, 100_000, 1_000_000, 10_000_000)
TAKER_FIELDS = (
    "taker_buy_size",
    "taker_sell_size",
    "taker_long_open_size",
    "taker_long_close_size",
    "taker_short_open_size",
    "taker_short_close_size",
    "taker_buy_value",
    "taker_sell_value",
)


def empty_flow_tiers():
    return [
        {
            "tier": tier,
            "min_position_value": Decimal(bound),
            **dict.fromkeys(TAKER_FIELDS, Decimal(0)),
        }
        for tier, bound in enumerate(POSITION_BOUNDS, 1)
    ]


def add_taker_fill(tiers, before, after, value):
    size = abs(after - before)
    if not size or value <= 0:
        raise ValueError("Position-size flow requires a positive execution")
    position_value = abs(before) * value / size
    tier = sum(position_value >= bound for bound in POSITION_BOUNDS) - 1
    row = tiers[tier]
    side = "buy" if after > before else "sell"
    row["taker_" + side + "_size"] += size
    row["taker_" + side + "_value"] += value
    for direction, sign in (("long", 1), ("short", -1)):
        change = max(sign * after, 0) - max(sign * before, 0)
        action = "open" if change > 0 else "close"
        row["taker_" + direction + "_" + action + "_size"] += abs(change)


def validate_flow_tiers(tiers, totals):
    if [row["tier"] for row in tiers] != list(range(1, len(POSITION_BOUNDS) + 1)):
        raise ValueError("Invalid position-size flow tiers")
    for row in tiers:
        if any(not row[k].is_finite() or row[k] < 0 for k in TAKER_FIELDS):
            raise ValueError("Invalid position-size flow quantities")
        if (
            row["taker_buy_size"]
            != row["taker_long_open_size"] + row["taker_short_close_size"]
            or row["taker_sell_size"]
            != row["taker_short_open_size"] + row["taker_long_close_size"]
        ):
            raise ValueError("Position-size flow components do not reconcile")
    if any(sum(row[k] for row in tiers) != totals[k] for k in TAKER_FIELDS):
        raise ValueError("Position-size flows do not reconcile with market totals")


def minute_open_interest(before, after, changes):
    current = high = low = peak = before
    increase = decrease = drawdown = Decimal(0)
    if not before.is_finite() or before < 0:
        raise ValueError("Invalid minute open-interest anchor")
    for minute in sorted(changes):
        change = changes[minute]
        if not change.is_finite():
            raise ValueError("Invalid minute open-interest change")
        current += change
        if current < 0:
            raise ValueError("Minute open interest became negative")
        high, low = max(high, current), min(low, current)
        peak = max(peak, current)
        drawdown = max(drawdown, peak - current)
        increase += max(change, 0)
        decrease += max(-change, 0)
    if current != after:
        raise ValueError("Minute open interest does not reconcile with its anchors")
    return {
        "minute_open_interest_open": before,
        "minute_open_interest_close": after,
        "minute_open_interest_high": high,
        "minute_open_interest_low": low,
        "minute_open_interest_increase": increase,
        "minute_open_interest_decrease": decrease,
        "minute_open_interest_max_drawdown": drawdown,
        "minute_open_interest_basis": "one_sided_positions",
    }
