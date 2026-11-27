"""Position-size flow brackets and minute-sampled open interest.

Flow tiers value the taker's absolute position immediately before each fill at
that fill's price, including market liquidations. They are not trade-size or
wallet-wealth brackets. A reversal allocates closing and opening quantities to
the same pre-fill bracket. All quantities use the market's base asset.
min_position_value is an inclusive quote-value boundary; the next tier supplies
the exclusive upper boundary, and the final tier is unbounded. Each tier's
liquidation subset contains market executions involving a liquidated account;
the taker is not necessarily that account. Subtract it to obtain executions
without liquidation involvement. Backstop transfers and ADL are excluded.

Concentration uses each account's gross aggressive buy/sell base quantity during
the interval, not net flow or wallet wealth. An account can appear on both sides.
Profit/loss magnitudes split reported execution PnL by maker/taker and the old
position's direction, including reversals and market liquidations. Fees, funding
and nontrade position events remain in their separate fields.

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

PNL_FIELDS = tuple(
    f"{role}_{side}_realized_{result}"
    for role in ("taker", "maker")
    for side in ("long", "short")
    for result in ("profit", "loss")
)


def empty_flow_tiers():
    return [
        {
            "tier": tier,
            "min_position_value": Decimal(bound),
            **dict.fromkeys(TAKER_FIELDS, Decimal(0)),
            "liquidation": dict.fromkeys(TAKER_FIELDS, Decimal(0)),
        }
        for tier, bound in enumerate(POSITION_BOUNDS, 1)
    ]


def add_taker_fill(tiers, before, after, value, *, liquidated=False):
    size = abs(after - before)
    if not size or value <= 0:
        raise ValueError("Position-size flow requires a positive execution")
    position_value = abs(before) * value / size
    tier = sum(position_value >= bound for bound in POSITION_BOUNDS) - 1
    row = tiers[tier]
    side = "buy" if after > before else "sell"
    amounts = {"taker_" + side + "_size": size, "taker_" + side + "_value": value}
    for direction, sign in (("long", 1), ("short", -1)):
        change = max(sign * after, 0) - max(sign * before, 0)
        action = "open" if change > 0 else "close"
        amounts["taker_" + direction + "_" + action + "_size"] = abs(change)
    for key, amount in amounts.items():
        row[key] += amount
        if liquidated:
            row["liquidation"][key] += amount


def validate_flow_tiers(tiers, totals):
    if [row["tier"] for row in tiers] != list(range(1, len(POSITION_BOUNDS) + 1)):
        raise ValueError("Invalid position-size flow tiers")
    for row in tiers:
        for part in (row, row["liquidation"]):
            if any(
                not part[k].is_finite() or not 0 <= part[k] <= row[k]
                for k in TAKER_FIELDS
            ):
                raise ValueError("Invalid position-size flow quantities")
            if (
                part["taker_buy_size"]
                != part["taker_long_open_size"] + part["taker_short_close_size"]
                or part["taker_sell_size"]
                != part["taker_short_open_size"] + part["taker_long_close_size"]
            ):
                raise ValueError("Position-size flow components do not reconcile")
    if any(sum(row[k] for row in tiers) != totals[k] for k in TAKER_FIELDS):
        raise ValueError("Position-size flows do not reconcile with market totals")
    for unit in ("size", "value"):
        if (
            sum(
                row["liquidation"]["taker_buy_" + unit]
                + row["liquidation"]["taker_sell_" + unit]
                for row in tiers
            )
            != totals["liquidated_" + unit]
        ):
            raise ValueError("Position-size liquidation flows do not reconcile")


def add_realized_pnl(row, before, after, pnl, *, taker):
    if not pnl:
        return
    side, sign = ("long", 1) if before > 0 else ("short", -1)
    if max(sign * before, 0) <= max(sign * after, 0):
        raise ValueError("Realized PnL lacks a closing position")
    role = "taker" if taker else "maker"
    result = "profit" if pnl > 0 else "loss"
    row[f"{role}_{side}_realized_{result}"] += abs(pnl)


def validate_realized_pnl(row):
    if any(not row[key].is_finite() or row[key] < 0 for key in PNL_FIELDS):
        raise ValueError("Invalid realized profit/loss amounts")
    for role in ("taker", "maker"):
        net = sum(
            row[f"{role}_{side}_realized_profit"] - row[f"{role}_{side}_realized_loss"]
            for side in ("long", "short")
        )
        if net != row[role + "_realized_pnl"]:
            raise ValueError("Realized profits and losses do not reconcile")


def concentration_values(accounts, sides, totals):
    counts = [sides[side]["accounts"] for side in ("buy", "sell")]
    if not max(counts) <= accounts <= sum(counts):
        raise ValueError("Invalid active taker account counts")
    result = {"accounts": accounts}
    for side in ("buy", "sell"):
        row = sides[side]
        size, count = row["size"], row["accounts"]
        first, five = row["top1_size"], row["top5_size"]
        if (
            not all(value.is_finite() for value in (size, first, five))
            or size != totals["taker_" + side + "_size"]
            or count < 0
            or (count == 0) != (size == 0)
            or not 0 <= first <= five <= size
            or (count > 0 and first <= 0)
            or five > min(5, count) * first
            or size > count * first
            or (count <= 5 and five != size)
        ):
            raise ValueError(
                "Taker concentration does not reconcile with market volume"
            )
        result[side] = {
            **row,
            "top1_share": first / size if size else None,
            "top5_share": five / size if size else None,
        }
    return result


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
