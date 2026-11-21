"""Forward observations of exposure under a static target-mark stress.

Only the target perpetual mark moves, by 1%, 3% or 5%; other marks, positions,
orders and collateral quantities are fixed. The initial adapters support quote
collateral, not portfolios borrowing against volatile spot assets. Unsupported
or incomplete portfolios remain unknown. Exposure is the position at risk, not
predicted market-order volume or a simulation of successive liquidations.

The position census and portfolio observations have separate timestamps. Only
unchanged census positions enter coverage and exposure, so a newer position or
a missing account cannot silently inflate coverage. Summaries are published in
their availability hour, never backdated to the census or historical trade row.
Sizes are BTC and values are current USDC notional. The bands are cumulative;
already-breached positions are reported separately from newly exposed ones.
"""

from decimal import Decimal

BANDS = (1, 3, 5)
VERSION = 1


def summarize(census, observations, *, totals=None):
    """Coverage is a fraction of each side of the timestamped position census."""
    result = {"scenario": "target_mark_only", "version": VERSION}
    for side, sign in (("long", 1), ("short", -1)):
        sizes = {key: abs(size) for key, size in census.items() if size * sign > 0}
        sampled = sum(sizes.values(), Decimal(0))
        total = totals[side + "_size"] if totals is not None else sampled
        accounts = totals[side + "_accounts"] if totals is not None else len(sizes)
        if total < sampled or accounts < len(sizes):
            raise ValueError("Portfolio sample exceeds its reference census")
        known, unknown, already, exposed = {}, {}, [], {band: [] for band in BANDS}
        if sampled < total:
            unknown["not_sampled"] = total - sampled
        for key, size in sizes.items():
            row = observations.get(key)
            reason = "missing_portfolio" if row is None else row.get("unknown")
            if (
                row is not None
                and row.get("size") is not None
                and row["size"] != size * sign
            ):
                reason = "position_changed"
            if reason:
                unknown[reason] = unknown.get(reason, Decimal(0)) + size
                continue
            known[key] = row
            values = row["buffers"]
            if len(values) != len(BANDS) + 1 or any(not v.is_finite() for v in values):
                raise ValueError("Invalid liquidation stress buffers")
            if values[0] <= 0:
                already.append((size, row["price"]))
            else:
                breached = False
                for index, band in enumerate(BANDS, 1):
                    breached |= values[index] <= 0
                    if breached:
                        exposed[band].append((size, row["price"]))
        covered = sum((sizes[key] for key in known), Decimal(0))
        values = {
            "reference_size": total,
            "reference_accounts": accounts,
            "covered_size": covered,
            "covered_accounts": len(known),
            "coverage": covered / total if total else None,
            "unknown_size": total - covered,
            "unknown_by_reason": unknown,
        }
        for name, entries in (("already", already), *exposed.items()):
            quantity = sum((size for size, _ in entries), Decimal(0))
            values[str(name)] = {
                "size": quantity if known else None,
                "value": sum((size * price for size, price in entries), Decimal(0))
                if known
                else None,
                "accounts": len(entries) if known else None,
                "reference_share": quantity / total if known and total else None,
                "largest_account_share": max((size for size, _ in entries), default=0)
                / quantity
                if quantity
                else None,
            }
        result[side] = values
    return result
