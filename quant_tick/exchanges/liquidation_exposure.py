"""Forward observations of exposure under a static target-mark stress.

Only the target perpetual mark moves, by 1%, 3% or 5%; other marks, positions,
orders and collateral quantities are fixed. The initial adapters support quote
collateral, not portfolios borrowing against volatile spot assets. Unsupported
or incomplete portfolios remain unknown. Exposure is the position at risk, not
predicted market-order volume or a simulation of successive liquidations.

Position quantities and stress buffers must describe the same observation.
Coverage measures the supported fraction of observed position size. It is
market-wide only when the adapter observes the entire market atomically;
otherwise selection coverage and missing observations are reported separately.
Summaries are published in their availability hour, never backdated to the
account-selection index or historical trade row.
Sizes are BTC and values are current USDC notional. The bands are cumulative;
already-breached positions are reported separately from newly exposed ones.
"""

from decimal import Decimal

BANDS = (1, 3, 5)
VERSION = 2


def summarize(positions, observations):
    """Aggregate observed positions; never extrapolate unobserved quantities."""
    result = {"scenario": "target_mark_only", "version": VERSION}
    for side, sign in (("long", 1), ("short", -1)):
        sizes = {key: abs(size) for key, size in positions.items() if size * sign > 0}
        total = sum(sizes.values(), Decimal(0))
        known, unknown, already, exposed = {}, {}, [], {band: [] for band in BANDS}
        for key, size in sizes.items():
            row = observations.get(key)
            reason = "missing_portfolio" if row is None else row.get("unknown")
            if (
                row is not None
                and row.get("size") is not None
                and row["size"] != size * sign
            ):
                raise ValueError(
                    "Position and liquidation observation quantities disagree"
                )
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
            "observed_size": total,
            "observed_accounts": len(sizes),
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
                "observed_share": quantity / total if known and total else None,
                "largest_account_share": max((size for size, _ in entries), default=0)
                / quantity
                if quantity
                else None,
            }
        result[side] = values
    return result
