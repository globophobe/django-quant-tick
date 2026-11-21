"""Static mark stress for standard and unified USDC portfolios.

Use isolated raw collateral for isolated positions. Standard cross margin uses
the native DEX account equity and maintenance; unified cross margin uses the
shared USDC spot balance, less isolated allocations, and maintenance across all
USDC DEXes. The account balance is counted once. Portfolio margin, unresolved
account modes and incomplete DEX responses are excluded, not treated as safe.

Margin tiers are captured with each forward snapshot. Historical position rows
cannot supply historical account modes or risk parameters. The venue's own
liquidation price independently checks each supported account's calculation.

The indexed census selects accounts only. Positions and collateral are taken
together from each fresh RPC state, including changes of size or direction.
Coverage describes the observed sample, not all current OI. The dated selection
coverage and unobserved accounts retain their own reference quantities.
"""

import os
from decimal import Decimal

import pandas as pd

from quant_tick.lib import get_current_time

from ..liquidation_exposure import BANDS, VERSION, summarize
from ..quicknode.api import number, timestamp
from ..quicknode.portfolios import ACCOUNT_LIMIT, portfolio_states, position_census
from .api import post_hyperliquid_info


def maintenance(size, price, tiers):
    notional = abs(size) * price
    value = Decimal(0)
    previous = None
    for index, tier in enumerate(tiers):
        lower, leverage = (
            number(tier["lowerBound"]),
            number(tier["maxLeverage"], positive=True),
        )
        if (index == 0 and lower != 0) or (previous is not None and lower <= previous):
            raise ValueError("Invalid Hyperliquid maintenance tiers")
        upper = (
            number(tiers[index + 1]["lowerBound"])
            if index + 1 < len(tiers)
            else notional
        )
        value += max(Decimal(0), min(notional, upper) - lower) / (2 * leverage)
        previous = lower
    if not tiers:
        raise ValueError("Missing Hyperliquid maintenance tiers")
    return value


def observe(state, metas, dexes, received):
    try:
        native = state["clearinghouseState"]["native"]
        moment = pd.Timestamp(native["time"], unit="ms", tz="UTC")
        if (
            not pd.Timedelta(0)
            <= timestamp(received) - moment
            <= pd.Timedelta(minutes=5)
        ):
            return {"unknown": "stale_portfolio"}
        if not isinstance(native["assetPositions"], list):
            return {"unknown": "incomplete_portfolio"}
        positions = [
            p["position"]
            for p in native["assetPositions"]
            if p["position"]["coin"] == "BTC"
        ]
        size = number(positions[0]["szi"]) if positions else Decimal(0)
    except (KeyError, TypeError):
        return {"unknown": "incomplete_portfolio"}
    if len(positions) > 1:
        raise ValueError("Duplicate Hyperliquid BTC position")
    result = {"size": size, "timestamp": moment.isoformat()}
    if not size:
        return result
    try:
        return {**result, **stress(state, positions[0], metas, dexes)}
    except (KeyError, TypeError):
        return {**result, "unknown": "incomplete_portfolio"}


def stress(state, position, metas, dexes):
    states = state["clearinghouseState"]
    native = states["native"]
    size = number(position["szi"])
    mode = state.get("userAbstraction")
    if mode not in ("disabled", "unifiedAccount"):
        return {"unknown": "unsupported_account_mode"}
    price = number(position["positionValue"], positive=True) / abs(size)
    metadata = next(m for m in metas[0]["universe"] if m["name"] == "BTC")
    table_id = metadata["marginTableId"]
    tiers = next(
        (t[1]["marginTiers"] for t in metas[0]["marginTables"] if t[0] == table_id),
        None,
    )
    if tiers is None and 0 < table_id < 50:
        tiers = [{"lowerBound": "0", "maxLeverage": table_id}]
    if tiers is None:
        return {"unknown": "missing_margin_tiers"}
    target_maintenance = maintenance(size, price, tiers)
    leverage = position["leverage"]
    if leverage["type"] == "isolated":
        if leverage.get("rawUsd") is None:
            return {"unknown": "missing_isolated_collateral"}
        equity = number(leverage["rawUsd"]) + size * price
        required = target_maintenance
    elif leverage["type"] != "cross":
        return {"unknown": "unsupported_margin_mode"}
    elif mode == "disabled":
        equity = number(native["crossMarginSummary"]["accountValue"])
        required = number(native["crossMaintenanceMarginUsed"])
    else:
        balances = state.get("spotClearinghouseState", {}).get("balances")
        if not isinstance(balances, list):
            return {"unknown": "missing_spot_collateral"}
        cash = [v for v in balances if v.get("token") == 0]
        if len(cash) != 1 or len(dexes) != len(metas):
            return {"unknown": "incomplete_unified_portfolio"}
        equity, required = number(cash[0]["total"]), Decimal(0)
        for index, meta in enumerate(metas):
            if meta.get("collateralToken") != 0:
                continue
            name = "native" if index == 0 else dexes[index]["name"]
            if name not in states:
                return {"unknown": "incomplete_unified_portfolio"}
            dex = states[name]
            if abs(int(dex["time"]) - int(native["time"])) > 1000:
                return {"unknown": "inconsistent_portfolio_times"}
            required += number(dex["crossMaintenanceMarginUsed"])
            for asset in dex["assetPositions"]:
                value = asset["position"]
                if value["leverage"]["type"] == "isolated":
                    equity -= number(value["marginUsed"])
                elif value["leverage"]["type"] != "cross":
                    return {"unknown": "unsupported_margin_mode"}
    if required < target_maintenance - Decimal("0.00001"):
        return {"unknown": "inconsistent_maintenance"}

    def buffer(stressed_price):
        return (
            equity
            + size * (stressed_price - price)
            - required
            + target_maintenance
            - maintenance(size, stressed_price, tiers)
        )

    venue_price = position.get("liquidationPx")
    if venue_price is not None:
        boundary = number(venue_price, positive=True)
        tolerance = max(Decimal("0.00001"), abs(size) * price * Decimal("1e-10"))
        if abs(buffer(boundary)) > tolerance:
            return {"unknown": "venue_liquidation_mismatch"}
    sign = 1 if size > 0 else -1
    buffers = [buffer(price * (1 - sign * Decimal(band) / 100)) for band in (0, *BANDS)]
    if venue_price is None and any(v <= 0 for v in buffers):
        return {"unknown": "venue_liquidation_mismatch"}
    return {"price": price, "buffers": buffers}


def selection_coverage(census, totals, observations):
    """All quantities here belong to the dated selection index, not live OI."""
    result = {}
    for side, sign in (("long", 1), ("short", -1)):
        positions = {key: abs(size) for key, size in census.items() if size * sign > 0}
        sampled = sum(positions.values(), Decimal(0))
        total, accounts = totals[side + "_size"], totals[side + "_accounts"]
        if sampled > total or len(positions) > accounts:
            raise ValueError("Portfolio sample exceeds its reference census")
        unknown = {}
        missing = 0
        for key, size in positions.items():
            row = observations.get(key, {})
            if row.get("size") is not None:
                continue
            reason = row.get("unknown", "missing_portfolio")
            unknown[reason] = unknown.get(reason, Decimal(0)) + size
            missing += 1
        result[side] = {
            "size": total,
            "accounts": accounts,
            "sampled_size": sampled,
            "sampled_accounts": len(positions),
            "sampled_share": sampled / total if total else None,
            "unobserved_size": sum(unknown.values(), Decimal(0)),
            "unobserved_accounts": missing,
            "unobserved_by_reason": unknown,
        }
    return result


def fetch_snapshot():
    if not os.environ.get("QUICKNODE_HYPERCORE_URL", "").strip():
        return None
    started = get_current_time()
    anchor, raw_positions, totals = position_census(ACCOUNT_LIMIT)
    metas = post_hyperliquid_info({"type": "allPerpMetas"})
    dexes = post_hyperliquid_info({"type": "perpDexs"})
    metadata = next(m for m in metas[0]["universe"] if m["name"] == "BTC")
    scale = Decimal(10) ** -metadata["szDecimals"]
    census = {key: size * scale for key, size in raw_positions.items()}
    totals = {
        key: number(value) * scale if key.endswith("_size") else value
        for key, value in totals.items()
    }
    users = list(census)[:ACCOUNT_LIMIT]
    states, failed = portfolio_states(users)
    received = get_current_time()
    observations = {
        key: observe(state, metas, dexes, received) for key, state in states.items()
    }
    positions = {key: row["size"] for key, row in observations.items() if "size" in row}
    completed = get_current_time()
    summary = {
        **summarize(positions, observations),
        "selection": {
            "timestamp": anchor["snapshot_time"],
            "block": anchor["block_number"],
            **selection_coverage(census, totals, observations),
        },
        "observed_from": started,
        "available_at": completed,
        "sampling": "largest_positions",
        "account_limit": ACCOUNT_LIMIT,
        "coverage_basis": "observed_sample_positions",
        "portfolio_timestamp_from": min(
            (row["timestamp"] for row in observations.values() if "timestamp" in row),
            default=None,
        ),
        "portfolio_timestamp_to": max(
            (row["timestamp"] for row in observations.values() if "timestamp" in row),
            default=None,
        ),
    }
    return {
        "version": VERSION,
        "available_at": completed,
        "summary": summary,
        "census": {"anchor": anchor, "positions": census, "totals": totals},
        "metas": metas,
        "dexes": dexes,
        "portfolios": states,
        "failed_accounts": failed,
        "observations": observations,
    }
