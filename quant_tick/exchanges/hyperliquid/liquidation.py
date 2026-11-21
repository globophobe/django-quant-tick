"""Static mark stress for standard and unified USDC portfolios.

Use isolated raw collateral for isolated positions. Standard cross margin uses
the native DEX account equity and maintenance; unified cross margin uses the
shared USDC spot balance, less isolated allocations, and maintenance across all
USDC DEXes. The account balance is counted once. Portfolio margin, unresolved
account modes and incomplete DEX responses are excluded, not treated as safe.

Margin tiers are captured with each forward snapshot. Historical position rows
cannot supply historical account modes or risk parameters. The venue's own
liquidation price independently checks each supported account's calculation.
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
    states = state.get("clearinghouseState", {})
    native = states.get("native", {})
    positions = [
        p["position"]
        for p in native.get("assetPositions", [])
        if p["position"]["coin"] == "BTC"
    ]
    if len(positions) > 1:
        raise ValueError("Duplicate Hyperliquid BTC position")
    if not positions:
        return {"size": Decimal(0), "unknown": "position_changed"}
    position = positions[0]
    size = number(position["szi"])
    result = {"size": size}
    if not size:
        return {**result, "unknown": "position_changed"}
    mode = state.get("userAbstraction")
    if mode not in ("disabled", "unifiedAccount"):
        return {**result, "unknown": "unsupported_account_mode"}
    moment = pd.Timestamp(native["time"], unit="ms", tz="UTC")
    if not pd.Timedelta(0) <= timestamp(received) - moment <= pd.Timedelta(minutes=5):
        return {**result, "unknown": "stale_portfolio"}
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
        return {**result, "unknown": "missing_margin_tiers"}
    target_maintenance = maintenance(size, price, tiers)
    leverage = position["leverage"]
    if leverage["type"] == "isolated":
        if leverage.get("rawUsd") is None:
            return {**result, "unknown": "missing_isolated_collateral"}
        equity = number(leverage["rawUsd"]) + size * price
        required = target_maintenance
    elif leverage["type"] != "cross":
        return {**result, "unknown": "unsupported_margin_mode"}
    elif mode == "disabled":
        equity = number(native["crossMarginSummary"]["accountValue"])
        required = number(native["crossMaintenanceMarginUsed"])
    else:
        balances = state.get("spotClearinghouseState", {}).get("balances")
        if not isinstance(balances, list):
            return {**result, "unknown": "missing_spot_collateral"}
        cash = [v for v in balances if v.get("token") == 0]
        if len(cash) != 1 or len(dexes) != len(metas):
            return {**result, "unknown": "incomplete_unified_portfolio"}
        equity, required = number(cash[0]["total"]), Decimal(0)
        for index, meta in enumerate(metas):
            if meta.get("collateralToken") != 0:
                continue
            name = "native" if index == 0 else dexes[index]["name"]
            if name not in states:
                return {**result, "unknown": "incomplete_unified_portfolio"}
            dex = states[name]
            if abs(int(dex["time"]) - int(native["time"])) > 1000:
                return {**result, "unknown": "inconsistent_portfolio_times"}
            required += number(dex["crossMaintenanceMarginUsed"])
            for asset in dex["assetPositions"]:
                value = asset["position"]
                if value["leverage"]["type"] == "isolated":
                    equity -= number(value["marginUsed"])
                elif value["leverage"]["type"] != "cross":
                    return {**result, "unknown": "unsupported_margin_mode"}
    if required < target_maintenance - Decimal("0.00001"):
        return {**result, "unknown": "inconsistent_maintenance"}

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
            return {**result, "unknown": "venue_liquidation_mismatch"}
    sign = 1 if size > 0 else -1
    buffers = [buffer(price * (1 - sign * Decimal(band) / 100)) for band in (0, *BANDS)]
    if venue_price is None and any(v <= 0 for v in buffers):
        return {**result, "unknown": "venue_liquidation_mismatch"}
    return {
        **result,
        "price": price,
        "timestamp": moment.isoformat(),
        "buffers": buffers,
    }


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
    observations = {}
    for key, state in states.items():
        try:
            observations[key] = observe(state, metas, dexes, received)
        except (KeyError, TypeError):
            observations[key] = {"unknown": "incomplete_portfolio"}
    completed = get_current_time()
    summary = {
        **summarize(census, observations, totals=totals),
        "reference_timestamp": anchor["snapshot_time"],
        "reference_block": anchor["block_number"],
        "observed_from": started,
        "available_at": completed,
        "sampling": "largest_positions",
        "account_limit": ACCOUNT_LIMIT,
        "coverage_basis": "unchanged_reference_positions",
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
