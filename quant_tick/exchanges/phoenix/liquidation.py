"""Static portfolio stress from one finalized program-account snapshot.

Hot collateral and positions come from GTI/ATB, cold ones from trader accounts.
Quote collateral is counted once, with every market's unsettled funding,
discounted positive PnL, size-dependent margin and resting-order reserves.
Arithmetic follows rise-public's margin_calc and compute_market_margin, in
integer quote lots. No orders execute during the price scenario. Enabled price
components must be fresh under their captured protocol thresholds; this does
not estimate stale-oracle margin inflation.
Scenario marks round outward to the next adverse price tick.

Spot collateral, unknown extensions and stale market marks remain unsupported.
Existing position-only replay checkpoints cannot backfill this observation.
The stored inputs retain decoded portfolios and risk parameters, rather than
unrelated program accounts, so each scenario remains reproducible.
"""

from decimal import Decimal
from itertools import pairwise

from quant_tick.lib import get_current_time

from ..liquidation_exposure import BANDS, VERSION, summarize
from ..quicknode.solana import available
from .accounts import TAGS, account_bytes, decode58, tree, unpack
from .snapshots import fetch_position_snapshot


def portfolios(raw):
    indexes, traders = {}, {}
    for row in raw["value"]:
        data = account_bytes(row["account"])
        kind = TAGS.get(data[:8])
        if kind == "trader":
            traders[decode58(row["pubkey"])] = data
        elif kind:
            indexes[kind] = data
    gti = tree(indexes["global_trader_index"], "32s", 16)
    atb = tree(indexes["active_trader_buffer"], "II", 64)
    result = {
        key[0]: {"collateral": unpack("q", state, 0)[0], "positions": {}}
        for key, state in gti.values()
    }
    for (index, asset), position in atb.values():
        key = gti[index][0][0]
        result[key]["positions"][asset] = position
    for key, data in traders.items():
        if key in result:
            continue
        count = unpack("Q", data, 224)[0]
        rows = {}
        for index in range(count):
            offset = 240 + index * 40
            asset = unpack("Q", data, offset)[0] & 0xFFFFFFFF
            if asset in rows:
                raise ValueError("Duplicate Phoenix portfolio asset")
            rows[asset] = data[offset + 8 : offset + 40] + bytes(32)
        result[key] = {"collateral": unpack("q", data, 88)[0], "positions": rows}
    data = indexes["perp_asset_map"]
    markets = {}
    for index in range(unpack("I", data, 32)[0]):
        start = 48 + index * 1584
        meta = data[start + 16 : start + 1584]
        if meta[1450] or meta[-1] != 0:
            continue
        tick_size, lower, decimals, _, upper = unpack("QHbBH", meta, 936)
        asset = lower | upper << 16
        tiers = [unpack("QQQ", meta, 1016 + i * 24) for i in range(4)]
        if (
            not tick_size
            or any(
                bound <= 0 or leverage <= 0 or risk > 10000
                for bound, leverage, risk in tiers
            )
            or any(
                a[0] >= b[0] or a[1] < b[1] or a[2] > b[2] for a, b in pairwise(tiers)
            )
        ):
            raise ValueError("Invalid Phoenix margin tiers")
        markets[asset] = {
            "symbol": data[start : start + 16].rstrip(b"\0").decode("ascii"),
            "tick_size": tick_size,
            "decimals": decimals,
            "mark_slot": unpack("Q", meta, 16)[0],
            "mark_ticks": unpack("Q", meta, 24)[0],
            "price_components": [
                {
                    "weight": unpack("Q", meta, offset)[0],
                    "stale_threshold": unpack("Q", meta, offset + 8)[0],
                    "prices": [
                        unpack("QQ", meta, start + i * 16) for i in range(count)
                    ],
                }
                for start, offset, count in ((32, 112, 5), (168, 248, 5), (264, 280, 1))
            ],
            "tiers": tiers,
            "upnl_factor": unpack("Q", meta, 1112)[0],
            "maintenance_factor": unpack("H", meta, 1192)[0],
            "funding": unpack("q", meta, 1312)[0],
        }
        if (
            not 0 <= markets[asset]["upnl_factor"] <= 10000
            or not 0 < markets[asset]["maintenance_factor"] <= 10000
        ):
            raise ValueError("Invalid Phoenix margin risk factors")
    return result, markets, indexes["global_configuration"][505]


def tier_value(tiers, size, column):
    for index, tier in enumerate(tiers):
        if size <= tier[0]:
            if not index:
                return tier[column]
            before = tiers[index - 1]
            return (
                before[column] * (tier[0] - size) + tier[column] * (size - before[0])
            ) // (tier[0] - before[0])
    return 1 if column == 1 else 10000


def ceil_div(value, divisor):
    return -(-value // divisor)


def margin(position, market, ticks):
    """Return effective PnL plus funding and maintenance, both in quote lots."""
    size, quote, funding = unpack("qqq", position, 0)
    ask, _, bid, _, ask_size, ask_reduce, bid_size, bid_reduce = unpack(
        "8I", position, 32
    )
    ask_total = ask_size + min(ask_reduce, max(size, 0))
    bid_total = bid_size + min(bid_reduce, max(-size, 0))
    price = ticks * market["tick_size"]
    tiers = market["tiers"]
    initial = ceil_div(abs(size) * price, tier_value(tiers, abs(size), 1))
    additions = []
    for delta in (bid_size, -ask_size):
        exposure = abs(size + delta)
        extra = max(
            0, ceil_div(exposure * price, tier_value(tiers, exposure, 1)) - initial
        )
        additions.append(ceil_div(extra * tier_value(tiers, exposure, 2), 10000))
    initial += max(additions)
    initial += bid_total * max(0, bid * market["tick_size"] - price)
    initial += ask_total * max(0, price - ask * market["tick_size"]) if ask else 0
    pnl = size * price + quote
    if pnl > 0:
        pnl = ceil_div(pnl * market["upnl_factor"], 10000)
    pnl -= (market["funding"] - funding) * size
    return pnl, initial * market["maintenance_factor"] // 10000


def observe(portfolio, markets, target, slot, quote_decimals):
    positions = portfolio["positions"]
    position = positions[target]
    raw_size = unpack("q", position, 0)[0]
    size = raw_size * Decimal(10) ** -markets[target]["decimals"]
    result = {"size": size}
    base = portfolio["collateral"]
    for asset, row in positions.items():
        if not any(row):
            continue
        if asset == 0xFFFF0001:
            # Locked deposits restrict withdrawals, not effective collateral.
            continue
        if asset >= 0xFF000000:
            return {**result, "unknown": "unsupported_spot_collateral"}
        if asset not in markets:
            return {**result, "unknown": "missing_market_parameters"}
        market = markets[asset]
        components = [v for v in market["price_components"] if v["weight"]]
        # Exclude stale feeds rather than approximate the protocol's oracle
        # margin inflation. Use the captured venue thresholds, not a wall clock.
        if (
            not components
            or not market["mark_ticks"]
            or any(
                not 0 <= slot - market["mark_slot"] <= component["stale_threshold"]
                or any(
                    price and not 0 <= slot - moment <= component["stale_threshold"]
                    for moment, price in component["prices"]
                )
                or not any(price for _, price in component["prices"])
                for component in components
            )
        ):
            return {**result, "unknown": "stale_market_mark"}
        if asset != target:
            pnl, required = margin(row, market, market["mark_ticks"])
            base += pnl - required
    market = markets[target]
    sign = 1 if raw_size > 0 else -1
    buffers = []
    for band in (0, *BANDS):
        ticks = market["mark_ticks"] * (100 - sign * band)
        ticks = ticks // 100 if sign == 1 else ceil_div(ticks, 100)
        pnl, required = margin(position, market, ticks)
        buffers.append(Decimal(base + pnl - required) * Decimal(10) ** -quote_decimals)
    price = Decimal(market["mark_ticks"] * market["tick_size"]) * Decimal(10) ** (
        market["decimals"] - quote_decimals
    )
    return {**result, "price": price, "buffers": buffers}


def fetch_snapshot():
    if not available():
        return None
    snapshot, block_time, _ = fetch_position_snapshot("BTC", include_accounts=True)
    raw = snapshot.pop("accounts")
    accounts, markets, quote_decimals = portfolios(raw)
    target = snapshot["asset_id"]
    scale = Decimal(10) ** -snapshot["base_lot_decimals"]
    census = {key.hex(): size * scale for key, size in snapshot["positions"].items()}
    observations = {
        key.hex(): observe(
            accounts[key], markets, target, snapshot["slot"], quote_decimals
        )
        for key in snapshot["positions"]
    }
    received = get_current_time()
    summary = {
        **summarize(census, observations),
        "reference_timestamp": block_time,
        "reference_slot": snapshot["slot"],
        "observed_from": block_time,
        "available_at": received,
        "sampling": "all_positions",
        "coverage_basis": "atomic_program_accounts",
    }
    return {
        "version": VERSION,
        "available_at": received,
        "summary": summary,
        "target": target,
        "quote_decimals": quote_decimals,
        "portfolios": {
            key.hex(): {
                "collateral": accounts[key]["collateral"],
                "positions": {
                    asset: value.hex()
                    for asset, value in accounts[key]["positions"].items()
                },
            }
            for key in snapshot["positions"]
        },
        "markets": markets,
        "observations": observations,
    }
