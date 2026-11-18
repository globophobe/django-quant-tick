"""Atomic position snapshots from Phoenix account layouts (rise-public f67e8b4).

The GTI identifies hot traders whose authoritative positions live in the ATB;
their trader accounts can contain stale copies. Cold positions come from trader
accounts. Tree structure, account identities and both sides of native OI must
reconcile. Additional GTI/ATB arenas require a decoder extension and fail closed.
"""

import base64
import hashlib
import struct
from decimal import Decimal

import pyarrow as pa

from .events import PROGRAM, decode58

ACCOUNT_TYPES = (
    "global_configuration",
    "perp_asset_map",
    "global_trader_index",
    "active_trader_buffer",
    "trader",
)
TAGS = {
    hashlib.sha256(("account:" + name).encode()).digest()[:8]: name
    for name in ACCOUNT_TYPES
}


def unpack(fmt, data, offset):
    try:
        return struct.unpack_from("<" + fmt, data, offset)
    except struct.error as exc:
        raise ValueError("Truncated Phoenix account") from exc


def account_bytes(account):
    if account.get("owner") != PROGRAM or account.get("executable") is not False:
        raise ValueError("Unexpected Phoenix account owner or executable flag")
    size = account["space"]
    if type(size) is not int or not 8 <= size <= 20_000_000:
        raise ValueError("Invalid Phoenix account size")
    encoded, encoding = account["data"]
    data = base64.b64decode(encoded, validate=True)
    if encoding == "base64+zstd":
        data = pa.decompress(data, decompressed_size=size, codec="zstd").to_pybytes()
    elif encoding != "base64":
        raise ValueError("Unexpected Phoenix account encoding")
    if len(data) != size:
        raise ValueError("Phoenix account size mismatch")
    return data


def tree(data, key_format, value_size):
    """Read live nodes only, validating allocator bounds and ordered pointers."""
    size, arenas, active, capacity, bump, free = unpack("IHHIII", data, 48)
    key_size = struct.calcsize("<" + key_format)
    node_size = 16 + key_size + value_size
    if arenas != 1:
        raise ValueError("Phoenix snapshot requires support for additional arenas")
    if (
        not capacity
        or active > arenas
        or (len(data) - 96) % node_size
        or not 0 <= size < bump <= (len(data) - 96) // node_size + 1 <= capacity + 1
        or free > bump
    ):
        raise ValueError("Invalid Phoenix account tree allocator")
    (root,) = unpack("I", data, 80)
    pending = [(root, 0, None, None)] if root else []
    nodes = {}
    while pending:
        index, parent, lower, upper = pending.pop()
        if index in nodes or not 0 < index < bump or len(nodes) >= size:
            raise ValueError("Invalid Phoenix account tree pointer")
        offset = 96 + (index - 1) * node_size
        left, right, actual_parent, color = unpack("IIII", data, offset)
        key = unpack(key_format, data, offset + 16)
        if (
            actual_parent != parent
            or color > 1
            or (lower is not None and key <= lower)
            or (upper is not None and key >= upper)
        ):
            raise ValueError("Invalid Phoenix account tree ordering")
        nodes[index] = (key, data[offset + 16 + key_size : offset + node_size])
        if right:
            pending.append((right, index, key, upper))
        if left:
            pending.append((left, index, lower, key))
    if len(nodes) != size:
        raise ValueError("Incomplete Phoenix account tree")
    return nodes


def market(data, symbol, quote_decimals):
    used, tombstones, capacity = unpack("IIQ", data, 32)
    if not 0 <= tombstones <= used <= capacity == 1024 or len(data) != 1622064:
        raise ValueError("Invalid Phoenix asset map")
    matches = []
    for index in range(used):
        offset = 48 + index * 1584
        name = data[offset : offset + 16].rstrip(b"\0").decode("ascii")
        metadata = data[offset + 16 : offset + 1584]
        if name != symbol or metadata[1450] or metadata[-1] != 0:
            continue
        tick_size, asset_lower, decimals, _, asset_upper = unpack(
            "QHbBH", metadata, 936
        )
        (price_ticks,) = unpack("Q", metadata, 24)
        (oi,) = unpack("Q", metadata, 1424)
        if not tick_size or not price_ticks or not -18 <= decimals <= 18:
            raise ValueError("Invalid Phoenix market price or units")
        matches.append(
            {
                "asset_id": asset_lower | asset_upper << 16,
                "base_lot_decimals": decimals,
                "open_interest_lots": oi,
                "price": Decimal(price_ticks)
                * tick_size
                * Decimal(10) ** (decimals - quote_decimals),
            }
        )
    if len(matches) != 1:
        raise ValueError("Phoenix snapshot does not contain exactly one market")
    return matches[0]


def decode_snapshot(result, symbol):
    slot = result["context"]["slot"]
    if type(slot) is not int or slot <= 0:
        raise ValueError("Invalid Phoenix snapshot slot")
    globals_by_kind, traders, seen = {}, {}, set()
    for row in result["value"]:
        key = row["pubkey"]
        if key in seen:
            raise ValueError("Duplicate Phoenix snapshot account")
        seen.add(key)
        data = account_bytes(row["account"])
        kind = TAGS.get(data[:8])
        if kind == "trader":
            count, capacity = unpack("QQ", data, 224)
            if count > capacity or len(data) != 240 + capacity * 40:
                raise ValueError("Invalid Phoenix trader position map")
            if count or unpack("I", data, 96)[0] & 1:
                if data[24:56] != decode58(key):
                    raise ValueError("Phoenix trader identity mismatch")
                traders[key] = data
        elif kind:
            if kind in globals_by_kind:
                raise ValueError("Duplicate Phoenix snapshot index")
            globals_by_kind[kind] = (key, data)
    if set(globals_by_kind) != set(ACCOUNT_TYPES) - {"trader"}:
        raise ValueError("Incomplete Phoenix snapshot indexes")
    config_key, config = globals_by_kind["global_configuration"]
    if len(config) < 1120 or config[8:40] != decode58(config_key):
        raise ValueError("Phoenix configuration identity mismatch")
    for kind, offset in (
        ("perp_asset_map", 360),
        ("global_trader_index", 392),
        ("active_trader_buffer", 424),
    ):
        if config[offset : offset + 32] != decode58(globals_by_kind[kind][0]):
            raise ValueError("Phoenix index identity mismatch")
    quote_decimals = config[505]
    if quote_decimals > 18:
        raise ValueError("Invalid Phoenix quote decimals")
    metadata = market(globals_by_kind["perp_asset_map"][1], symbol, quote_decimals)
    gti = tree(globals_by_kind["global_trader_index"][1], "32s", 16)
    atb = tree(globals_by_kind["active_trader_buffer"][1], "II", 64)
    hot_keys, positions = set(), {}
    for (key,), state in gti.values():
        if not unpack("I", state, 8)[0] & 1:
            raise ValueError("Phoenix GTI contains a cold trader")
        hot_keys.add(key)
    for (trader_index, asset), position in atb.values():
        if trader_index not in gti:
            raise ValueError("Orphan Phoenix active position")
        if asset == metadata["asset_id"]:
            (size,) = unpack("q", position, 0)
            if size:
                positions[gti[trader_index][0][0]] = size
    for key, data in traders.items():
        address = decode58(key)
        if address in hot_keys:
            continue
        if unpack("I", data, 96)[0] & 1:
            raise ValueError("Phoenix hot trader is missing from GTI")
        (count,) = unpack("Q", data, 224)
        assets = set()
        for index in range(count):
            raw_asset, size = unpack("Qq", data, 240 + index * 40)
            asset = raw_asset & 0xFFFFFFFF
            if asset in assets:
                raise ValueError("Duplicate Phoenix trader asset")
            assets.add(asset)
            if asset == metadata["asset_id"] and size:
                positions[address] = size
    long_lots = sum(max(size, 0) for size in positions.values())
    short_lots = sum(max(-size, 0) for size in positions.values())
    if long_lots != short_lots or long_lots != metadata["open_interest_lots"]:
        raise ValueError("Phoenix snapshot positions do not reconcile with native OI")
    return {
        "slot": slot,
        "positions": positions,
        "history_address": globals_by_kind["active_trader_buffer"][0],
        **metadata,
    }


def cohort_values(snapshot):
    scale = Decimal(10) ** -snapshot["base_lot_decimals"]
    price = snapshot["price"]
    oi = snapshot["open_interest_lots"] * scale
    tiers = [
        {
            "tier": tier,
            "longSz": Decimal(0),
            "shortSz": Decimal(0),
            "longAccounts": 0,
            "shortAccounts": 0,
        }
        for tier in range(1, 6)
    ]
    result = {
        "cohort_slot": snapshot["slot"],
        "cohort_price": price,
        "cohort_price_source": "onchain_mark",
        "cohort_open_interest": oi * 2,
        "cohort_open_interest_basis": "sum_absolute_positions",
        "cohort_single_open_interest": oi,
        "cohort_source": "solana_program_accounts",
        "cohort_account_basis": "trader_subaccount",
        "cohort_snapshot_validated": True,
        "cohorts": tiers,
    }
    for side, sign in (("long", 1), ("short", -1)):
        sizes = sorted(
            (
                size * sign * scale
                for size in snapshot["positions"].values()
                if size * sign > 0
            ),
            reverse=True,
        )
        for size in sizes:
            tier = sum(
                size * price >= bound for bound in (10000, 100000, 1000000, 10000000)
            )
            tiers[tier][side + "Sz"] += size
            tiers[tier][side + "Accounts"] += 1
        result[side + "_accounts"] = len(sizes)
        result["cohort_top10_" + side + "_size"] = sum(sizes[:10], Decimal(0))
        result["cohort_largest_" + side + "_share"] = sizes[0] / oi if oi else None
        result["cohort_top_decile_" + side + "_share"] = (
            sum(sizes[: (len(sizes) + 9) // 10], Decimal(0)) / oi if oi else None
        )
        result["large_" + side + "_size"] = sum(
            (tier[side + "Sz"] for tier in tiers if tier["tier"] >= 4), Decimal(0)
        )
    longs, shorts = result["long_accounts"], result["short_accounts"]
    total = longs + shorts
    result.update(
        long_account_ratio=Decimal(longs) / total if total else None,
        short_account_ratio=Decimal(shorts) / total if total else None,
        long_short_account_ratio=Decimal(longs) / shorts if shorts else None,
    )
    return result
