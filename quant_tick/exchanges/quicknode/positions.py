"""Hourly positions from funding settlements, before trades in the same block.

Every position receives a funding record. Long and short quantities must balance,
and users must be unique. Dollar cohorts use the last executed trade before the
settlement, rather than deriving a price from rounded funding payments.
"""

from collections import defaultdict
from decimal import Decimal

import pandas as pd

from .api import coin_literal, number, query, time_literal, timestamp
from .trades import TRADE_COLUMNS, ordered_blocks


def funding_points(coin, start, end, bounds):
    market, lower, upper = coin_literal(coin), time_literal(start), time_literal(end)
    block_filter = ""
    if bounds is not None:
        first, last = bounds
        block_filter = f"AND block_number BETWEEN {first} AND {last} "
    rows = query(
        "SELECT block_number, min(time) AS funding_time, max(time) AS last_time, "
        "count() AS records, uniqExact(user) AS users, "
        "countIf(szi>0) AS long_accounts, countIf(szi<0) AS short_accounts, "
        "sumIf(szi,szi>0) AS long_size, -sumIf(szi,szi<0) AS short_size, "
        "min(funding_rate) AS rate, max(funding_rate) AS last_rate, "
        "arraySum(arraySlice(arrayReverseSort(groupArrayIf(szi,szi>0)),1,10)) AS top10_long, "
        "arraySum(arraySlice(arrayReverseSort(groupArrayIf(-szi,szi<0)),1,10)) AS top10_short "
        "FROM hyperliquid_funding FINAL "
        f"WHERE coin={market} {block_filter}AND block_time >= {lower} "
        f"AND block_time < {upper} AND time >= {lower} AND time < {upper} "
        "GROUP BY block_number ORDER BY block_number LIMIT 100",
        scan_limit=2147483648,
    )
    points = {}
    previous = None
    for row in rows:
        moment = timestamp(row["funding_time"])
        hour = moment.floor("h")
        block = int(row["block_number"])
        for key in (
            "long_size",
            "short_size",
            "rate",
            "last_rate",
            "top10_long",
            "top10_short",
        ):
            row[key] = number(row[key])
        if (
            not start <= moment < end
            or moment != timestamp(row["last_time"])
            or moment - hour >= pd.Timedelta(minutes=1)
            or hour in points
            or (previous is not None and block <= previous)
            or int(row["records"]) != int(row["users"])
            or int(row["records"])
            != int(row["long_accounts"]) + int(row["short_accounts"])
            or row["long_size"] != row["short_size"]
            or row["long_size"] <= 0
            or int(row["long_accounts"]) <= 0
            or int(row["short_accounts"]) <= 0
            or row["rate"] != row["last_rate"]
            or not 0 < row["top10_long"] <= row["long_size"]
            or not 0 < row["top10_short"] <= row["short_size"]
        ):
            raise ValueError("QuickNode funding positions do not reconcile")
        previous = block
        points[hour] = {**row, "block_number": block, "time": moment}
    if len(rows) == 100:
        raise ValueError("QuickNode funding snapshot limit reached")
    return points


def preceding_prices(coin, points, start, end, bounds):
    if not points:
        return {}
    targets = [p["block_number"] for p in points.values()]
    market, lower, upper = coin_literal(coin), time_literal(start), time_literal(end)
    block_filter = ""
    if bounds is not None:
        first, last = bounds
        block_filter = f"AND block_number BETWEEN {first} AND {last} "
    rows = query(
        "SELECT target, maxIf(block_number,block_number<target) AS last_block "
        f"FROM hyperliquid_trades FINAL ARRAY JOIN {targets} AS target "
        f"WHERE coin={market} {block_filter}AND block_time >= {lower} "
        f"AND block_time < {upper} GROUP BY target ORDER BY target LIMIT 100",
        scan_limit=2147483648,
    )
    found = {}
    for row in rows:
        target, block = int(row["target"]), int(row["last_block"])
        if target not in targets or target in found or not 0 <= block < target:
            raise ValueError("QuickNode returned invalid preceding trade blocks")
        found[target] = block
    if rows and set(found) != set(targets):
        raise ValueError("QuickNode omitted a preceding price boundary")
    blocks = sorted({block for block in found.values() if block})
    if not blocks:
        return {}
    block_list = ",".join(map(str, blocks))
    rows = query(
        f"SELECT {TRADE_COLUMNS} FROM hyperliquid_trades FINAL "
        f"WHERE coin={market} AND block_number IN ({block_list}) "
        f"AND block_time >= {lower} AND block_time < {upper} "
        "ORDER BY block_number,trade_id LIMIT 100001"
    )
    if len(rows) >= 100001:
        raise ValueError("QuickNode preceding trade block limit reached")
    groups = defaultdict(list)
    seen = set()
    for row in rows:
        block, trade = int(row["block_number"]), int(row["trade_id"])
        if block not in blocks or (block, trade) in seen:
            raise ValueError("QuickNode returned unexpected preceding trades")
        seen.add((block, trade))
        row = {
            **row,
            "block_number": block,
            "trade_id": trade,
            "timestamp": timestamp(row["timestamp"]),
        }
        for key in ("price", "size", "buyer_start_position", "seller_start_position"):
            row[key] = number(row[key], positive=key in ("price", "size"))
        groups[block].append(row)
    if set(groups) != set(blocks):
        raise ValueError("QuickNode omitted a preceding trade block")
    prices, ambiguous = {}, []
    for block, group in groups.items():
        unique = {r["price"] for r in group}
        if len(unique) == 1:
            prices[block] = unique.pop()
        else:
            ambiguous.append(group)
    for row in ordered_blocks(coin, ambiguous):
        prices[row["block_number"]] = row["price"]
    return {target: prices[block] for target, block in found.items() if block}


def cohort_tiers(coin, points, prices, start, end):
    available = [
        (p["block_number"], prices[p["block_number"]])
        for p in points.values()
        if p["block_number"] in prices
    ]
    if not available:
        return {}
    entries = ",".join(
        f"{block},toDecimal128('{price}',8)" for block, price in available
    )
    block_list = ",".join(str(block) for block, _ in available)
    market, lower, upper = coin_literal(coin), time_literal(start), time_literal(end)
    rows = query(
        f"WITH map({entries}) AS prices, abs(szi)*prices[block_number] AS usd "
        "SELECT block_number, multiIf(usd<10000,1,usd<100000,2,usd<1000000,3,usd<10000000,4,5) AS tier, "
        "sumIf(szi,szi>0) AS long_size, -sumIf(szi,szi<0) AS short_size, "
        "countIf(szi>0) AS long_accounts, countIf(szi<0) AS short_accounts "
        "FROM hyperliquid_funding FINAL "
        f"WHERE coin={market} AND block_number IN ({block_list}) "
        f"AND block_time >= {lower} AND block_time < {upper} "
        "GROUP BY block_number,tier ORDER BY block_number,tier LIMIT 501",
        scan_limit=2147483648,
    )
    if len(rows) >= 501:
        raise ValueError("QuickNode cohort limit reached")
    result = defaultdict(list)
    seen = set()
    for row in rows:
        block, tier = int(row["block_number"]), int(row["tier"])
        if block not in prices or not 1 <= tier <= 5 or (block, tier) in seen:
            raise ValueError("QuickNode returned invalid cohort tiers")
        seen.add((block, tier))
        result[block].append(
            {
                "tier": tier,
                "longSz": number(row["long_size"]),
                "shortSz": number(row["short_size"]),
                "longAccounts": int(row["long_accounts"]),
                "shortAccounts": int(row["short_accounts"]),
            }
        )
    return result


def cohort_values(point, price, tiers):
    for side in ("long", "short"):
        if sum((c[side + "Sz"] for c in tiers), Decimal(0)) != point[
            side + "_size"
        ] or sum(c[side + "Accounts"] for c in tiers) != int(point[side + "_accounts"]):
            raise ValueError("QuickNode position cohorts do not reconcile")
    longs, shorts = Decimal(point["long_accounts"]), Decimal(point["short_accounts"])
    total = longs + shorts
    return {
        "long_accounts": longs,
        "short_accounts": shorts,
        "long_account_ratio": longs / total,
        "short_account_ratio": shorts / total,
        "long_short_account_ratio": longs / shorts,
        "cohort_timestamp": point["time"],
        "cohort_block": point["block_number"],
        "cohort_price": price,
        "cohort_price_source": "last_trade_before_funding",
        "cohort_open_interest": point["long_size"] * 2,
        "cohort_top10_long_size": point["top10_long"],
        "cohort_top10_short_size": point["top10_short"],
        "cohorts": tiers,
        **{
            "large_" + side + "_size": sum(
                (c[side + "Sz"] for c in tiers if c["tier"] >= 4), Decimal(0)
            )
            for side in ("long", "short")
        },
    }
