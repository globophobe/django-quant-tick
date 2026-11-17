"""Flip-aware position flows between consecutive funding blocks.

A buy first closes an existing short, then opens a long; a sell does the reverse.
These sums do not depend on intra-block execution order. A liquidation is counted
once per paired trade, on the liquidated account's side.
"""

from .api import coin_literal, number, query, time_literal

FLOW_FIELDS = (
    "long_open_size",
    "long_close_size",
    "short_open_size",
    "short_close_size",
    "taker_long_open_size",
    "taker_short_open_size",
    "taker_long_close_size",
    "taker_short_close_size",
    "liquidated_size",
    "liquidated_long_size",
    "liquidated_short_size",
    "taker_buy_size",
    "taker_sell_size",
    "taker_twap_buy_size",
    "taker_twap_sell_size",
)


def position_flows(coin, points, start, end):
    blocks = [p["block_number"] for p in points.values()]
    if len(blocks) < 2:
        return {}
    first, last = blocks[0], blocks[-1]
    market, lower, upper = coin_literal(coin), time_literal(start), time_literal(end)
    rows = query(
        "WITH toDecimal128(0,18) AS zero, "
        "greatest(size+least(buyer_start_position,zero),zero) AS lo, "
        "least(size,greatest(-buyer_start_position,zero)) AS sc, "
        "greatest(size-greatest(seller_start_position,zero),zero) AS so, "
        "least(size,greatest(seller_start_position,zero)) AS lc "
        f"SELECT arrayCount(b->block_number>=b,{blocks})-1 AS interval_index, "
        "count() AS trades, sum(lo) AS long_open_size, sum(so) AS short_open_size, "
        "sum(lc) AS long_close_size, sum(sc) AS short_close_size, "
        "sumIf(lo,side='B') AS taker_long_open_size, sumIf(so,side='A') AS taker_short_open_size, "
        "sumIf(lc,side='A') AS taker_long_close_size, sumIf(sc,side='B') AS taker_short_close_size, "
        "sumIf(size,notEmpty(ifNull(liquidated_user,''))) AS liquidated_size, "
        "sumIf(size,liquidated_user=seller_address) AS liquidated_long_size, "
        "sumIf(size,liquidated_user=buyer_address) AS liquidated_short_size, "
        "sumIf(size,side='B') AS taker_buy_size, sumIf(size,side='A') AS taker_sell_size, "
        "sumIf(size,side='B' AND isNotNull(buyer_twap_id)) AS taker_twap_buy_size, "
        "sumIf(size,side='A' AND isNotNull(seller_twap_id)) AS taker_twap_sell_size, "
        "sum(toInt64(buyer_start_position+size>0)-toInt64(buyer_start_position>0)"
        "+toInt64(seller_start_position-size>0)-toInt64(seller_start_position>0)) AS long_account_delta, "
        "sum(toInt64(buyer_start_position+size<0)-toInt64(buyer_start_position<0)"
        "+toInt64(seller_start_position-size<0)-toInt64(seller_start_position<0)) AS short_account_delta "
        "FROM hyperliquid_trades FINAL "
        f"WHERE coin={market} AND block_number>={first} AND block_number<{last} "
        f"AND block_time>={lower} AND block_time<{upper} "
        "GROUP BY interval_index ORDER BY interval_index LIMIT 100",
        scan_limit=2147483648,
    )
    result = {}
    for row in rows:
        index = int(row.pop("interval_index"))
        if (
            index in result
            or not 0 <= index < len(blocks) - 1
            or int(row["trades"]) <= 0
        ):
            raise ValueError("QuickNode returned invalid position flow intervals")
        for key in FLOW_FIELDS:
            row[key] = number(row[key])
            if row[key] < 0:
                raise ValueError("QuickNode returned negative position flows")
        result[index] = row
    return result


def validate_flow(before, after, flow):
    for side in ("long", "short"):
        if before[side + "_size"] + flow[side + "_open_size"] - flow[
            side + "_close_size"
        ] != after[side + "_size"] or int(before[side + "_accounts"]) + int(
            flow[side + "_account_delta"]
        ) != int(after[side + "_accounts"]):
            raise ValueError("QuickNode trades do not reconcile with funding positions")
    bought, sold = flow["taker_buy_size"], flow["taker_sell_size"]
    if (
        flow["long_open_size"] + flow["short_close_size"] != bought + sold
        or flow["short_open_size"] + flow["long_close_size"] != bought + sold
        or flow["taker_long_open_size"] + flow["taker_short_close_size"] != bought
        or flow["taker_short_open_size"] + flow["taker_long_close_size"] != sold
        or flow["liquidated_long_size"] + flow["liquidated_short_size"]
        != flow["liquidated_size"]
        or flow["taker_twap_buy_size"] > bought
        or flow["taker_twap_sell_size"] > sold
    ):
        raise ValueError("QuickNode position flow components do not reconcile")
