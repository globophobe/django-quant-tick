"""Position changes and market executions between consecutive funding blocks.

Pairs are built from user fills: the derived trades table can omit complete
pairs present in both fills and native blocks. Each pair must contain one buyer
and one seller with identical execution details before contributing to totals.
All fills reconcile positions, including backstop transfers, ADL and vault
netting. Only order-book executions contribute to trades and taker/maker metrics.
liquidated_* describes market liquidations; backstop_* and adl_* are disjoint
categories, counted once per paired event. Their long/short side is the distressed
position, not the profitable counterparty closed by ADL. Values use event prices.
Unknown directions or inconsistent liquidation metadata cannot become valid rows.
"""

from .api import coin_literal, number, query, time_literal

CATEGORIES = ("liquidated", "backstop", "adl")
FLOW_FIELDS = (
    "long_open_size",
    "long_close_size",
    "short_open_size",
    "short_close_size",
    "taker_long_open_size",
    "taker_short_open_size",
    "taker_long_close_size",
    "taker_short_close_size",
    "taker_buy_size",
    "taker_sell_size",
    "taker_twap_buy_size",
    "taker_twap_sell_size",
    "taker_buy_value",
    "taker_sell_value",
    "netting_size",
    "netting_value",
    *(
        f"{kind}_{side}{unit}"
        for kind in CATEGORIES
        for side in ("", "long_", "short_")
        for unit in ("size", "value")
    ),
)
SIGNED_FLOW_FIELDS = (
    "taker_realized_pnl",
    "maker_realized_pnl",
    "taker_fees",
    "maker_fees",
    "nontrade_realized_pnl",
)
COUNT_FIELDS = (
    "position_events",
    "trades",
    "liquidations",
    "backstop_events",
    "adl_events",
    "netting_events",
    "long_position_flips",
    "short_position_flips",
)


def paired_fills(coin, first, last, start, end):
    market, lower, upper = coin_literal(coin), time_literal(start), time_literal(end)
    columns = [
        "block_number",
        "any(f.size) AS size",
        "any(f.price) AS price",
        "any(f.fee_token) AS fee_token",
        "any(f.liquidation_method) AS liquidation_method",
        "any(f.liquidated_user) AS liquidated_user",
        "if(anyIf(f.crossed,f.side='B'),'B','A') AS side",
        (
            "(count()=2 AND countIf(f.side='B')=1 AND countIf(f.side='A')=1 "
            "AND uniqExact(f.user)=2 AND sum(f.crossed)=1 "
            "AND min(f.size)>0 AND min(f.price)>0 "
            "AND uniqExact(tuple(f.size,f.price,f.time,f.fee_token,"
            "f.liquidation_method,f.liquidated_user))=1) AS valid_pair"
        ),
    ]
    for side, direction in (("buyer", "B"), ("seller", "A")):
        for field, alias in (
            ("user", "address"),
            ("start_position", "start_position"),
            ("dir", "dir"),
            ("twap_id", "twap_id"),
            ("closed_pnl", "closed_pnl"),
            ("fee", "fee"),
        ):
            # A tuple preserves NULL twap IDs through conditional aggregation.
            columns.append(
                f"anyIf(tuple(f.{field}),f.side='{direction}').1 AS {side}_{alias}"
            )
    return (
        "SELECT " + ", ".join(columns) + " FROM hyperliquid_fills AS f FINAL "
        f"WHERE coin={market} AND block_number>={first} AND block_number<{last} "
        f"AND block_time>={lower} AND block_time<{upper} "
        "GROUP BY block_number,tid"
    )


def position_flows(coin, points, start, end):
    blocks = [p["block_number"] for p in points.values()]
    if len(blocks) < 2:
        return {}
    first, last = blocks[0], blocks[-1]
    pairs = paired_fills(coin, first, last, start, end)
    columns = [
        f"arrayCount(b->block_number>=b,{blocks})-1 AS interval_index",
        "count() AS position_events",
        "countIf(is_market) AS trades",
        "countIf(is_market_liquidation) AS liquidations",
        "countIf(is_backstop) AS backstop_events",
        "countIf(is_adl) AS adl_events",
        "countIf(is_netting) AS netting_events",
        "sum(lo) AS long_open_size",
        "sum(so) AS short_open_size",
        "sum(lc) AS long_close_size",
        "sum(sc) AS short_close_size",
        "sumIf(lo,is_market AND side='B') AS taker_long_open_size",
        "sumIf(so,is_market AND side='A') AS taker_short_open_size",
        "sumIf(lc,is_market AND side='A') AS taker_long_close_size",
        "sumIf(sc,is_market AND side='B') AS taker_short_close_size",
        "sumIf(size,is_market AND side='B') AS taker_buy_size",
        "sumIf(size,is_market AND side='A') AS taker_sell_size",
        "sumIf(size,is_market AND side='B' AND isNotNull(buyer_twap_id)) AS taker_twap_buy_size",
        "sumIf(size,is_market AND side='A' AND isNotNull(seller_twap_id)) AS taker_twap_sell_size",
        "toString(sumIf(trade_value,is_market AND side='B')) AS taker_buy_value",
        "toString(sumIf(trade_value,is_market AND side='A')) AS taker_sell_value",
        "sumIf(size,is_netting) AS netting_size",
        "toString(sumIf(trade_value,is_netting)) AS netting_value",
        "toString(sumIf(if(side='B',buyer_closed_pnl,seller_closed_pnl),is_market)) AS taker_realized_pnl",
        "toString(sumIf(if(side='B',seller_closed_pnl,buyer_closed_pnl),is_market)) AS maker_realized_pnl",
        "toString(sumIf(if(side='B',buyer_fee,seller_fee),is_market)) AS taker_fees",
        "toString(sumIf(if(side='B',seller_fee,buyer_fee),is_market)) AS maker_fees",
        "toString(sumIf(buyer_closed_pnl+seller_closed_pnl,NOT is_market)) AS nontrade_realized_pnl",
        "countIf(buyer_start_position<0 AND buyer_start_position+size>0) AS long_position_flips",
        "countIf(seller_start_position>0 AND seller_start_position-size<0) AS short_position_flips",
        "groupUniqArray(fee_token) AS fee_tokens",
        (
            "sum(toInt64(buyer_start_position+size>0)-toInt64(buyer_start_position>0)"
            "+toInt64(seller_start_position-size>0)-toInt64(seller_start_position>0)) AS long_account_delta"
        ),
        (
            "sum(toInt64(buyer_start_position+size<0)-toInt64(buyer_start_position<0)"
            "+toInt64(seller_start_position-size<0)-toInt64(seller_start_position<0)) AS short_account_delta"
        ),
        "countIf(NOT ifNull(valid_event,0)) AS invalid_events",
    ]
    for kind, condition in (
        ("liquidated", "is_market_liquidation"),
        ("backstop", "is_backstop"),
        ("adl", "is_adl"),
    ):
        for side, direction in (
            ("", "1"),
            ("long_", "distressed_long"),
            ("short_", "distressed_short"),
        ):
            predicate = f"{condition} AND {direction}"
            columns.extend(
                (
                    f"sumIf(size,{predicate}) AS {kind}_{side}size",
                    f"toString(sumIf(trade_value,{predicate})) AS {kind}_{side}value",
                )
            )
    rows = query(
        "WITH toDecimal128(0,18) AS zero, "
        "toDecimal256(size,18)*toDecimal256(price,18) AS trade_value, "
        "greatest(size+least(buyer_start_position,zero),zero) AS lo, "
        "least(size,greatest(-buyer_start_position,zero)) AS sc, "
        "greatest(size-greatest(seller_start_position,zero),zero) AS so, "
        "least(size,greatest(seller_start_position,zero)) AS lc, "
        "ifNull(liquidation_method,'') AS method, "
        "notEmpty(ifNull(liquidated_user,'')) AS tagged, "
        "(buyer_dir='Auto-Deleveraging' OR seller_dir='Auto-Deleveraging') AS is_adl, "
        "(method='backstop' AND NOT is_adl) AS is_backstop, "
        "(buyer_dir='Net Child Vaults' AND seller_dir='Net Child Vaults') AS is_netting, "
        "NOT (is_adl OR is_backstop OR is_netting) AS is_market, "
        "(method='market' AND is_market) AS is_market_liquidation, "
        "if(is_adl,buyer_dir='Auto-Deleveraging',ifNull(liquidated_user=seller_address,0)) AS distressed_long, "
        "if(is_adl,seller_dir='Auto-Deleveraging',ifNull(liquidated_user=buyer_address,0)) AS distressed_short, "
        "(valid_pair AND method IN ('','market','backstop') AND tagged=(method!='') "
        "AND (NOT tagged OR (buyer_address!=seller_address "
        "AND ((distressed_long AND liquidated_user=seller_address) "
        "OR (distressed_short AND liquidated_user=buyer_address)))) "
        "AND ((is_adl AND buyer_dir!=seller_dir AND method IN ('','backstop') "
        "AND buyer_start_position<=-size AND seller_start_position>=size) "
        "OR (is_backstop AND ((distressed_long AND seller_start_position>=size) "
        "OR (distressed_short AND buyer_start_position<=-size))) "
        "OR (is_netting AND method='') "
        "OR (is_market AND buyer_dir IN ('Open Long','Close Short','Short > Long') "
        "AND seller_dir IN ('Open Short','Close Long','Long > Short')))) AS valid_event "
        "SELECT " + ", ".join(columns) + f" FROM ({pairs}) "
        "GROUP BY interval_index ORDER BY interval_index LIMIT 100",
        # Both counterparty fills are scanned instead of one derived trade row.
        scan_limit=4294967296,
    )
    result = {}
    for row in rows:
        if int(row.pop("invalid_events")):
            raise ValueError(
                "QuickNode returned incomplete, unclassified or inconsistent position events"
            )
        index = int(row.pop("interval_index"))
        if (
            index in result
            or not 0 <= index < len(blocks) - 1
            or int(row["position_events"]) <= 0
        ):
            raise ValueError("QuickNode returned invalid position flow intervals")
        for key in FLOW_FIELDS:
            row[key] = number(row[key])
            if row[key] < 0:
                raise ValueError("QuickNode returned negative position flows")
        for key in SIGNED_FLOW_FIELDS:
            row[key] = number(row[key])
        for key in COUNT_FIELDS:
            row[key] = int(row[key])
            if not 0 <= row[key] <= int(row["position_events"]):
                raise ValueError("QuickNode returned invalid position event counts")
        tokens = row.pop("fee_tokens")
        if len(tokens) != 1 or not tokens[0]:
            raise ValueError("QuickNode fee currencies cannot be combined")
        row["fee_currency"] = tokens[0]
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
    total = (
        bought + sold + sum(flow[k + "_size"] for k in ("backstop", "adl", "netting"))
    )
    if (
        flow["long_open_size"] + flow["short_close_size"] != total
        or flow["short_open_size"] + flow["long_close_size"] != total
        or flow["taker_long_open_size"] + flow["taker_short_close_size"] != bought
        or flow["taker_short_open_size"] + flow["taker_long_close_size"] != sold
        or flow["taker_twap_buy_size"] > bought
        or flow["taker_twap_sell_size"] > sold
        or flow["liquidated_size"] > bought + sold
        or flow["liquidations"] > flow["trades"]
        or flow["position_events"]
        != sum(
            flow[k]
            for k in ("trades", "backstop_events", "adl_events", "netting_events")
        )
        or any(
            flow[k + "_long_" + unit] + flow[k + "_short_" + unit]
            != flow[k + "_" + unit]
            for k in CATEGORIES
            for unit in ("size", "value")
        )
    ):
        raise ValueError("QuickNode position flow components do not reconcile")
