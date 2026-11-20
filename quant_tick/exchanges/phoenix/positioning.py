"""Hourly trade-derived positioning, reconciled against the complete native feed.

Native OI retains its venue sampling convention. RPC enrichment describes fills
within the completed hour, not a snapshot of all open accounts. PnL events prove
opening/closing flow, including reversals through zero. Market trade sequence
numbers and account position continuity detect missing executions; every fill
must also match native timestamp and signed quantities, preserving multiplicity.
Transfers/ADL are aggregated separately from the complete position replay, not
inferred from fills. liquidated_* contains only market-liquidation executions.
When supplied, replayed_groups must have passed full reverse position and sequence
validation; this includes transfers between trades that fill-only checks cannot
reconcile. These groups replace individual paid transaction lookups.
Taker fees are the reported TradeSummary fees; supplementary fee and liquidation
charge events are not aggregated. position_size_flows values each taker position
before each individual fill at its quote/base execution price, splitting multi-fill
orders across tiers as the position changes. Its liquidation subset reuses those
same fills. Account concentration and gross realized PnL reuse the validated
TradeSummary/PnL events; they do not request additional transactions.
"""

from collections import Counter, defaultdict
from decimal import Decimal

import pandas as pd

from ..position_metrics import (
    PNL_FIELDS,
    add_realized_pnl,
    add_taker_fill,
    concentration_values,
    empty_flow_tiers,
    validate_flow_tiers,
    validate_realized_pnl,
)
from ..quicknode.solana import transaction
from .candles import phoenix_candles
from .events import NONTRADE_EVENTS, transaction_events
from .trades import get_trades

DECIMAL_FIELDS = (
    "long_open_size",
    "long_close_size",
    "short_open_size",
    "short_close_size",
    "taker_long_open_size",
    "taker_long_close_size",
    "taker_short_open_size",
    "taker_short_close_size",
    "taker_buy_size",
    "taker_sell_size",
    "taker_buy_value",
    "taker_sell_value",
    "liquidated_size",
    "liquidated_long_size",
    "liquidated_short_size",
    "liquidated_value",
    "liquidated_long_value",
    "liquidated_short_value",
    "taker_realized_pnl",
    "maker_realized_pnl",
    "taker_fees",
)
COUNT_FIELDS = (
    "trades",
    "liquidations",
    "long_account_delta",
    "short_account_delta",
    "long_position_flips",
    "short_position_flips",
)


def parsed_trades(groups, symbol):
    result = []
    for group in groups:
        header = context = None
        fills, pnls, trades = [], [], []
        for item in group:
            kind = item["kind"]
            if kind == "SlotContext":
                context = item
            elif kind == "Header":
                if fills or pnls:
                    raise ValueError("Phoenix position changes lack a trade summary")
                header = item
            elif kind == "PnL" and item["symbol"] == symbol:
                if item["base_lots_before"] != item["base_lots_after"]:
                    if header is None or header["symbol"] != symbol:
                        raise ValueError("Phoenix position change lacks market context")
                    pnls.append(item)
            elif header is not None and header["symbol"] == symbol:
                if kind in ("OrderFilled", "SplineFilled"):
                    fills.append(item)
                elif kind == "TradeSummary":
                    if context is None or not fills or not pnls:
                        raise ValueError(
                            "Phoenix trade summary lacks fills or positions"
                        )
                    trade = {
                        "header": header,
                        "context": context,
                        "summary": item,
                        "fills": fills,
                        "pnls": pnls,
                        "liquidated": False,
                    }
                    validate_trade(trade)
                    trades.append(trade)
                    fills, pnls = [], []
        if fills or pnls:
            raise ValueError("Phoenix position changes lack a trade summary")
        liquidations = [item for item in group if item["kind"] == "Liquidation"]
        enhanced = [item for item in group if item["kind"] == "LiquidationWithFee"]
        for item in liquidations or enhanced:
            matches = [
                trade
                for trade in trades
                if trade["header"]["asset_id"] == item["asset_id"]
                and trade["summary"]["trader"] == item["liquidated_trader"]
            ]
            if not matches:
                if any(
                    item["asset_id"] == trade["header"]["asset_id"] for trade in trades
                ):
                    raise ValueError("Phoenix liquidation lacks its trade")
                continue
            if (
                len(matches) != 1
                or matches[0]["liquidated"]
                or matches[0]["summary"]["base_lots_filled"] != item["base_lots_filled"]
            ):
                raise ValueError(
                    "Phoenix liquidation does not reconcile with its trade"
                )
            matches[0]["liquidated"] = True
        result.extend(trades)
    return result


def validate_trade(trade):
    summary, fills, pnls = trade["summary"], trade["fills"], trade["pnls"]
    takers = [pnl for pnl in pnls if pnl["trader"] == summary["trader"]]
    makers = [pnl for pnl in pnls if pnl["trader"] != summary["trader"]]
    sign = 1 if summary["side"] == "bid" else -1
    if (
        len(takers) != 1
        or len(makers) != len(fills)
        or sum(fill["base_lots_filled"] for fill in fills)
        != summary["base_lots_filled"]
        or sum(fill["quote_lots_filled"] for fill in fills)
        != summary["quote_lots_filled"]
        or sum(pnl["base_lots_after"] - pnl["base_lots_before"] for pnl in pnls) != 0
    ):
        raise ValueError("Phoenix fills and position changes do not reconcile")
    taker = takers[0]
    if (
        taker["base_lots_after"] != summary["base_lot_position"]
        or taker["base_lots_after"] - taker["base_lots_before"]
        != sign * summary["base_lots_filled"]
    ):
        raise ValueError("Phoenix taker position does not reconcile with fills")
    for fill, pnl in zip(fills, makers, strict=True):
        if (
            fill["side"] == summary["side"]
            or fill["maker"] != pnl["trader"]
            or fill["maker_base_lot_position"] != pnl["base_lots_after"]
            or pnl["base_lots_after"] - pnl["base_lots_before"]
            != -sign * fill["base_lots_filled"]
            or fill["base_lots_filled"] <= 0
            or fill["quote_lots_filled"] <= 0
            or fill["price"] <= 0
        ):
            raise ValueError("Phoenix maker position does not reconcile with fills")


def units(header):
    if (
        not -18 <= header["base_lot_decimals"] <= 18
        or not 0 <= header["quote_lot_decimals"] <= 18
        or header["tick_size"] <= 0
    ):
        raise ValueError("Invalid Phoenix market units")
    base = Decimal(10) ** -header["base_lot_decimals"]
    quote = Decimal(10) ** -header["quote_lot_decimals"]
    return base, quote, header["tick_size"] * quote / base


def validate_native(trades, native, start, end):
    expected, observed = Counter(), Counter()
    for trade in trades:
        base, quote, tick = units(trade["header"])
        timestamp = pd.Timestamp(trade["context"]["timestamp"], unit="s", tz="UTC")
        if not start <= timestamp < end:
            raise ValueError("Phoenix transaction is outside the requested hour")
        sign = 1 if trade["summary"]["side"] == "bid" else -1
        for fill in trade["fills"]:
            quantity = sign * fill["base_lots_filled"] * base
            value = -sign * fill["quote_lots_filled"] * quote
            # Spline fills can span prices; their quote/base ratio is the VWAP.
            if (
                fill["kind"] == "OrderFilled"
                and abs(abs(quantity) * fill["price"] * tick - abs(value)) > quote
            ):
                raise ValueError("Phoenix fill price does not reconcile with its lots")
            expected[timestamp, quantity, value] += 1
    for row in native:
        quantity, value, price = (
            Decimal(row[key]) for key in ("baseQty", "quoteQty", "price")
        )
        if (
            not all(number.is_finite() for number in (quantity, value, price))
            or quantity * value >= 0
            or price <= 0
        ):
            raise ValueError("Invalid Phoenix native fill quantities")
        # Native price strings can retain binary floating-point division noise.
        if abs(price - abs(value / quantity)) > price * Decimal("1e-12"):
            raise ValueError("Phoenix native fill price does not reconcile")
        observed[pd.Timestamp(row["timestamp"]), quantity, value] += 1
    if observed != expected:
        raise ValueError("Phoenix RPC fills do not match the native trade feed")


def add_position_change(row, before, after, *, taker=False):
    for direction, sign in (("long", 1), ("short", -1)):
        change = max(sign * after, 0) - max(sign * before, 0)
        action = "open" if change > 0 else "close"
        field = direction + "_" + action + "_size"
        row[field] += abs(change)
        if taker:
            row["taker_" + field] += abs(change)
        row[direction + "_account_delta"] += int(sign * after > 0) - int(
            sign * before > 0
        )
        row[direction + "_position_flips"] += int(sign * before < 0 < sign * after)


def nontrade_flows(groups, asset_id, base_lot_decimals):
    """Aggregate position-validated invocations from a complete hourly replay.

    Backstop and ADL quantities count each event once, on the distressed side.
    Shutdown closes are separate from ADL. No quote value is inferred from virtual
    quote lots or the hourly mark price. Position changes include both accounts;
    none of these events contributes to trade counts or taker flow.
    """
    base = Decimal(10) ** -base_lot_decimals
    row = {
        f"{kind}_{side}size": Decimal(0)
        for kind in ("backstop", "adl")
        for side in ("", "long_", "short_")
    }
    row.update({kind + "_events": 0 for kind in ("backstop", "adl", "shutdown")})
    row["shutdown_size"] = Decimal(0)
    for direction in ("long", "short"):
        row[direction + "_open_size"] = Decimal(0)
        row[direction + "_close_size"] = Decimal(0)
        row[direction + "_account_delta"] = 0
        row[direction + "_position_flips"] = 0
    for group in groups:
        selected = [item for item in group if item.get("asset_id") == asset_id]
        if any(item["kind"] == "TradeSummary" for item in group):
            raise ValueError(
                "Phoenix mixed trading and transfer invocation is unsupported"
            )
        for item in selected:
            kind = item["kind"]
            if kind == "PnL":
                add_position_change(
                    row, item["base_lots_before"] * base, item["base_lots_after"] * base
                )
            elif kind in NONTRADE_EVENTS:
                if kind == "LiquidationTransfer":
                    quantity = item["base_lots_transferred"]
                    category, side = "backstop", "long" if quantity > 0 else "short"
                elif kind == "CloseMatchedPositions":
                    quantity = item["base_lots_closed"]
                    if item["closed_long"] == item["closed_short"] or item[
                        "in_profit_account"
                    ] not in (item["closed_long"], item["closed_short"]):
                        raise ValueError(
                            "Phoenix ADL lacks its profitable counterparty"
                        )
                    category = "adl"
                    side = (
                        "long"
                        if item["in_profit_account"] == item["closed_short"]
                        else "short"
                    )
                else:
                    quantity = item["base_lots_closed"]
                    category, side = "shutdown", None
                if not quantity:
                    raise ValueError("Phoenix forced position event has zero size")
                row[category + "_events"] += 1
                row[category + "_size"] += abs(quantity) * base
                if side is not None:
                    row[category + "_" + side + "_size"] += abs(quantity) * base
    return row


def aggregate(trades, *, replayed=False):
    row = dict.fromkeys((*DECIMAL_FIELDS, *PNL_FIELDS), Decimal(0)) | dict.fromkeys(
        COUNT_FIELDS, 0
    )
    positions = {}
    tiers = empty_flow_tiers()
    participants = {side: defaultdict(Decimal) for side in ("buy", "sell")}
    previous = None
    for trade in trades:
        summary = trade["summary"]
        sequence = summary["trade_sequence_number"]
        if not replayed and previous is not None and sequence != previous + 1:
            raise ValueError("Phoenix market trade sequence has a gap or duplicate")
        previous = sequence
        base, quote, _ = units(trade["header"])
        bought = summary["side"] == "bid"
        side = "buy" if bought else "sell"
        size, value = (
            summary["base_lots_filled"] * base,
            summary["quote_lots_filled"] * quote,
        )
        participants[side][summary["trader"]] += size
        row["trades"] += len(trade["fills"])
        row["taker_" + side + "_size"] += size
        row["taker_" + side + "_value"] += value
        row["taker_fees"] += summary["fee_in_quote_lots"] * quote
        before = next(
            pnl["base_lots_before"] * base
            for pnl in trade["pnls"]
            if pnl["trader"] == summary["trader"]
        )
        for fill in trade["fills"]:
            after = before + (1 if bought else -1) * fill["base_lots_filled"] * base
            add_taker_fill(
                tiers,
                before,
                after,
                fill["quote_lots_filled"] * quote,
                liquidated=trade["liquidated"],
            )
            before = after
        for pnl in trade["pnls"]:
            before, after = (
                pnl["base_lots_before"] * base,
                pnl["base_lots_after"] * base,
            )
            account = pnl["trader"]
            if not replayed and account in positions and positions[account] != before:
                raise ValueError("Phoenix account position continuity is broken")
            positions[account] = after
            taker = account == summary["trader"]
            row["taker_realized_pnl" if taker else "maker_realized_pnl"] += (
                pnl["realized_pnl"] * quote
            )
            add_realized_pnl(
                row, before, after, pnl["realized_pnl"] * quote, taker=taker
            )
            add_position_change(row, before, after, taker=taker)
        if trade["liquidated"]:
            direction = "short" if bought else "long"
            row["liquidations"] += 1
            for suffix, amount in (("size", size), ("value", value)):
                row["liquidated_" + suffix] += amount
                row["liquidated_" + direction + "_" + suffix] += amount
    validate_flow_tiers(tiers, row)
    validate_realized_pnl(row)
    concentration = {}
    for side, accounts in participants.items():
        sizes = sorted(accounts.values(), reverse=True)
        concentration[side] = {
            "accounts": len(sizes),
            "size": sum(sizes, Decimal(0)),
            "top1_size": sizes[0] if sizes else Decimal(0),
            "top5_size": sum(sizes[:5], Decimal(0)),
        }
    row["taker_flow_concentration"] = concentration_values(
        len(participants["buy"].keys() | participants["sell"].keys()),
        concentration,
        row,
    )
    row["position_size_flows"] = tiers
    row["active_trader_accounts"] = len(positions)
    row["taker_long_short_volume_ratio"] = (
        row["taker_buy_size"] / row["taker_sell_size"]
        if row["taker_sell_size"]
        else None
    )
    row["fee_currency"] = "USDC"
    row["trade_position_flow_validated"] = True
    row["position_flow_source"] = "quicknode_phoenix_events"
    return row


def phoenix_positioning(symbol, start, end, *, replayed_groups=None):
    native, _, _ = get_trades(symbol, start, end)
    candles = phoenix_candles(symbol, start, end, resolution="1h")
    if len(candles) != 1 or candles.index[0] != start:
        raise ValueError("Phoenix positioning lacks its completed native candle")
    candle = candles.iloc[0]
    if candle["trades"] != len(native):
        raise ValueError("Phoenix native fill count does not reconcile with candle")
    for field, key in (("notional", "baseQty"), ("volume", "quoteQty")):
        expected = sum((abs(Decimal(row[key])) for row in native), Decimal(0))
        value = Decimal(str(candle[field]))
        if not value.is_finite() or abs(value - expected) > max(
            Decimal("1e-12"), expected * Decimal("1e-12")
        ):
            raise ValueError(
                "Phoenix native fill volume does not reconcile with candle"
            )
    by_signature = defaultdict(list)
    for row in native:
        by_signature[row["transactionSignature"]].append(row)
    if replayed_groups is not None and replayed_groups.keys() != by_signature.keys():
        raise ValueError(
            "Phoenix replay and native feed have different trade transactions"
        )
    trades = []
    for signature, rows in by_signature.items():
        if replayed_groups is None:
            groups = transaction_events(transaction(signature), signature)
        else:
            groups = []
            for group in replayed_groups[signature]:
                if any(item["kind"] in NONTRADE_EVENTS for item in group):
                    if any(
                        item["kind"] in ("TradeSummary", "OrderFilled", "SplineFilled")
                        for item in group
                    ):
                        raise ValueError(
                            "Phoenix mixed trading and transfer invocation is unsupported"
                        )
                else:
                    groups.append(group)
        parsed = parsed_trades(groups, symbol)
        validate_native(parsed, rows, start, end)
        trades.extend(parsed)
    # This is the venue's explicit execution sequence, not timestamp ordering.
    trades.sort(key=lambda trade: trade["summary"]["trade_sequence_number"])
    return aggregate(trades, replayed=replayed_groups is not None)
