"""Hourly trade-derived positioning, reconciled against the complete native feed.

Native OI retains its venue sampling convention. RPC enrichment describes fills
within the completed hour, not a snapshot of all open accounts. PnL events prove
opening/closing flow, including reversals through zero. Market trade sequence
numbers and account position continuity detect missing executions; every fill
must also match native timestamp and signed quantities, preserving multiplicity.
Non-trading transfers/ADL and global account cohorts are not inferred from fills.
Taker fees are the reported TradeSummary fees; supplementary fee and liquidation
charge events are not aggregated.
"""

from collections import Counter, defaultdict
from decimal import Decimal

import pandas as pd

from ..quicknode.solana import transaction
from .candles import phoenix_candles
from .events import transaction_events
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


def aggregate(trades):
    row = dict.fromkeys(DECIMAL_FIELDS, Decimal(0)) | dict.fromkeys(COUNT_FIELDS, 0)
    positions = {}
    previous = None
    for trade in trades:
        summary = trade["summary"]
        sequence = summary["trade_sequence_number"]
        if previous is not None and sequence != previous + 1:
            raise ValueError("Phoenix market trade sequence has a gap or duplicate")
        previous = sequence
        base, quote, _ = units(trade["header"])
        bought = summary["side"] == "bid"
        side = "buy" if bought else "sell"
        size, value = (
            summary["base_lots_filled"] * base,
            summary["quote_lots_filled"] * quote,
        )
        row["trades"] += len(trade["fills"])
        row["taker_" + side + "_size"] += size
        row["taker_" + side + "_value"] += value
        row["taker_fees"] += summary["fee_in_quote_lots"] * quote
        for pnl in trade["pnls"]:
            before, after = (
                pnl["base_lots_before"] * base,
                pnl["base_lots_after"] * base,
            )
            account = pnl["trader"]
            if account in positions and positions[account] != before:
                raise ValueError("Phoenix account position continuity is broken")
            positions[account] = after
            taker = account == summary["trader"]
            row["taker_realized_pnl" if taker else "maker_realized_pnl"] += (
                pnl["realized_pnl"] * quote
            )
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
                row[direction + "_position_flips"] += int(
                    sign * before < 0 < sign * after
                )
        if trade["liquidated"]:
            direction = "short" if bought else "long"
            row["liquidations"] += 1
            for suffix, amount in (("size", size), ("value", value)):
                row["liquidated_" + suffix] += amount
                row["liquidated_" + direction + "_" + suffix] += amount
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


def phoenix_positioning(symbol, start, end):
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
    trades = []
    for signature, rows in by_signature.items():
        groups = transaction_events(transaction(signature), signature)
        parsed = parsed_trades(groups, symbol)
        validate_native(parsed, rows, start, end)
        trades.extend(parsed)
    # This is the venue's explicit execution sequence, not timestamp ordering.
    trades.sort(key=lambda trade: trade["summary"]["trade_sequence_number"])
    return aggregate(trades)
