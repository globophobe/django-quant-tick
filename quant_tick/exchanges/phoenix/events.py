"""Length-framed Phoenix perpetual event decoding for public trade statistics.

Layouts follow phoenix-rise-events 0.10.2 (rise-public f67e8b4). Preserve CPI
ancestry, execution order and integer lots. Known unrelated events are skipped
using their explicit lengths; unknown variants, legacy unframed batches and
incomplete metadata fail rather than produce partial statistics.
"""

import struct

PROGRAM = "EtrnLzgbS7nMMy5fbD42kXiUzGg8XQzJ972Xtk1cjWih"
LOG = 0xAACFD109F2D6E68D
LENGTHS = 0x479947B5CB8607F7
ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"

LAYOUTS = {
    0: ("SlotContext", "QQ", "timestamp slot"),
    1: (
        "Header",
        "QQ16sIIbB32s32s",
        "sequence_number prev_sequence_number_slot symbol asset_id tick_size base_lot_decimals quote_lot_decimals signer trader_account",
    ),
    3: (
        "OrderFilled",
        "QBQQQQ32siqqqq",
        "order_sequence_number side price base_lots_filled quote_lots_filled quantity_remaining maker maker_fee_rate maker_base_lot_position maker_virtual_quote_lot_position maker_quote_lot_collateral maker_cumulative_funding_snapshot",
    ),
    5: (
        "SplineFilled",
        "QBQQQ32siqqqq",
        "spline_sequence_number side price base_lots_filled quote_lots_filled maker maker_fee_rate maker_base_lot_position maker_virtual_quote_lot_position maker_quote_lot_collateral maker_cumulative_funding_snapshot",
    ),
    6: (
        "TradeSummary",
        "32sQQBQQQqqqq",
        "trader trade_sequence_number prev_trade_sequence_number_slot side base_lots_filled quote_lots_filled fee_in_quote_lots base_lot_position virtual_quote_lot_position quote_lot_collateral cumulative_funding_snapshot",
    ),
    32: (
        "Liquidation",
        "32s32sIQQQQB",
        "liquidator liquidated_trader asset_id liquidation_size mark_price base_lots_filled quote_lots_filled position_closed",
    ),
    37: (
        "PnL",
        "32sI16sqqqqqq",
        "trader asset_id symbol realized_pnl funding_payment base_lots_before base_lots_after virtual_quote_lots_before virtual_quote_lots_after",
    ),
}


def decode58(value):
    number = 0
    for char in value:
        try:
            digit = ALPHABET.index(char)
        except ValueError as exc:
            raise ValueError("Invalid Solana base58 data") from exc
        number = number * 58 + digit
    return b"\0" * (len(value) - len(value.lstrip("1"))) + number.to_bytes(
        (number.bit_length() + 7) // 8, "big"
    )


class Reader:
    def __init__(self, data):
        self.data = data
        self.offset = 0

    def read(self, fmt):
        size = struct.calcsize("<" + fmt)
        if self.offset + size > len(self.data):
            raise ValueError("Truncated Phoenix event data")
        values = struct.unpack_from("<" + fmt, self.data, self.offset)
        self.offset += size
        return values

    def optional(self, fmt):
        (present,) = self.read("B")
        if present not in (0, 1):
            raise ValueError("Invalid Phoenix optional field")
        return self.read(fmt)[0] if present else None

    def fields(self, fmt, names):
        return dict(zip(names.split(), self.read(fmt), strict=True))

    def finish(self):
        if self.offset != len(self.data):
            raise ValueError("Unexpected Phoenix event length")


def event(data):
    reader = Reader(data)
    (tag,) = reader.read("B")
    if tag in LAYOUTS:
        name, fmt, names = LAYOUTS[tag]
        result = reader.fields(fmt, names)
    elif tag == 8:
        name = "MarketSummary"
        result = reader.fields("16sIQ", "symbol asset_id open_interest")
        result["total_maker_quote_lot_fees"] = reader.optional("q")
        result["total_taker_quote_lot_fees"] = reader.optional("Q")
        result.update(reader.fields("QQ", "mark_price spot_price"))
    elif tag == 76:
        name = "LiquidationWithFee"
        result = reader.fields(*LAYOUTS[32][1:])
        result["liquidator_trader"] = reader.optional("32s")
        result.update(
            reader.fields(
                "HQQQ",
                "liquidation_fee_bps filled_notional fee_charged fee_charged_liquidator_portion",
            )
        )
    elif tag == 71:
        name = "AdditionalFeeAccrued"
        result = reader.fields("32s", "trader")
        result["order_sequence_number"] = reader.optional("Q")
        result.update(
            reader.fields(
                "BQQ", "additional_fee_tenths_bps quote_lots_filled fee_in_quote_lots"
            )
        )
    elif tag <= 76:
        return None
    else:
        raise ValueError("Unknown Phoenix event variant")
    reader.finish()
    if "symbol" in result:
        result["symbol"] = result["symbol"].rstrip(b"\0").decode("ascii")
    if "side" in result:
        if result["side"] not in (0, 1):
            raise ValueError("Invalid Phoenix event side")
        result["side"] = ("bid", "ask")[result["side"]]
    if "position_closed" in result:
        if result["position_closed"] not in (0, 1):
            raise ValueError("Invalid Phoenix boolean field")
        result["position_closed"] = bool(result["position_closed"])
    return {"kind": name, **result}


def decode_logs(logs):
    lengths, batches = {}, []
    for tag, data in logs:
        reader = Reader(data)
        index, count = reader.read("II")
        if tag == LENGTHS:
            if index in lengths:
                raise ValueError("Duplicate Phoenix event length batch")
            if count > (len(data) - reader.offset) // 2:
                raise ValueError("Truncated Phoenix event length batch")
            lengths[index] = reader.read("H" * count)
            reader.finish()
        else:
            batches.append((index, count, data[reader.offset :]))
    result = []
    for index, count, data in batches:
        sizes = lengths.pop(index, None)
        if sizes is None or len(sizes) != count or sum(sizes) != len(data):
            raise ValueError("Missing or inconsistent Phoenix event length framing")
        offset = 0
        for size in sizes:
            parsed = event(data[offset : offset + size])
            if parsed is not None:
                result.append(parsed)
            offset += size
    if lengths:
        raise ValueError("Phoenix event payload missing for its length batch")
    return result


def transaction_events(tx, signature):
    """Return each emitting invocation's relevant events, in execution order."""
    if (
        tx.get("meta") is None
        or tx["meta"].get("err") is not None
        or tx["transaction"]["signatures"][0] != signature
    ):
        raise ValueError("Missing, failed or unexpected Phoenix transaction")
    message, meta = tx["transaction"]["message"], tx["meta"]
    loaded = meta.get("loadedAddresses", {})
    if message.get("addressTableLookups") and not loaded:
        raise ValueError("Missing Solana lookup addresses")
    keys = (
        message["accountKeys"] + loaded.get("writable", []) + loaded.get("readonly", [])
    )
    groups = {}
    inner = meta.get("innerInstructions")
    if inner is None:
        raise ValueError("Missing Phoenix inner instructions")

    def append(ix, path):
        index = ix["programIdIndex"]
        if not 0 <= index < len(keys):
            raise ValueError("Invalid Solana program index")
        if keys[index] != PROGRAM:
            return
        data = decode58(ix["data"])
        if len(data) < 8:
            raise ValueError("Truncated Phoenix instruction")
        tag = int.from_bytes(data[:8], "little")
        if tag in (LOG, LENGTHS):
            if path[:-1] not in groups:
                raise ValueError("Orphan Phoenix event log")
            groups[path[:-1]].append((tag, data[8:]))
        else:
            groups[path] = []

    by_index = {}
    for group in inner:
        index = group["index"]
        if index in by_index or not 0 <= index < len(message["instructions"]):
            raise ValueError("Invalid Solana inner instruction group")
        by_index[index] = group["instructions"]
    for index, ix in enumerate(message["instructions"]):
        append(ix, (index,))
        path = []
        for child in by_index.get(index, []):
            height = child.get("stackHeight")
            if height is None or not 2 <= height <= len(path) + 2:
                raise ValueError("Missing or invalid Phoenix CPI ancestry")
            depth = height - 1
            if depth > len(path):
                path.append(0)
            else:
                path = path[:depth]
                path[-1] += 1
            append(child, (index, *path))
    result = []
    for logs in groups.values():
        events = decode_logs(logs)
        for item in events:
            if item["kind"] == "SlotContext" and item["slot"] != tx["slot"]:
                raise ValueError("Phoenix event slot does not match transaction")
        if events:
            result.append(events)
    if not result:
        raise ValueError("Phoenix transaction has no decodable events")
    return result
