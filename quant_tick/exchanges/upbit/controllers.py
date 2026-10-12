from datetime import UTC, timedelta
from decimal import Decimal

import pandas as pd

from quant_tick.controllers import ChunkedExchangeS3, ExchangeREST
from quant_tick.controllers.s3 import get_archive_followup_start
from quant_tick.lib import get_current_time, parse_datetime, zip_downloader

from .api import get_upbit_response
from .candles import upbit_candles


def upbit_trades(
    symbol, timestamp_from, timestamp_to, on_data_frame, retry=False, verbose=False
):
    kwargs = {
        "timestamp_from": timestamp_from,
        "timestamp_to": timestamp_to,
        "on_data_frame": on_data_frame,
        "retry": retry,
        "verbose": verbose,
    }
    UpbitArchive(symbol, **kwargs).main()
    kwargs["timestamp_from"] = get_archive_followup_start(timestamp_from)
    if kwargs["timestamp_from"] < timestamp_to:
        UpbitTrades(symbol, **kwargs).main()


class UpbitMixin:
    def get_candles(self, timestamp_from, timestamp_to):
        return upbit_candles(self.symbol.api_symbol, timestamp_from, timestamp_to)


class UpbitTrades(UpbitMixin, ExchangeREST):
    partition_scoped = True

    def get_pagination_id(self, timestamp_to):
        return timestamp_to

    def iter_api(self, timestamp_from, pagination_id):
        pagination_id = pagination_id.astimezone(UTC)
        day = (pagination_id - timedelta(microseconds=1)).date()
        days_ago = (get_current_time().date() - day).days
        if not 0 <= days_ago <= 7:
            raise ValueError("Upbit REST trades are limited to the past seven days")
        params = {"market": self.symbol.api_symbol, "days_ago": days_ago, "count": 500}
        if pagination_id.date() == day:
            params["to"] = pagination_id.strftime("%H:%M:%S")
        rows = []
        previous_id = None
        previous_time = pagination_id
        while True:
            page = get_upbit_response("/trades/ticks", params)
            if not page:
                break
            for row in page:
                trade_id = int(row["sequential_id"])
                timestamp = parse_datetime(row["timestamp"], unit="ms")
                if previous_id is not None and trade_id >= previous_id:
                    raise ValueError("Upbit trade pagination did not advance")
                if timestamp > previous_time:
                    raise ValueError("Upbit trades are not ordered newest first")
                previous_id, previous_time = trade_id, timestamp
                if timestamp < timestamp_from:
                    return rows, True, None
                if timestamp < pagination_id:
                    rows.append(row)
            params["cursor"] = previous_id
        return rows, True, None

    def parse_data(self, data):
        rows = []
        for row in data:
            if row["market"] != self.symbol.api_symbol or row["ask_bid"] not in (
                "ASK",
                "BID",
            ):
                raise ValueError("Upbit trade has an invalid market or side")
            price = Decimal(str(row["trade_price"]))
            quantity = Decimal(str(row["trade_volume"]))
            rows.append(
                {
                    "uid": str(row["sequential_id"]),
                    "index": int(row["sequential_id"]),
                    "timestamp": parse_datetime(row["timestamp"], unit="ms"),
                    "nanoseconds": 0,
                    "price": price,
                    "notional": quantity,
                    "volume": price * quantity,
                    "tickRule": 1 if row["ask_bid"] == "BID" else -1,
                }
            )
        return rows


class UpbitArchive(UpbitMixin, ChunkedExchangeS3):
    """Archive sequence IDs are file-local; row order resolves second-precision ties.

    Archives contain timestamp inversions. Sort the whole day by time and sequence
    before dividing it into chunks so late rows reach the correct hour.
    Prices, quote amounts and significant-trade filter thresholds use the market's
    quote currency (KRW for KRW-BTC).
    """

    csv_columns = ("seq", "timestamp", "volume", "price", "ask_bid")

    def get_data_frame_chunks(self, value):
        symbol = self.symbol.api_symbol
        frame = zip_downloader(
            f"https://crix-data.upbit.com/trade/{symbol}/daily/{value:%Y}/"
            f"{symbol}_trade_{value:%Y%m%d}.zip",
            self.csv_columns,
        )
        if frame is None:
            return None
        frame = frame.loc[frame["seq"] != "seq"].copy()
        frame[["timestamp", "seq"]] = frame[["timestamp", "seq"]].astype("int64")
        frame = frame.sort_values(["timestamp", "seq"], kind="stable")
        return (
            frame.iloc[start : start + self.archive_chunksize]
            for start in range(0, len(frame), self.archive_chunksize)
        )

    def get_archive_timestamp_nanoseconds(self, frame):
        return frame["timestamp"].astype("int64") * 1_000_000

    def parse_dtypes_and_strip_columns(self, frame):
        if not frame["ask_bid"].isin(("ASK", "BID")).all():
            raise ValueError("Upbit archive has an invalid trade side")
        timestamps = pd.to_datetime(
            frame["timestamp"].astype("int64"), unit="ms", utc=True
        )
        result = pd.DataFrame(
            {
                "uid": timestamps.dt.strftime("%Y%m%d")
                + ":"
                + frame["seq"].astype(str),
                "timestamp": timestamps,
                "nanoseconds": 0,
                "price": frame["price"].map(Decimal),
                "notional": frame["volume"].map(Decimal),
                "tickRule": frame["ask_bid"].map({"BID": 1, "ASK": -1}),
            }
        )
        result["volume"] = result["price"] * result["notional"]
        return result[self.columns]
