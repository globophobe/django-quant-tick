from datetime import date, timedelta
from decimal import Decimal
from functools import cached_property

import pandas as pd

from quant_tick.controllers import ChunkedExchangeS3, ExchangeREST
from quant_tick.controllers.s3 import get_archive_followup_start
from quant_tick.lib import parse_datetime, zip_chunk_downloader

from .api import contract_value, get_okx_response
from .candles import okx_candles


def okx_trades(
    symbol, timestamp_from, timestamp_to, on_data_frame, retry=False, verbose=False
):
    kwargs = {
        "timestamp_from": timestamp_from,
        "timestamp_to": timestamp_to,
        "on_data_frame": on_data_frame,
        "retry": retry,
        "verbose": verbose,
    }
    OKXArchive(symbol, **kwargs).main()
    kwargs["timestamp_from"] = get_archive_followup_start(timestamp_from)
    if kwargs["timestamp_from"] < timestamp_to:
        OKXTrades(symbol, **kwargs).main()


class OKXMixin:
    @cached_property
    def base_per_contract(self):
        return contract_value(self.symbol.api_symbol)

    def get_candles(self, timestamp_from, timestamp_to):
        return okx_candles(self.symbol.api_symbol, timestamp_from, timestamp_to)


class OKXTrades(OKXMixin, ExchangeREST):
    partition_scoped = True

    def get_pagination_id(self, timestamp_to):
        return timestamp_to

    def iter_api(self, timestamp_from, pagination_id):
        params = {
            "instId": self.symbol.api_symbol,
            "type": "2",
            "after": int(pagination_id.timestamp() * 1000),
            "limit": 100,
        }
        rows = []
        previous_id = None
        previous_time = pagination_id
        while True:
            page = get_okx_response("/market/history-trades", params)
            if not page:
                break
            for row in page:
                timestamp = parse_datetime(int(row["ts"]), unit="ms")
                trade_id = int(row["tradeId"])
                if (
                    previous_id is not None and trade_id >= previous_id
                ) or timestamp > previous_time:
                    raise ValueError("OKX trades are not ordered newest first")
                previous_id, previous_time = trade_id, timestamp
                if timestamp < timestamp_from:
                    return rows, True, None
                if timestamp < pagination_id:
                    rows.append(row)
            params = {
                "instId": self.symbol.api_symbol,
                "type": "1",
                "after": str(previous_id),
                "limit": 100,
            }
        return rows, True, None

    def parse_data(self, data):
        rows = []
        for row in data:
            if (
                row["side"] not in ("buy", "sell")
                or row["instId"] != self.symbol.api_symbol
            ):
                raise ValueError("OKX trade has an invalid market or side")
            price = Decimal(row["px"])
            quantity = Decimal(row["sz"]) * self.base_per_contract
            rows.append(
                {
                    "uid": row["tradeId"],
                    "index": int(row["tradeId"]),
                    "timestamp": parse_datetime(int(row["ts"]), unit="ms"),
                    "nanoseconds": 0,
                    "price": price,
                    "notional": quantity,
                    "volume": price * quantity,
                    "tickRule": 1 if row["side"] == "buy" else -1,
                }
            )
        return rows


class OKXArchive(OKXMixin, ChunkedExchangeS3):
    """Two UTC+8 archive dates cover one complete UTC partition day."""

    csv_columns = (
        "instrument_name",
        "trade_id",
        "side",
        "price",
        "size",
        "created_time",
    )

    def get_data_frame_chunks(self, value):
        readers = []
        dates = (value, value + timedelta(days=1))
        symbol = self.symbol.api_symbol
        try:
            for day in dates:
                url = (
                    "https://static.okx.com/cdn/okex/traderecords/trades/daily/"
                    f"{day:%Y%m%d}/{symbol}-trades-{day.isoformat()}.zip"
                )
                reader = zip_chunk_downloader(
                    url, self.csv_columns, chunksize=self.archive_chunksize
                )
                if reader is None:
                    for previous in readers:
                        previous.close()
                    return None
                readers.append(reader)
        except Exception:
            for reader in readers:
                reader.close()
            raise

        repair_chunks = self._repair_archive_chunks

        class Chunks:
            def __iter__(self):
                for day, reader in zip(dates, readers, strict=True):
                    yield from repair_chunks(day, reader)

            def close(self):
                for reader in readers:
                    reader.close()

        return Chunks()

    def _repair_archive_chunks(self, day, frames):
        if self.symbol.api_symbol == "BTC-USDT" and day == date(2022, 7, 22):
            # This file has one-second regressions across CSV read boundaries.
            frame = self.prepare_archive_chunk(pd.concat(frames, ignore_index=True))
            frame = frame.sort_values(
                "created_time", key=lambda values: values.astype("int64"), kind="stable"
            )
            for start in range(0, len(frame), self.archive_chunksize):
                yield frame.iloc[start : start + self.archive_chunksize]
            return
        for frame in frames:
            yield self._repair_archive_chunk(day, frame)

    def _repair_archive_chunk(self, day, frame):
        if day == date(2023, 2, 10):
            # These files repeat the previous archive's final fills.
            repeated_ids = {
                "BTC-USDT": range(397123645, 397123650),
                "BTC-USDT-SWAP": range(478916473, 478916484),
            }.get(self.symbol.api_symbol, ())
            return frame.loc[~frame["trade_id"].isin(map(str, repeated_ids))]
        if self.symbol.api_symbol != "BTC-USDT":
            return frame
        if day == date(2022, 9, 14):
            before = frame["trade_id"].eq("372030796")
            if before.any():
                # These two fills were published in September 15's misplaced prefix.
                restored = pd.DataFrame(
                    [
                        [
                            "BTC-USDT",
                            "372030794",
                            "buy",
                            "20370.5",
                            "0.02953379",
                            "1663123239265",
                        ],
                        [
                            "BTC-USDT",
                            "372030795",
                            "buy",
                            "20370.5",
                            "0.00525621",
                            "1663123239265",
                        ],
                    ],
                    columns=self.csv_columns,
                )
                offset = before.to_numpy().argmax()
                return pd.concat(
                    [frame.iloc[:offset], restored, frame.iloc[offset:]],
                    ignore_index=True,
                )
        elif day == date(2022, 9, 15):
            # Two duplicates plus the two fills restored to September 14 above.
            return frame.loc[
                ~frame["trade_id"].isin(
                    ("372030793", "372030794", "372030795", "372030796")
                )
            ]
        return frame

    def prepare_archive_chunk(self, frame):
        return frame.loc[frame["trade_id"] != "trade_id"].copy()

    def get_archive_timestamp_nanoseconds(self, frame):
        return frame["created_time"].astype("int64") * 1_000_000

    def parse_dtypes_and_strip_columns(self, frame):
        side = frame["side"].str.lower()
        if (
            not frame["instrument_name"].eq(self.symbol.api_symbol).all()
            or not side.isin(("buy", "sell")).all()
        ):
            raise ValueError("OKX archive has an invalid market or side")
        result = pd.DataFrame(
            {
                "uid": frame["trade_id"].astype(str),
                "timestamp": pd.to_datetime(
                    frame["created_time"].astype("int64"), unit="ms", utc=True
                ),
                "nanoseconds": 0,
                "price": frame["price"].map(Decimal),
                "notional": frame["size"].map(Decimal) * self.base_per_contract,
                "tickRule": side.map({"buy": 1, "sell": -1}),
            }
        )
        result["volume"] = result["price"] * result["notional"]
        return result[self.columns]
