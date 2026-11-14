"""Published trade archives, cached for one partition per collection run.

Files are partitioned by receipt time, so exchange-time boundary minutes may be
incomplete. Callers must verify minute quantities before choosing archive rows
over REST. Missing objects allow fallback; access and content errors do not.
"""

import time
from datetime import UTC, datetime, timedelta
from io import BytesIO
from urllib.parse import quote
from xml.etree import ElementTree

import httpx2
import pandas as pd

from quant_tick.controllers import (
    HTTPX_ERRORS,
    increment_api_total_requests,
    throttle_api_requests,
)
from quant_tick.lib import get_current_time
from quant_tick.lib.download import ArchiveDownloadError

from .api import get_coinapi_key, get_coinapi_symbol_type

BASE_URL = "https://s3.flatfiles.coinapi.io/coinapi/"
HOURLY_FROM = datetime(2026, 6, 9, tzinfo=UTC)
CSV_COLUMNS = (
    "time_exchange",
    "time_coinapi",
    "guid",
    "price",
    "base_amount",
    "taker_side",
    "id_exch_guid",
    "id_exch_int_inc",
)


def get_flat_file_content(key="", *, params=None):
    headers = {"Authorization": get_coinapi_key(), "Accept": "application/xml"}
    url = BASE_URL + quote(key, safe="/")
    for attempt in range(4):
        throttle_api_requests("COINAPI_FILES_RESET", "COINAPI_FILES_TOTAL", 1, 2)
        increment_api_total_requests("COINAPI_FILES_TOTAL")
        try:
            response = httpx2.get(url, params=params, headers=headers, timeout=60)
            if response.status_code in (404, 550):
                return None
            response.raise_for_status()
        except HTTPX_ERRORS as exc:
            status = (
                exc.response.status_code
                if isinstance(exc, httpx2.HTTPStatusError)
                else None
            )
            if attempt == 3 or (status is not None and status != 429 and status < 500):
                raise ArchiveDownloadError(
                    f"CoinAPI Flat Files request failed (HTTP {status})",
                    status_code=status,
                ) from exc
            time.sleep(2)
            continue
        if not response.content:
            raise ArchiveDownloadError("CoinAPI Flat Files returned empty content")
        return response.content
    raise AssertionError("unreachable")


def find_trade_file(api_symbol, partition):
    get_coinapi_symbol_type(api_symbol)
    exchange = api_symbol.split("_SPOT_")[0].split("_PERP_")[0]
    prefix = f"T-TRADES/D-{partition}/E-{exchange}/"
    marker = f"+SC-{api_symbol}+S-"
    params = {"prefix": prefix, "list-type": "2"}
    matches = []
    tokens = set()
    while True:
        content = get_flat_file_content(params=params)
        if content is None:
            return None
        root = ElementTree.fromstring(content)
        if root.tag.rsplit("}", 1)[-1] != "ListBucketResult":
            raise ValueError("CoinAPI Flat Files returned an unexpected listing")
        for item in root.findall("{*}Contents"):
            key = item.findtext("{*}Key", "")
            if (
                key.startswith(prefix + "IDDI-")
                and marker in key
                and key.endswith(".csv.gz")
                and "/" not in key[len(prefix) :]
            ):
                matches.append(key)
        truncated = root.findtext("{*}IsTruncated")
        if truncated == "false":
            break
        token = root.findtext("{*}NextContinuationToken")
        if truncated != "true" or not token or token in tokens:
            raise ValueError("CoinAPI Flat Files listing did not advance")
        tokens.add(token)
        params["continuation-token"] = token
    if len(matches) > 1:
        raise ValueError("CoinAPI Flat Files has ambiguous symbol files")
    return matches[0] if matches else None


class CoinAPITradeFiles:
    def __init__(self, api_symbol):
        self.api_symbol = api_symbol
        self.partition = None
        self.content = None

    def get_trades(self, timestamp_from, timestamp_to):
        hourly = timestamp_from >= HOURLY_FROM
        start = pd.Timestamp(timestamp_from).floor("h" if hourly else "D")
        end = start + timedelta(hours=1 if hourly else 24)
        if timestamp_to > end or end > get_current_time():
            return None
        partition = start.strftime("%Y%m%d%H" if hourly else "%Y%m%d")
        if partition != self.partition:
            # Release the previous compressed archive before downloading another.
            self.content = None
            self.partition = None
            key = find_trade_file(self.api_symbol, partition)
            self.content = get_flat_file_content(key) if key else None
            self.partition = partition
        if self.content is None:
            return None
        rows = []
        with pd.read_csv(
            BytesIO(self.content),
            sep=";",
            compression="gzip",
            usecols=CSV_COLUMNS,
            dtype=str,
            keep_default_na=False,
            chunksize=200_000,
        ) as chunks:
            for chunk in chunks:
                timestamps = pd.to_datetime(
                    chunk.time_exchange, format="ISO8601", utc=True
                )
                if timestamps.isna().any():
                    raise ValueError("CoinAPI Flat Files has missing trade timestamps")
                chunk = chunk.loc[
                    (timestamps >= timestamp_from) & (timestamps < timestamp_to)
                ]
                for row in chunk.to_dict("records"):
                    rows.append(
                        {
                            "time_exchange": row["time_exchange"],
                            "time_coinapi": row["time_coinapi"],
                            "uuid": row["guid"],
                            "price": row["price"],
                            "size": row["base_amount"],
                            "taker_side": row["taker_side"],
                            "id_trade": row["id_exch_guid"]
                            or row["id_exch_int_inc"]
                            or None,
                            "coinapi_transport": "flat_files",
                        }
                    )
        # Receipt ordering can differ from exchange time; keep tied rows stable.
        rows.sort(key=lambda row: pd.to_datetime(row["time_exchange"], utc=True))
        return rows
