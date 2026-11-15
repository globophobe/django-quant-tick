"""Published trade archives, parsed once and cached for one partition per run.

Files are partitioned by receipt time, so exchange-time boundary minutes may be
incomplete. Callers must verify minute quantities before choosing archive rows
over REST. Missing objects allow fallback; access and content errors do not.

Receipt ordering does not permit flushing an exchange-time hour before the
archive ends. Retain one sorted partition (at most a day), release it on the next
partition, and reuse its parsed timestamps across the requested hourly windows.
"""

import logging
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

logger = logging.getLogger(__name__)

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
    limit_reported = False
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
            if status in (402, 429) and not limit_reported:
                logger.error(
                    "CoinAPI Flat Files credit or rate limit reached (HTTP %s)", status
                )
                limit_reported = True
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
        self.frame = None

    def get_trades(self, timestamp_from, timestamp_to):
        hourly = timestamp_from >= HOURLY_FROM
        start = pd.Timestamp(timestamp_from).floor("h" if hourly else "D")
        end = start + timedelta(hours=1 if hourly else 24)
        if timestamp_to > end or end > get_current_time():
            return None
        partition = start.strftime("%Y%m%d%H" if hourly else "%Y%m%d")
        if partition != self.partition:
            self.frame = None
            self.partition = None
            key = find_trade_file(self.api_symbol, partition)
            content = get_flat_file_content(key) if key else None
            if content is not None:
                self.frame = self.parse_archive(content)
            self.partition = partition
        if self.frame is None:
            return None
        rows = []
        selected = self.frame.loc[
            (self.frame.time_exchange >= timestamp_from)
            & (self.frame.time_exchange < timestamp_to)
        ]
        for row in selected.to_dict("records"):
            rows.append(
                {
                    "time_exchange": row["time_exchange"],
                    "time_coinapi": row["time_coinapi"],
                    "uuid": row["guid"],
                    "price": row["price"],
                    "size": row["base_amount"],
                    "taker_side": row["taker_side"],
                    "id_trade": row["id_exch_guid"] or row["id_exch_int_inc"] or None,
                    "coinapi_transport": "flat_files",
                }
            )
        return rows

    @staticmethod
    def parse_archive(content):
        frames = []
        with pd.read_csv(
            BytesIO(content),
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
                chunk["time_exchange"] = timestamps
                frames.append(chunk)
        # Receipt ordering can differ from exchange time; keep tied rows stable.
        return pd.concat(frames, ignore_index=True).sort_values(
            "time_exchange", kind="stable"
        )
