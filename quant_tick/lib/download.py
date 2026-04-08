import logging
import zipfile
from collections.abc import Iterable
from io import BytesIO

import httpx
import pandas as pd
from pandas import DataFrame

logger = logging.getLogger(__name__)


class ArchiveDownloadError(RuntimeError):
    pass


def download_content(url: str) -> bytes | None:
    """Download response content."""
    try:
        response = httpx.get(url)
    except httpx.RequestError as exc:
        raise ArchiveDownloadError(f"Archive download failed: {url}") from exc
    if response.status_code == 200:
        return response.content
    logger.error(f"Error {response.status_code}: {url}")
    return None


def gzip_downloader(url: str, columns: Iterable[str]) -> DataFrame | None:
    """Download and parse a gzipped CSV.

    Streaming downloads gave many EOFErrors, so regular download.
    """
    content = download_content(url)
    if content is None:
        return None
    if not content:
        logger.warning(f"No data: {url}")
        return None
    return pd.read_csv(
        BytesIO(content),
        usecols=columns,
        compression="gzip",
        dtype={col: "str" for col in columns},
    )


def zip_downloader(url: str, columns: Iterable[str]) -> DataFrame | None:
    """Download and parse a ZIP archive containing one CSV."""
    content = download_content(url)
    if content is None:
        return None
    with zipfile.ZipFile(BytesIO(content)) as zf:
        csv_files = [f for f in zf.namelist() if f.endswith(".csv")]
        if csv_files:
            with zf.open(csv_files[0]) as csv_file:
                return pd.read_csv(
                    csv_file,
                    names=columns,
                    dtype={col: "str" for col in columns},
                )
        logger.warning(f"No CSV in ZIP: {url}")
    return None
