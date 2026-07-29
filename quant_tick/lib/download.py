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
    if response.status_code == 404:
        logger.info(f"Archive not found: {url}")
        return None
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


def gzip_chunk_downloader(
    url: str,
    columns: Iterable[str],
    *,
    chunksize: int,
) -> Iterable[DataFrame] | None:
    """Download a gzipped CSV and return a chunked reader."""
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
        chunksize=chunksize,
    )


def zip_chunk_downloader(
    url: str,
    columns: Iterable[str],
    *,
    chunksize: int,
    usecols: Iterable[str] | None = None,
) -> Iterable[DataFrame] | None:
    """Download a ZIP archive and return a chunked reader for its CSV."""
    content = download_content(url)
    if content is None:
        return None
    archive = BytesIO(content)
    zf = zipfile.ZipFile(archive)
    csv_files = [name for name in zf.namelist() if name.endswith(".csv")]
    if not csv_files:
        logger.warning(f"No CSV in ZIP: {url}")
        zf.close()
        archive.close()
        return None
    selected_columns = list(usecols or columns)
    csv_file = zf.open(csv_files[0])
    try:
        reader = pd.read_csv(
            csv_file,
            names=columns,
            usecols=selected_columns,
            dtype={col: "str" for col in selected_columns},
            chunksize=chunksize,
        )
    except Exception:
        csv_file.close()
        zf.close()
        archive.close()
        raise

    class Chunks:
        def __iter__(self):
            return self

        def __next__(self):
            return next(reader)

        def close(self):
            reader.close()
            csv_file.close()
            zf.close()
            archive.close()

    return Chunks()


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
