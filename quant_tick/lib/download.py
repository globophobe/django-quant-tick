import logging
import zipfile
from collections.abc import Iterable
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile

import httpx
import pandas as pd
from pandas import DataFrame

logger = logging.getLogger(__name__)


def gzip_downloader(url: str, columns: Iterable[str]) -> DataFrame:
    """GZIP downloader.

    Streaming downloads gave many EOFErrors, so regular download.
    """
    response = httpx.get(url)
    if response.status_code == 200:
        temp_file = NamedTemporaryFile()
        filename = temp_file.name
        with open(filename, "wb+") as temp:
            temp.write(response.content)
            path = Path(filename)
            size = path.stat().st_size
            if size > 0:
                # Extract gzip.
                return pd.read_csv(
                    filename,
                    usecols=columns,
                    engine="python",
                    compression="gzip",
                    dtype={col: "str" for col in columns},
                )
            else:
                logger.warning(f"No data: {url}")
    else:
        logger.error(f"Error {response.status_code}: {url}")


def zip_downloader(url: str, columns: Iterable[str]) -> DataFrame | None:
    """ZIP downloader for Binance data.binance.vision archives."""
    response = httpx.get(url)
    if response.status_code == 200:
        with zipfile.ZipFile(BytesIO(response.content)) as zf:
            csv_files = [f for f in zf.namelist() if f.endswith(".csv")]
            if csv_files:
                with zf.open(csv_files[0]) as csv_file:
                    return pd.read_csv(
                        csv_file,
                        names=columns,
                        dtype={col: "str" for col in columns},
                    )
            else:
                logger.warning(f"No CSV in ZIP: {url}")
    else:
        logger.error(f"Error {response.status_code}: {url}")
    return None
