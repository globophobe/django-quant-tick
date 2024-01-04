import logging
from collections.abc import Iterable
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
            size = Path.stat(filename).st_size
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
                logger.warn(f"No data: {url}")
    else:
        logger.error(f"Error {response.status_code}: {url}")
