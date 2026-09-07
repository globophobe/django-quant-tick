from io import BytesIO
from unittest.mock import Mock, patch
from zipfile import ZipFile

from django.test import SimpleTestCase

from quant_tick.lib.download import (
    ArchiveDownloadError,
    download_content,
    zip_chunk_downloader,
    zip_downloader,
)


class DownloadContentTest(SimpleTestCase):
    @patch("quant_tick.lib.download.httpx2.get")
    def test_archive_not_found_is_not_logged_as_error(self, get):
        get.return_value = Mock(status_code=404)

        with patch("quant_tick.lib.download.logger.error") as error:
            result = download_content("https://example.com/archive.zip")

        self.assertIsNone(result)
        error.assert_not_called()

    @patch("quant_tick.lib.download.httpx2.get")
    def test_other_http_error_is_logged(self, get):
        get.return_value = Mock(status_code=503)

        with (
            patch("quant_tick.lib.download.logger.error") as error,
            self.assertRaisesRegex(ArchiveDownloadError, "HTTP 503"),
        ):
            download_content("https://example.com/archive.zip")

        error.assert_called_once_with(
            "Error 503: https://example.com/archive.zip"
        )

    @patch("quant_tick.lib.download.httpx2.get")
    def test_empty_success_response_raises(self, get):
        get.return_value = Mock(status_code=200, content=b"")

        with self.assertRaisesRegex(ArchiveDownloadError, "Archive is empty"):
            download_content("https://example.com/archive.zip")

    @patch("quant_tick.lib.download.download_content")
    def test_zip_without_csv_raises_instead_of_looking_missing(self, download):
        archive = BytesIO()
        with ZipFile(archive, "w") as zip_file:
            zip_file.writestr("README.txt", "not market data")
        download.return_value = archive.getvalue()

        with self.assertRaisesRegex(ArchiveDownloadError, "contains no CSV"):
            zip_downloader("https://example.com/archive.zip", ["timestamp"])
        with self.assertRaisesRegex(ArchiveDownloadError, "contains no CSV"):
            zip_chunk_downloader(
                "https://example.com/archive.zip",
                ["timestamp"],
                chunksize=10,
            )
