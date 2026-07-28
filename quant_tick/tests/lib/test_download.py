from unittest.mock import Mock, patch

from django.test import SimpleTestCase

from quant_tick.lib.download import download_content


class DownloadContentTest(SimpleTestCase):
    @patch("quant_tick.lib.download.httpx.get")
    def test_archive_not_found_is_not_logged_as_error(self, get):
        get.return_value = Mock(status_code=404)

        with patch("quant_tick.lib.download.logger.error") as error:
            result = download_content("https://example.com/archive.zip")

        self.assertIsNone(result)
        error.assert_not_called()

    @patch("quant_tick.lib.download.httpx.get")
    def test_other_http_error_is_logged(self, get):
        get.return_value = Mock(status_code=503)

        with patch("quant_tick.lib.download.logger.error") as error:
            result = download_content("https://example.com/archive.zip")

        self.assertIsNone(result)
        error.assert_called_once_with("Error 503: https://example.com/archive.zip")
