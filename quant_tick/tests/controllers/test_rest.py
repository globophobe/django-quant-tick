from datetime import UTC, datetime, timedelta

from django.test import SimpleTestCase

from quant_tick.controllers import is_terminal_page


class FixedIntervalTerminalPageTest(SimpleTestCase):
    def test_short_page_spanning_complete_time_window_is_not_terminal(self):
        timestamp_from = datetime(2025, 10, 22, 12, tzinfo=UTC)
        data = [
            timestamp_from + timedelta(hours=hour)
            for hour in range(300)
            if hour not in {12, 53, 109, 208, 240}
        ]

        self.assertFalse(
            is_terminal_page(
                data,
                get_timestamp=lambda item: item,
                interval=timedelta(hours=1),
                max_results=300,
            )
        )

    def test_short_page_with_short_time_span_is_terminal(self):
        timestamp_from = datetime(2025, 10, 22, 12, tzinfo=UTC)
        data = [timestamp_from + timedelta(hours=hour) for hour in range(295)]

        self.assertTrue(
            is_terminal_page(
                data,
                get_timestamp=lambda item: item,
                interval=timedelta(hours=1),
                max_results=300,
            )
        )
