from datetime import UTC, datetime
from unittest.mock import patch

import time_machine
from django.core.management import call_command
from django.test import TestCase

from quant_tick.constants import Frequency
from quant_tick.models import CandleCache, CandleData
from quant_tick.tests.models.candle_types.base import BaseMinuteIteratorTest
from quant_tick.tests.models.candle_types.time_based.base import BaseTimeBasedCandleTest


@time_machine.travel(datetime(2009, 1, 4, tzinfo=UTC), tick=False)
@patch(
    "quant_tick.models.candles.get_current_time",
    return_value=datetime(2009, 1, 4, 0, 3, tzinfo=UTC),
)
class CandleCommandTest(BaseMinuteIteratorTest, BaseTimeBasedCandleTest, TestCase):
    window = "1min"

    def test_management_command_stops_at_trade_data_gap(self, mock_get_current_time):
        filtered_1 = self.get_filtered(self.timestamp_from)
        self.write_trade_data(self.timestamp_from, self.one_minute_from_now, filtered_1)
        filtered_2 = self.get_filtered(self.two_minutes_from_now)
        self.write_trade_data(
            self.two_minutes_from_now, self.three_minutes_from_now, filtered_2
        )

        with (
            time_machine.travel(self.three_minutes_from_now, tick=False),
            self.assertLogs("quant_tick.models.candles", level="WARNING") as logs,
        ):
            call_command(
                "candles",
                code_name=[self.candle.code_name],
                date_from="2009-01-04",
                time_from="00:00",
                date_to="2009-01-04",
                time_to="00:03",
            )

        self.assertEqual(
            list(CandleData.objects.values_list("timestamp", flat=True)),
            [self.timestamp_from],
        )
        cache = CandleCache.objects.get(candle=self.candle)
        self.assertEqual(cache.timestamp, self.timestamp_from)
        self.assertEqual(cache.frequency, Frequency.MINUTE)
        self.assertIn("stopped on TradeData gap", logs.output[0])
        self.assertIn(str(self.candle), logs.output[0])
