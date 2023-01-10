from datetime import datetime, time, timezone
from typing import List

import pandas as pd
from django.test import SimpleTestCase

from quant_candles.lib import (
    get_current_time,
    get_min_time,
    get_next_time,
    get_range,
    iter_missing,
    iter_timeframe,
    iter_window,
    parse_period_from_to,
)


class GetMinTimeTest(SimpleTestCase):
    def test_get_min_time_1d(self):
        """Get start of current day."""
        now = get_current_time()
        min_time = get_min_time(now, value="1d")
        self.assertEqual(
            min_time,
            datetime.combine(min_time.date(), time.min).replace(tzinfo=timezone.utc),
        )


class GetNexttimeTest(SimpleTestCase):
    def test_get_next_minute(self):
        """Get start of next day."""
        now = get_current_time()
        tomorrow = get_next_time(now, value="1d")
        self.assertEqual(
            tomorrow,
            datetime.combine(tomorrow.date(), time.min).replace(tzinfo=timezone.utc),
        )


class GetRangeTest(SimpleTestCase):
    def setUp(self):
        now = get_current_time()
        self.timestamp_from = get_min_time(now, value="1d")

    def test_get_range_1m(self):
        """Get range, by 1 minute."""
        one_minute = pd.Timedelta("1t")
        timestamp_to = get_next_time(self.timestamp_from, value="1d") - one_minute
        values = get_range(self.timestamp_from, timestamp_to)
        self.assertEqual(len(values), 1440)
        self.assertEqual(values[0], self.timestamp_from)
        self.assertEqual(values[-1], timestamp_to)

    def test_get_range_1d(self):
        """Get range, by 1 day."""
        values = get_range(self.timestamp_from, self.timestamp_from, value="1d")
        self.assertEqual(len(values), 1)
        self.assertEqual(values[0], self.timestamp_from)


class IterWindowTest(SimpleTestCase):
    def setUp(self):
        one_day = pd.Timedelta("1d")
        self.now = get_current_time()
        self.yesterday = self.now - one_day
        self.two_days_ago = self.yesterday - one_day

    def test_iter_window(self):
        """Iter window by days."""
        values = [
            value for value in iter_window(self.two_days_ago, self.now, value="1d")
        ]
        self.assertEqual(len(values), 2)
        self.assertEqual(values[0][0], get_min_time(self.two_days_ago, value="1d"))
        self.assertEqual(values[1][1], get_min_time(self.now, value="1d"))

    def test_iter_window_reverse(self):
        """Iter window by days, in reverse."""
        values = [
            value
            for value in iter_window(
                self.two_days_ago, self.now, value="1d", reverse=True
            )
        ]
        self.assertEqual(len(values), 2)
        self.assertEqual(values[0][1], get_min_time(self.now, value="1d"))
        self.assertEqual(values[1][0], get_min_time(self.two_days_ago, value="1d"))


class IterTimeframeTest(SimpleTestCase):
    def setUp(self):
        self.now = get_current_time()

    def get_values(
        self, timestamp_from: datetime, timestamp_to: datetime, value: str = "1d"
    ) -> List[tuple]:
        """Get values for timeframe."""
        return [
            value
            for value in iter_timeframe(
                timestamp_from, timestamp_to, value=value, reverse=True
            )
        ]

    def test_iter_timeframe_with_head_and_no_body(self):
        """Current day only."""
        date_from = self.now.date().isoformat()
        time_from = self.now.time().isoformat()
        timestamp_from, timestamp_to = parse_period_from_to(
            date_from=date_from, time_from=time_from
        )
        values = self.get_values(timestamp_from, timestamp_to)
        self.assertEqual(len(values), 1)
        ts_from, ts_to = values[0]
        self.assertEqual(ts_from, get_min_time(timestamp_from, "1t"))
        self.assertEqual(ts_to, get_min_time(timestamp_to, "1t"))

    def test_iter_timeframe_with_tail_only(self):
        """Previous day only."""
        yesterday = self.now - pd.Timedelta("1d")
        date_from = yesterday.date().isoformat()
        time_from = yesterday.time().isoformat()
        date_to = self.now.date().isoformat()
        timestamp_from, timestamp_to = parse_period_from_to(
            date_from=date_from, time_from=time_from, date_to=date_to
        )
        values = self.get_values(timestamp_from, timestamp_to)
        self.assertEqual(len(values), 1)

    def test_iter_timeframe_with_head(self):
        """1 min after midnight yesterday, until today."""
        yesterday = get_min_time(self.now, "2d") + pd.Timedelta("1m")
        today = get_min_time(self.now, "1d")
        timestamp_from, timestamp_to = parse_period_from_to(
            date_from=yesterday.date().isoformat(),
            time_from=yesterday.time().isoformat(),
            date_to=today.date().isoformat(),
            time_to=today.time().isoformat(),
        )
        values = self.get_values(timestamp_from, timestamp_to)
        target = len(values) - 1
        for index, value in enumerate(values):
            ts_from, ts_to = value
            if index != target:
                self.assertEqual(ts_from + pd.Timedelta("1d"), ts_to)
            else:
                self.assertEqual(ts_from, timestamp_from)
                self.assertEqual(ts_to, get_min_time(ts_to, value="1d"))

    def test_iter_timeframe_with_neither_head_nor_tail(self):
        """Two days ago until yesterday."""
        yesterday = self.now - pd.Timedelta("1d")
        two_days_ago = self.now - pd.Timedelta("2d")
        timestamp_from, timestamp_to = parse_period_from_to(
            date_from=two_days_ago.date().isoformat(),
            date_to=yesterday.date().isoformat(),
        )
        values = self.get_values(timestamp_from, timestamp_to)
        self.assertEqual(len(values), 1)
        self.assertEqual(values[0][0], timestamp_from)
        self.assertEqual(values[0][1], timestamp_to)

    def test_iter_timeframe_with_tail(self):
        """Yesterday, 1 minute to midnight."""
        timestamp = get_min_time(self.now, "1d") - pd.Timedelta("1m")
        timestamp_from, timestamp_to = parse_period_from_to(
            date_to=timestamp.date().isoformat(), time_to=timestamp.time().isoformat()
        )
        values = self.get_values(timestamp_from, timestamp_to)
        for index, value in enumerate(values):
            ts_from, ts_to = value
            if index == 0:
                self.assertEqual(ts_from, get_min_time(ts_from, value="1d"))
                self.assertEqual(ts_to, timestamp_to)
            else:
                self.assertEqual(ts_from + pd.Timedelta("1d"), ts_to)


class IterMissingTest(SimpleTestCase):
    def setUp(self):
        self.one_minute = pd.Timedelta("1t")
        self.timestamp_from = get_min_time(get_current_time(), "1d")
        self.timestamp_to = self.timestamp_from + (self.one_minute * 5)
        self.timestamps = get_range(self.timestamp_from, self.timestamp_to)

    def test_iter_missing_with_no_missing(self):
        """No missing timestamps."""
        values = [
            value for value in iter_missing(self.timestamp_from, self.timestamp_to, [])
        ]
        self.assertEqual(len(values), 1)
        self.assertEqual(values[0][0], self.timestamp_from)
        self.assertEqual(values[-1][1], self.timestamp_to)

    def test_iter_missing_with_head(self):
        """First timestamp is OK."""
        existing = self.timestamps[0]
        values = [
            value
            for value in iter_missing(
                self.timestamp_from, self.timestamp_to, [existing]
            )
        ]
        self.assertEqual(len(values), 1)
        self.assertEqual(values[0][0], self.timestamp_from + self.one_minute)
        self.assertEqual(values[-1][1], self.timestamp_to)

    def test_iter_missing_with_one_timestamp_ok(self):
        """Second timestamp is OK."""
        existing = self.timestamps[1]
        values = [
            value
            for value in iter_missing(
                self.timestamp_from, self.timestamp_to, [existing]
            )
        ]
        self.assertEqual(len(values), 2)
        self.assertEqual(values[0][0], self.timestamp_from)
        self.assertEqual(values[0][1], existing)
        self.assertEqual(values[-1][0], existing + self.one_minute)
        self.assertEqual(values[-1][1], self.timestamp_to)

    def test_iter_missing_with_two_timestamps_ok(self):
        """Second and fourth timestamps are OK."""
        existing_one = self.timestamps[1]
        existing_two = self.timestamps[3]
        values = [
            value
            for value in iter_missing(
                self.timestamp_from, self.timestamp_to, [existing_one, existing_two]
            )
        ]
        self.assertEqual(len(values), 3)
        self.assertEqual(values[0][0], self.timestamp_from)
        self.assertEqual(values[0][1], existing_one)
        self.assertEqual(values[1][0], existing_one + self.one_minute)
        self.assertEqual(values[1][1], existing_two)
        self.assertEqual(values[-1][0], existing_two + self.one_minute)
        self.assertEqual(values[-1][1], self.timestamp_to)

    def test_iter_missing_with_tail(self):
        """Last timestamp is OK."""
        existing = self.timestamps[-1]
        values = [
            value
            for value in iter_missing(
                self.timestamp_from, self.timestamp_to, [existing]
            )
        ]
        self.assertEqual(len(values), 1)
        self.assertEqual(values[0][0], self.timestamp_from)
        self.assertEqual(values[0][1], self.timestamp_to)
