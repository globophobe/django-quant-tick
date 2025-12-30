from datetime import datetime, timezone
from decimal import Decimal

import pandas as pd
from django.test import SimpleTestCase

from quant_tick.lib.meta import build_event_dataset


class DummyCandle:
    """Minimal stub exposing get_candle_data."""

    def __init__(self, df: pd.DataFrame):
        self._df = df

    def get_candle_data(self, *args, **kwargs) -> pd.DataFrame:
        return self._df.copy()


class MetaBuilderTest(SimpleTestCase):
    """Tests for the meta dataset builder."""

    def _make_df(self, directions: list[int], closes: list[Decimal]) -> pd.DataFrame:
        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        rows = []
        for i, (d, c) in enumerate(zip(directions, closes, strict=True)):
            rows.append(
                {
                    "timestamp": base.replace(minute=i),
                    "renko_kind": "body",
                    "renko_direction": d,
                    "close": Decimal(str(c)),
                }
            )
        return pd.DataFrame(rows)

    def test_builder_creates_events_with_exit(self):
        """Confirmed entries (2 bodies) and exit on next reversal are captured."""
        # Directions: + + | - - | + +
        df = self._make_df(
            directions=[1, 1, -1, -1, 1, 1],
            closes=[100, 101, 99, 98, 99, 100],
        )
        candle = DummyCandle(df)

        events = build_event_dataset(
            candle,
            timestamp_from=df["timestamp"].min(),
            timestamp_to=df["timestamp"].max(),
            cost=0,
            include_incomplete=False,
        )

        self.assertEqual(len(events), 1)
        evt = events.iloc[0]

        # Entry is second body of the down run (index 3), exit is first body of next up run (index 4)
        self.assertEqual(evt["direction"], -1)
        self.assertEqual(evt["entry_price"], Decimal("98"))
        self.assertEqual(evt["exit_price"], Decimal("99"))
        expected_gross = -1 * (Decimal("99") / Decimal("98") - 1)
        self.assertAlmostEqual(float(evt["gross_return"]), float(expected_gross))
        self.assertEqual(evt["run_length_prev"], 2)

    def test_includes_incomplete_when_requested(self):
        """include_incomplete=True keeps last event without an exit."""
        # Directions: + + | - -
        df = self._make_df(
            directions=[1, 1, -1, -1],
            closes=[100, 101, 99, 98],
        )
        candle = DummyCandle(df)

        events = build_event_dataset(
            candle,
            timestamp_from=df["timestamp"].min(),
            timestamp_to=df["timestamp"].max(),
            cost=0,
            include_incomplete=True,
        )

        self.assertEqual(len(events), 1)
        evt = events.iloc[0]
        self.assertEqual(evt["direction"], -1)
        self.assertIsNone(evt["exit_price"])
        self.assertIsNone(evt["net_return"])
        self.assertIsNone(evt["label"])
