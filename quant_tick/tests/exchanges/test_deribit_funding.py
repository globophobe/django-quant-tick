from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import call, patch

import pandas as pd
from django.test import SimpleTestCase

from quant_tick.constants import Exchange, SymbolType
from quant_tick.exchanges.api import funding_api
from quant_tick.exchanges.deribit.funding import DeribitFunding, deribit_funding


class DeribitFundingTest(SimpleTestCase):
    def test_deribit_funding_normalizes_hourly_rate_and_preserves_audit_fields(self):
        timestamp_from = datetime(2026, 4, 25, tzinfo=UTC)
        timestamp_to = timestamp_from + timedelta(hours=2)
        response = [
            {
                "timestamp": int(timestamp_from.timestamp() * 1000),
                "interest_1h": "0.000001",
                "interest_8h": "0.000008",
                "index_price": "95000.5",
                "prev_index_price": None,
            }
        ]

        with patch(
            "quant_tick.exchanges.deribit.funding.get_deribit_funding_response",
            return_value=response,
        ) as mocked:
            result = deribit_funding(
                "BTC-PERPETUAL",
                timestamp_from,
                timestamp_to,
            )

        mocked.assert_called_once_with(
            "BTC-PERPETUAL",
            int(timestamp_from.timestamp() * 1000) - 1,
            int(timestamp_to.timestamp() * 1000) - 1,
        )
        self.assertEqual(list(result.index), [pd.Timestamp(timestamp_from)])
        row = result.iloc[0]
        self.assertEqual(row.funding_rate, Decimal("0.000001"))
        self.assertEqual(row.interest_8h, Decimal("0.000008"))
        self.assertEqual(row.index_price, Decimal("95000.5"))
        self.assertIsNone(row.prev_index_price)

    def test_deribit_funding_uses_non_overlapping_30_day_windows(self):
        timestamp_from = datetime(2026, 1, 1, tzinfo=UTC)
        timestamp_to = timestamp_from + timedelta(days=31)

        with patch(
            "quant_tick.exchanges.deribit.funding.get_deribit_funding_response",
            side_effect=[[], []],
        ) as mocked:
            result = deribit_funding(
                "BTC-PERPETUAL",
                timestamp_from,
                timestamp_to,
            )

        start_ms = int(timestamp_from.timestamp() * 1000)
        first_end_ms = int((timestamp_from + timedelta(days=30)).timestamp() * 1000) - 1
        self.assertEqual(
            mocked.call_args_list,
            [
                call("BTC-PERPETUAL", start_ms - 1, first_end_ms),
                call(
                    "BTC-PERPETUAL",
                    first_end_ms,
                    int(timestamp_to.timestamp() * 1000) - 1,
                ),
            ],
        )
        self.assertTrue(result.empty)

    def test_pre_history_deribit_funding_is_not_expected(self):
        self.assertEqual(
            DeribitFunding.expected_timestamps(
                datetime(2019, 4, 30, 8, tzinfo=UTC),
                datetime(2019, 4, 30, 12, tzinfo=UTC),
            ),
            [
                datetime(2019, 4, 30, 10, tzinfo=UTC),
                datetime(2019, 4, 30, 11, tzinfo=UTC),
            ],
        )

    def test_known_deribit_history_gap_is_not_expected(self):
        timestamp_from = datetime(2020, 8, 27, 5, tzinfo=UTC)
        timestamp_to = datetime(2020, 8, 27, 9, tzinfo=UTC)

        self.assertEqual(
            DeribitFunding.expected_timestamps(timestamp_from, timestamp_to),
            [
                datetime(2020, 8, 27, 5, tzinfo=UTC),
                datetime(2020, 8, 27, 8, tzinfo=UTC),
            ],
        )

    def test_funding_api_dispatches_deribit_perpetuals(self):
        symbol = SimpleNamespace(
            exchange=Exchange.DERIBIT,
            api_symbol="BTC-PERPETUAL",
            symbol_type=SymbolType.PERPETUAL,
            clamp_timestamp_range=lambda ts_from, ts_to: (ts_from, ts_to),
        )
        timestamp_from = datetime(2026, 4, 25, tzinfo=UTC)
        timestamp_to = datetime(2026, 4, 26, tzinfo=UTC)
        expected = pd.DataFrame([])

        with (
            patch(
                "quant_tick.exchanges.api.deribit_funding",
                return_value=expected,
            ) as mocked,
            patch("quant_tick.exchanges.api.refresh_funding_interval") as refresh,
        ):
            result = funding_api(symbol, timestamp_from, timestamp_to)

        refresh.assert_called_once_with(symbol)
        mocked.assert_called_once_with(
            "BTC-PERPETUAL",
            timestamp_from,
            timestamp_to,
        )
        self.assertTrue(result.equals(expected))
