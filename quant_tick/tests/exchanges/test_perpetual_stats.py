from datetime import UTC, datetime, timedelta
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import call, patch

import pandas as pd
from django.test import SimpleTestCase, TestCase

from quant_tick.constants import Exchange, SymbolType
from quant_tick.exchanges.binance_futures.market_history import (
    HISTORY_EXHAUSTED_ATTR,
)
from quant_tick.exchanges.perpetual_stats import (
    perpetual_stats,
    perpetual_stats_api,
    perpetual_stats_frequency,
)
from quant_tick.models import PerpetualStatsData

from ..base import BaseSymbolTest


def binance_market_row(timestamp: datetime, index: int = 0) -> dict:
    return {
        "timestamp": timestamp,
        "open_interest": Decimal(str(100 + index)),
        "open_interest_value": Decimal(str(1_000_000 + index)),
        "top_trader_long_short_account_ratio": Decimal("1.1"),
        "top_trader_long_short_position_ratio": Decimal("1.2"),
        "long_short_account_ratio": Decimal("1.3"),
        "taker_long_short_volume_ratio": Decimal("1.4"),
    }


class PerpetualStatsAdapterTest(SimpleTestCase):
    def test_dispatches_bybit_identity_to_matching_category(self):
        timestamp_from = datetime(2026, 4, 25, tzinfo=UTC)
        timestamp_to = timestamp_from + timedelta(hours=4)
        expected = pd.DataFrame()
        symbol = SimpleNamespace(
            exchange=Exchange.BYBIT_INVERSE,
            api_symbol="BTCUSD",
            symbol_type=SymbolType.PERPETUAL,
        )

        with patch(
            "quant_tick.exchanges.perpetual_stats.bybit_market_history",
            return_value=expected,
        ) as mocked:
            result = perpetual_stats_api(
                symbol,
                timestamp_from,
                timestamp_to,
            )

        self.assertIs(result, expected)
        mocked.assert_called_once_with(
            "BTCUSD",
            timestamp_from,
            timestamp_to,
            category="inverse",
        )
        self.assertEqual(perpetual_stats_frequency(symbol), 240)

    def test_rejects_spot_symbol(self):
        symbol = SimpleNamespace(
            exchange=Exchange.BYBIT,
            api_symbol="BTCUSDT",
            symbol_type=SymbolType.SPOT,
        )

        with self.assertRaisesRegex(ValueError, "only available for perpetuals"):
            perpetual_stats_frequency(symbol)


class PerpetualStatsCollectionTest(BaseSymbolTest, TestCase):
    def test_empty_history_stops_backfill_unless_retry(self):
        timestamp_to = datetime(2026, 4, 25, tzinfo=UTC)
        timestamp_from = timestamp_to - timedelta(days=270)
        for exchange, api_symbol in (
            (Exchange.BINANCE_FUTURES, "BTCUSDT"),
            (Exchange.BYBIT_LINEAR, "BTCUSDT"),
            (Exchange.BYBIT_INVERSE, "BTCUSD"),
            (Exchange.PHOENIX, "BTC"),
        ):
            symbol = self.get_symbol(
                exchange=exchange,
                api_symbol=api_symbol,
                symbol_type=SymbolType.PERPETUAL,
            )
            for retry in (False, True):
                with self.subTest(exchange=exchange, retry=retry):
                    with patch(
                        "quant_tick.exchanges.perpetual_stats.perpetual_stats_api",
                        return_value=pd.DataFrame(),
                    ) as mocked:
                        perpetual_stats(
                            symbol, timestamp_from, timestamp_to, retry=retry
                        )

                    expected = [
                        call(
                            symbol,
                            timestamp_to - timedelta(days=90 * (index + 1)),
                            timestamp_to - timedelta(days=90 * index),
                        )
                        for index in range(3 if retry else 1)
                    ]
                    self.assertEqual(mocked.call_args_list, expected)
                    self.assertFalse(
                        PerpetualStatsData.objects.filter(symbol=symbol).exists()
                    )

    def test_empty_gaps_preserve_backfill_with_complete_or_partial_history(self):
        symbol = self.get_symbol(
            exchange=Exchange.BYBIT_LINEAR,
            api_symbol="BTCUSDT",
            symbol_type=SymbolType.PERPETUAL,
        )
        timestamp_to = datetime(2026, 4, 25, tzinfo=UTC)
        timestamp_from = timestamp_to - timedelta(days=270)
        values = {
            "open_interest": Decimal(100),
            "single_open_interest": Decimal(50),
            "long_account_ratio": Decimal("0.6"),
            "short_account_ratio": Decimal("0.4"),
        }
        returned_timestamp = timestamp_from + timedelta(days=90, hours=4)
        frame = pd.DataFrame(
            [{"timestamp": returned_timestamp, **values}]
        ).set_index("timestamp")

        def fetch(_symbol, start, end):
            return frame if start <= returned_timestamp < end else pd.DataFrame()

        for age_days in (90, 180):
            for complete in (False, True):
                with self.subTest(age_days=age_days, complete=complete):
                    PerpetualStatsData.objects.filter(symbol=symbol).delete()
                    stored_values = values | {
                        "short_account_ratio": Decimal("0.4") if complete else None
                    }
                    stored = PerpetualStatsData.objects.create(
                        symbol=symbol,
                        frequency=240,
                        timestamp=timestamp_to - timedelta(days=age_days),
                        **stored_values,
                    )
                    with patch(
                        "quant_tick.exchanges.perpetual_stats.perpetual_stats_api",
                        side_effect=fetch,
                    ) as mocked:
                        perpetual_stats(symbol, timestamp_from, timestamp_to)

                    self.assertEqual(mocked.call_count, 3)
                    self.assertEqual(
                        mocked.call_args,
                        call(
                            symbol, timestamp_from, timestamp_from + timedelta(days=90)
                        ),
                    )
                    self.assertEqual(
                        list(
                            PerpetualStatsData.objects.filter(
                                symbol=symbol
                            ).values_list("timestamp", flat=True)
                        ),
                        sorted([returned_timestamp, stored.timestamp]),
                    )
                    stored.refresh_from_db()
                    self.assertEqual(stored.open_interest, Decimal(100))
                    self.assertEqual(
                        stored.short_account_ratio,
                        Decimal("0.4") if complete else None,
                    )

    def test_collects_complete_rows_and_repairs_partial_observations(self):
        symbol = self.get_symbol(
            exchange=Exchange.BINANCE_FUTURES,
            api_symbol="BTCUSDT",
            symbol_type=SymbolType.PERPETUAL,
        )
        timestamp_from = datetime(2026, 4, 25, tzinfo=UTC)
        timestamp_to = timestamp_from + timedelta(hours=1)
        expected_timestamps = [
            timestamp_from + timedelta(minutes=5 * index) for index in range(12)
        ]
        frame = pd.DataFrame(
            [
                binance_market_row(timestamp, index)
                for index, timestamp in enumerate(expected_timestamps)
            ]
        ).set_index("timestamp")
        missing_timestamp = expected_timestamps[-1]
        partial_frame = frame.copy()
        partial_frame.loc[
            missing_timestamp,
            "top_trader_long_short_account_ratio",
        ] = None

        with patch(
            "quant_tick.exchanges.perpetual_stats.perpetual_stats_api",
            return_value=partial_frame,
        ) as mocked:
            perpetual_stats(symbol, timestamp_from, timestamp_to)

        mocked.assert_called_once_with(symbol, timestamp_from, timestamp_to)
        self.assertEqual(
            list(
                PerpetualStatsData.objects.filter(symbol=symbol).values_list(
                    "timestamp",
                    flat=True,
                )
            ),
            expected_timestamps,
        )
        self.assertEqual(
            PerpetualStatsData.objects.complete_for_exchange(
                symbol.exchange
            ).count(),
            11,
        )

        with patch(
            "quant_tick.exchanges.perpetual_stats.perpetual_stats_api",
            return_value=frame.loc[[missing_timestamp]],
        ) as mocked:
            perpetual_stats(symbol, timestamp_from, timestamp_to)
        mocked.assert_called_once_with(
            symbol,
            missing_timestamp,
            missing_timestamp + timedelta(minutes=5),
        )
        self.assertEqual(
            PerpetualStatsData.objects.filter(symbol=symbol).count(),
            12,
        )

        with patch(
            "quant_tick.exchanges.perpetual_stats.perpetual_stats_api",
            return_value=partial_frame,
        ):
            perpetual_stats(
                symbol,
                timestamp_from,
                timestamp_to,
                retry=True,
            )
        self.assertEqual(
            PerpetualStatsData.objects.complete_for_exchange(
                symbol.exchange
            ).count(),
            12,
        )

        legacy_partial_timestamp = expected_timestamps[1]
        PerpetualStatsData.objects.filter(
            symbol=symbol,
            timestamp=legacy_partial_timestamp,
        ).update(taker_long_short_volume_ratio=None)
        with patch(
            "quant_tick.exchanges.perpetual_stats.perpetual_stats_api",
            return_value=frame.loc[[legacy_partial_timestamp]],
        ) as mocked:
            perpetual_stats(symbol, timestamp_from, timestamp_to)

        mocked.assert_called_once_with(
            symbol,
            legacy_partial_timestamp,
            legacy_partial_timestamp + timedelta(minutes=5),
        )
        self.assertEqual(
            PerpetualStatsData.objects.complete_for_exchange(
                symbol.exchange
            ).count(),
            12,
        )

        with patch(
            "quant_tick.exchanges.perpetual_stats.perpetual_stats_api"
        ) as mocked:
            perpetual_stats(symbol, timestamp_from, timestamp_to)
        mocked.assert_not_called()

    def test_bybit_waits_for_both_market_history_endpoints(self):
        symbol = self.get_symbol(
            exchange=Exchange.BYBIT_LINEAR,
            api_symbol="BTCUSDT",
            symbol_type=SymbolType.PERPETUAL,
        )
        timestamp_from = datetime(2026, 4, 25, tzinfo=UTC)
        timestamp_to = timestamp_from + timedelta(hours=4)
        frame = pd.DataFrame(
            [
                {
                    "timestamp": timestamp_from,
                    "open_interest": Decimal(100),
                    "single_open_interest": Decimal(50),
                    "long_account_ratio": Decimal("0.6"),
                    "short_account_ratio": Decimal("0.4"),
                    "long_short_account_ratio": Decimal("1.5"),
                }
            ]
        ).set_index("timestamp")
        partial_frame = frame.copy()
        partial_frame.loc[timestamp_from, "short_account_ratio"] = None

        with patch(
            "quant_tick.exchanges.perpetual_stats.perpetual_stats_api",
            return_value=partial_frame,
        ):
            perpetual_stats(symbol, timestamp_from, timestamp_to)
        self.assertEqual(
            PerpetualStatsData.objects.filter(symbol=symbol).count(),
            1,
        )
        self.assertFalse(
            PerpetualStatsData.objects.complete_for_exchange(
                symbol.exchange
            ).exists()
        )

        with patch(
            "quant_tick.exchanges.perpetual_stats.perpetual_stats_api",
            return_value=frame,
        ) as mocked:
            perpetual_stats(symbol, timestamp_from, timestamp_to)
        mocked.assert_called_once_with(symbol, timestamp_from, timestamp_to)
        self.assertEqual(
            PerpetualStatsData.objects.complete_for_exchange(
                symbol.exchange
            ).count(),
            1,
        )

    def test_bybit_collects_boundary_snapshot_before_next_boundary(self):
        symbol = self.get_symbol(
            exchange=Exchange.BYBIT_LINEAR,
            api_symbol="BTCUSDT",
            symbol_type=SymbolType.PERPETUAL,
        )
        timestamp_from = datetime(2026, 4, 25, tzinfo=UTC)
        requested_to = timestamp_from + timedelta(hours=2, minutes=52)
        frame = pd.DataFrame(
            [
                {
                    "timestamp": timestamp_from,
                    "open_interest": Decimal(100),
                    "single_open_interest": Decimal(50),
                    "long_account_ratio": Decimal("0.6"),
                    "short_account_ratio": Decimal("0.4"),
                    "long_short_account_ratio": Decimal("1.5"),
                }
            ]
        ).set_index("timestamp")

        with patch(
            "quant_tick.exchanges.perpetual_stats.perpetual_stats_api",
            return_value=frame,
        ) as mocked:
            perpetual_stats(symbol, timestamp_from, requested_to)

        mocked.assert_called_once_with(symbol, timestamp_from, requested_to)
        self.assertEqual(
            list(
                PerpetualStatsData.objects.filter(symbol=symbol).values_list(
                    "timestamp",
                    flat=True,
                )
            ),
            [timestamp_from],
        )

    def test_aligns_partial_range_to_native_grid_and_complete_hour(self):
        symbol = self.get_symbol(
            exchange=Exchange.BINANCE_FUTURES,
            api_symbol="BTCUSDT",
            symbol_type=SymbolType.PERPETUAL,
        )
        requested_from = datetime(2026, 4, 25, 0, 2, tzinfo=UTC)
        requested_to = datetime(2026, 4, 25, 1, 12, tzinfo=UTC)
        expected_from = datetime(2026, 4, 25, 0, 5, tzinfo=UTC)
        expected_to = datetime(2026, 4, 25, 1, 0, tzinfo=UTC)

        with patch(
            "quant_tick.exchanges.perpetual_stats.perpetual_stats_api",
            return_value=pd.DataFrame(),
        ) as mocked:
            perpetual_stats(
                symbol,
                requested_from,
                requested_to,
            )

        mocked.assert_called_once_with(
            symbol,
            expected_from,
            expected_to,
        )

    def test_defers_subhour_observations_until_hour_close(self):
        symbol = self.get_symbol(
            exchange=Exchange.BINANCE_FUTURES,
            api_symbol="BTCUSDT",
            symbol_type=SymbolType.PERPETUAL,
        )
        timestamp_from = datetime(2026, 4, 25, tzinfo=UTC)
        timestamp_to = timestamp_from + timedelta(minutes=59)

        with patch(
            "quant_tick.exchanges.perpetual_stats.perpetual_stats_api"
        ) as mocked:
            perpetual_stats(
                symbol,
                timestamp_from,
                timestamp_to,
            )

        mocked.assert_not_called()

    def test_history_exhaustion_writes_partial_frame_and_stops_older_chunks(self):
        symbol = self.get_symbol(
            exchange=Exchange.BINANCE_FUTURES,
            api_symbol="BTCUSDT",
            symbol_type=SymbolType.PERPETUAL,
        )
        timestamp_from = datetime(2020, 1, 1, tzinfo=UTC)
        timestamp_to = datetime(2020, 7, 1, tzinfo=UTC)
        newest_chunk_from = timestamp_to - timedelta(days=90)
        frame = pd.DataFrame(
            [binance_market_row(newest_chunk_from)]
        ).set_index("timestamp")
        frame.attrs[HISTORY_EXHAUSTED_ATTR] = True

        with patch(
            "quant_tick.exchanges.perpetual_stats.perpetual_stats_api",
            return_value=frame,
        ) as mocked:
            perpetual_stats(
                symbol,
                timestamp_from,
                timestamp_to,
                retry=True,
            )

        mocked.assert_called_once_with(symbol, newest_chunk_from, timestamp_to)
        self.assertEqual(
            list(
                PerpetualStatsData.objects.filter(symbol=symbol).values_list(
                    "timestamp",
                    flat=True,
                )
            ),
            [newest_chunk_from],
        )
