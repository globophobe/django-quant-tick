from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from unittest.mock import patch

import pandas as pd
from django.test import SimpleTestCase

from quant_tick.exchanges.binance_futures.market_history import (
    MARKET_HISTORY_COLUMNS,
    _fetch_rest_series,
    binance_market_history,
    binance_market_history_archives,
    binance_market_history_rest,
    get_binance_metrics_archive,
)


def market_frame(timestamp: datetime, source: str) -> pd.DataFrame:
    values = {column: None for column in MARKET_HISTORY_COLUMNS}
    values.update(
        {
            "open_interest": Decimal("100"),
            "market_history_interval": "5m",
            "market_history_source": source,
        }
    )
    return pd.DataFrame(
        [values],
        index=pd.DatetimeIndex([timestamp], name="timestamp"),
    )


class BinanceMarketHistoryTest(SimpleTestCase):
    def test_archive_parses_metrics_and_deduplicates_known_early_rows(self):
        raw = pd.DataFrame(
            [
                {
                    "create_time": "create_time",
                    "symbol": "symbol",
                    "sum_open_interest": "sum_open_interest",
                    "sum_open_interest_value": "sum_open_interest_value",
                    "count_toptrader_long_short_ratio": (
                        "count_toptrader_long_short_ratio"
                    ),
                    "sum_toptrader_long_short_ratio": (
                        "sum_toptrader_long_short_ratio"
                    ),
                    "count_long_short_ratio": "count_long_short_ratio",
                    "sum_taker_long_short_vol_ratio": (
                        "sum_taker_long_short_vol_ratio"
                    ),
                },
                {
                    "create_time": "2021-01-01 00:00:00",
                    "symbol": "BTCUSDT",
                    "sum_open_interest": "100",
                    "sum_open_interest_value": "9000000",
                    "count_toptrader_long_short_ratio": "1.1",
                    "sum_toptrader_long_short_ratio": "1.2",
                    "count_long_short_ratio": "1.3",
                    "sum_taker_long_short_vol_ratio": "1.4",
                },
                {
                    "create_time": "2021-01-01 00:00:00",
                    "symbol": "BTCUSDT",
                    "sum_open_interest": "101",
                    "sum_open_interest_value": "9100000",
                    "count_toptrader_long_short_ratio": "1.11",
                    "sum_toptrader_long_short_ratio": "1.21",
                    "count_long_short_ratio": "1.31",
                    "sum_taker_long_short_vol_ratio": "1.41",
                },
            ]
        )

        with patch(
            "quant_tick.exchanges.binance_futures.market_history.zip_downloader",
            return_value=raw,
        ) as mocked:
            result = get_binance_metrics_archive(" btcusdt ", date(2021, 1, 1))

        self.assertEqual(len(result), 1)
        self.assertEqual(result.iloc[0].open_interest, Decimal("101"))
        self.assertEqual(result.iloc[0].open_interest_value, Decimal("9100000"))
        self.assertEqual(
            result.iloc[0].top_trader_long_short_account_ratio,
            Decimal("1.11"),
        )
        self.assertEqual(
            result.iloc[0].top_trader_long_short_position_ratio,
            Decimal("1.21"),
        )
        self.assertEqual(result.iloc[0].long_short_account_ratio, Decimal("1.31"))
        self.assertEqual(
            result.iloc[0].taker_long_short_volume_ratio,
            Decimal("1.41"),
        )
        self.assertEqual(result.iloc[0].open_interest_unit, "base_asset")
        self.assertEqual(result.iloc[0].market_history_source, "data_vision")
        self.assertIn(
            "BTCUSDT/BTCUSDT-metrics-2021-01-01.zip",
            mocked.call_args.args[0],
        )

    def test_archives_walk_newest_to_oldest_and_stop_on_missing_file(self):
        timestamp_from = datetime(2025, 12, 31, tzinfo=UTC)
        timestamp_to = datetime(2026, 1, 4, tzinfo=UTC)
        newest = market_frame(datetime(2026, 1, 3, tzinfo=UTC), "data_vision")
        prior = market_frame(datetime(2026, 1, 2, tzinfo=UTC), "data_vision")

        with patch(
            "quant_tick.exchanges.binance_futures.market_history.get_binance_metrics_archive",
            side_effect=[newest, prior, None],
        ) as mocked:
            result = binance_market_history_archives(
                "BTCUSDT",
                timestamp_from,
                timestamp_to,
            )

        self.assertEqual(
            [item.args[1] for item in mocked.call_args_list],
            [date(2026, 1, 3), date(2026, 1, 2), date(2026, 1, 1)],
        )
        self.assertEqual(
            list(result.index),
            [
                pd.Timestamp("2026-01-02T00:00:00Z"),
                pd.Timestamp("2026-01-03T00:00:00Z"),
            ],
        )

    def test_rest_series_paginates_forward_without_overlap(self):
        timestamp_from = datetime(2026, 1, 1, tzinfo=UTC)
        timestamp_to = timestamp_from + timedelta(minutes=20)
        first = int(timestamp_from.timestamp() * 1000)
        page_one = [
            {"timestamp": first, "sumOpenInterest": "100"},
            {"timestamp": first + 300_000, "sumOpenInterest": "101"},
        ]
        page_two = [
            {"timestamp": first + 600_000, "sumOpenInterest": "102"},
        ]

        with (
            patch(
                "quant_tick.exchanges.binance_futures.market_history."
                "MARKET_HISTORY_MAX_RESULTS",
                2,
            ),
            patch(
                "quant_tick.exchanges.binance_futures.market_history."
                "get_binance_market_history_response",
                side_effect=[page_one, page_two],
            ) as mocked,
        ):
            result = _fetch_rest_series(
                "btcusdt",
                "openInterestHist",
                timestamp_from,
                timestamp_to,
            )

        self.assertEqual(result, [*page_one, *page_two])
        self.assertEqual(mocked.call_count, 2)
        self.assertIn(f"startTime={first + 300_001}", mocked.call_args.args[0])
        self.assertIn("symbol=BTCUSDT", mocked.call_args.args[0])
        self.assertIn("period=5m", mocked.call_args.args[0])

    def test_rest_combines_all_market_history_series(self):
        timestamp = datetime(2026, 1, 1, tzinfo=UTC)
        timestamp_ms = int(timestamp.timestamp() * 1000)
        rows = {
            "openInterestHist": {
                "timestamp": timestamp_ms,
                "sumOpenInterest": "100",
                "sumOpenInterestValue": "9000000",
            },
            "topLongShortAccountRatio": {
                "timestamp": timestamp_ms,
                "longShortRatio": "1.1",
            },
            "topLongShortPositionRatio": {
                "timestamp": timestamp_ms,
                "longShortRatio": "1.2",
            },
            "globalLongShortAccountRatio": {
                "timestamp": timestamp_ms,
                "longShortRatio": "1.3",
            },
            "takerlongshortRatio": {
                "timestamp": timestamp_ms,
                "buySellRatio": "1.4",
            },
        }

        def fetch(api_symbol, endpoint, timestamp_from, timestamp_to):
            self.assertEqual(api_symbol, "BTCUSDT")
            return [rows[endpoint]]

        with patch(
            "quant_tick.exchanges.binance_futures.market_history._fetch_rest_series",
            side_effect=fetch,
        ):
            result = binance_market_history_rest(
                "BTCUSDT",
                timestamp,
                timestamp + timedelta(minutes=5),
            )

        row = result.iloc[0]
        self.assertEqual(row.open_interest, Decimal("100"))
        self.assertEqual(row.open_interest_value, Decimal("9000000"))
        self.assertEqual(row.top_trader_long_short_account_ratio, Decimal("1.1"))
        self.assertEqual(row.top_trader_long_short_position_ratio, Decimal("1.2"))
        self.assertEqual(row.long_short_account_ratio, Decimal("1.3"))
        self.assertEqual(row.taker_long_short_volume_ratio, Decimal("1.4"))
        self.assertEqual(row.market_history_interval, "5m")
        self.assertEqual(row.market_history_source, "rest")

    def test_missing_recent_archive_uses_rest_before_older_archive(self):
        timestamp_from = datetime(2026, 1, 3, tzinfo=UTC)
        timestamp_to = datetime(2026, 1, 5, tzinfo=UTC)
        archived = market_frame(timestamp_from, "data_vision")
        recent = market_frame(datetime(2026, 1, 4, tzinfo=UTC), "rest")

        with (
            patch(
                "quant_tick.exchanges.binance_futures.market_history."
                "get_current_time",
                return_value=timestamp_to,
            ),
            patch(
                "quant_tick.exchanges.binance_futures.market_history."
                "get_binance_metrics_archive",
                side_effect=[None, archived],
            ) as archive_mock,
            patch(
                "quant_tick.exchanges.binance_futures.market_history."
                "binance_market_history_rest",
                return_value=recent,
            ) as rest_mock,
        ):
            result = binance_market_history(
                "BTCUSDT",
                timestamp_from,
                timestamp_to,
            )

        self.assertEqual(
            [call.args[1] for call in archive_mock.call_args_list],
            [date(2026, 1, 4), date(2026, 1, 3)],
        )
        rest_mock.assert_called_once_with(
            "BTCUSDT",
            datetime(2026, 1, 4, tzinfo=UTC),
            timestamp_to,
        )
        self.assertEqual(
            list(result.market_history_source),
            ["data_vision", "rest"],
        )
