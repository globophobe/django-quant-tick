from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pandas as pd
from django.test import TestCase

from quant_tick.constants import Exchange, SymbolType
from quant_tick.models import PerpetualStatsData

from ..base import BaseSymbolTest


class PerpetualStatsDataTest(BaseSymbolTest, TestCase):
    def test_write_upserts_returned_timestamps_without_deleting_others(self):
        symbol = self.get_symbol(
            exchange=Exchange.BINANCE_FUTURES,
            api_symbol="BTCUSDT",
            symbol_type=SymbolType.PERPETUAL,
        )
        timestamp_from = datetime(2026, 4, 25, tzinfo=UTC)
        timestamp_to = timestamp_from + timedelta(minutes=10)
        data = pd.DataFrame(
            [
                {
                    "timestamp": timestamp_from,
                    "open_interest": Decimal(100),
                    "long_short_account_ratio": Decimal("1.2"),
                    "open_interest_unit": "base_asset",
                    "market_history_source": "data_vision",
                },
                {
                    "timestamp": timestamp_from + timedelta(minutes=5),
                    "open_interest": Decimal(101),
                    "long_short_account_ratio": Decimal("1.3"),
                    "open_interest_unit": "base_asset",
                    "market_history_source": "data_vision",
                },
            ]
        )

        PerpetualStatsData.write(symbol, 5, timestamp_from, timestamp_to, data)
        PerpetualStatsData.write(
            symbol,
            5,
            timestamp_from,
            timestamp_to,
            data.iloc[:1].assign(open_interest=Decimal(102)),
        )

        rows = list(PerpetualStatsData.objects.filter(symbol=symbol))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].timestamp, timestamp_from)
        self.assertEqual(rows[0].open_interest, Decimal(102))
        self.assertEqual(rows[0].long_short_account_ratio, Decimal("1.2"))
        self.assertEqual(rows[0].open_interest_unit, "base_asset")
        self.assertEqual(rows[0].json_data["market_history_source"], "data_vision")
        self.assertEqual(rows[0].coverage_end(), timestamp_from + timedelta(minutes=5))
        self.assertEqual(
            rows[1].timestamp,
            timestamp_from + timedelta(minutes=5),
        )
        self.assertEqual(rows[1].open_interest, Decimal(101))

    def test_write_rejects_lost_lease_before_replacing_rows(self):
        symbol = self.get_symbol(
            exchange=Exchange.BINANCE_FUTURES,
            api_symbol="BTCUSDT",
            symbol_type=SymbolType.PERPETUAL,
        )
        timestamp_from = datetime(2026, 4, 25, tzinfo=UTC)
        timestamp_to = timestamp_from + timedelta(minutes=5)
        original = pd.DataFrame(
            [{"timestamp": timestamp_from, "open_interest": Decimal(100)}]
        )
        replacement = pd.DataFrame(
            [{"timestamp": timestamp_from, "open_interest": Decimal(101)}]
        )
        PerpetualStatsData.write(
            symbol,
            5,
            timestamp_from,
            timestamp_to,
            original,
        )

        def reject_lease():
            type(symbol).objects.filter(pk=symbol.pk).update(api_symbol="stale-write")
            raise RuntimeError("lease ownership lost")

        with self.assertRaisesRegex(RuntimeError, "ownership lost"):
            PerpetualStatsData.write(
                symbol,
                5,
                timestamp_from,
                timestamp_to,
                replacement,
                assert_lease_owned=reject_lease,
            )

        symbol.refresh_from_db()
        self.assertEqual(symbol.api_symbol, "BTCUSDT")
        self.assertEqual(
            PerpetualStatsData.objects.get(
                symbol=symbol,
                timestamp=timestamp_from,
                frequency=5,
            ).open_interest,
            Decimal(100),
        )

    def test_write_rejects_spot_symbol(self):
        symbol = self.get_symbol(symbol_type=SymbolType.SPOT)
        timestamp_from = datetime(2026, 4, 25, tzinfo=UTC)
        data = pd.DataFrame(
            [{"timestamp": timestamp_from, "open_interest": Decimal(100)}]
        )

        with self.assertRaisesRegex(ValueError, "only for perpetuals"):
            PerpetualStatsData.write(
                symbol,
                5,
                timestamp_from,
                timestamp_from + timedelta(minutes=5),
                data,
            )
