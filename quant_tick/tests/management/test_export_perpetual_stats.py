from datetime import UTC, datetime, timedelta
from decimal import Decimal
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd
from django.core.management import call_command
from django.test import TestCase

from quant_tick.constants import Exchange, SymbolType
from quant_tick.management.commands.export_perpetual_stats import get_output_path
from quant_tick.models import PerpetualStatsData

from ..base import BaseSymbolTest


class ExportPerpetualStatsCommandTest(BaseSymbolTest, TestCase):
    def setUp(self):
        super().setUp()
        self.symbol = self.get_symbol(
            exchange=Exchange.BINANCE_FUTURES,
            api_symbol="BTCUSDT",
            symbol_type=SymbolType.PERPETUAL,
        )
        self.timestamp = datetime(2020, 9, 1, tzinfo=UTC)

    def create_row(self, index: int, *, complete: bool = True) -> None:
        PerpetualStatsData.objects.create(
            symbol=self.symbol,
            timestamp=self.timestamp + timedelta(minutes=5 * index),
            frequency=5,
            open_interest=Decimal(100 + index),
            open_interest_value=Decimal(1_000_000 + index),
            top_trader_long_short_account_ratio=Decimal(f"1.{index + 1}"),
            top_trader_long_short_position_ratio=Decimal("1.2"),
            long_short_account_ratio=Decimal("1.3"),
            taker_long_short_volume_ratio=(
                Decimal("1.4") if complete else None
            ),
            open_interest_unit="base_asset",
        )

    def test_exports_selected_complete_field_in_bounded_batches(self):
        self.create_row(0)
        self.create_row(1)
        self.create_row(2, complete=False)

        with TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "market-history.parquet"
            stdout = StringIO()
            with patch(
                "quant_tick.management.commands.export_perpetual_stats."
                "EXPORT_BATCH_SIZE",
                1,
            ):
                call_command(
                    "export_perpetual_stats",
                    "--code-name",
                    self.symbol.code_name,
                    "--frequency",
                    "5",
                    "--field",
                    "top_trader_long_short_account_ratio",
                    "--output",
                    str(output),
                    stdout=stdout,
                )

            frame = pd.read_parquet(output)
            self.assertEqual(
                frame.columns.tolist(),
                ["timestamp", "top_trader_long_short_account_ratio"],
            )
            self.assertEqual(
                frame["timestamp"].tolist(),
                [
                    pd.Timestamp("2020-09-01T00:00:00Z"),
                    pd.Timestamp("2020-09-01T00:05:00Z"),
                ],
            )
            self.assertEqual(
                frame["top_trader_long_short_account_ratio"].tolist(),
                [1.1, 1.2],
            )
            self.assertEqual(list(Path(temp_dir).iterdir()), [output])
            self.assertIn("Exported 2 perpetual stats rows", stdout.getvalue())

    def test_field_specific_default_filename_is_explicit(self):
        self.assertEqual(
            get_output_path(
                self.symbol,
                5,
                ("top_trader_long_short_account_ratio",),
                "20260814",
            ),
            (
                "binance-futures-btcusdt-perpetual-5m-"
                "top-trader-long-short-account-ratio-20260814.parquet"
            ),
        )
