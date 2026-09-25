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

    def create_row(self, index: int, *, complete: bool = True) -> PerpetualStatsData:
        return PerpetualStatsData.objects.create(
            symbol=self.symbol,
            timestamp=self.timestamp + timedelta(minutes=5 * index),
            frequency=5,
            open_interest=Decimal(100 + index),
            open_interest_value=Decimal(1_000_000 + index),
            top_trader_long_short_account_ratio=Decimal(f"1.{index + 1}"),
            top_trader_long_short_position_ratio=Decimal("1.2"),
            long_short_account_ratio=Decimal("1.3"),
            taker_long_short_volume_ratio=(Decimal("1.4") if complete else None),
            open_interest_unit="base_asset",
        )

    def test_exports_rows_complete_for_selected_fields_in_bounded_batches(self):
        self.create_row(0, complete=False)
        self.create_row(1)
        self.create_row(2, complete=False)
        missing = self.create_row(3)
        missing.top_trader_long_short_account_ratio = None
        missing.save(update_fields=["top_trader_long_short_account_ratio"])

        ratio = "top_trader_long_short_account_ratio"
        default_columns = [
            "open_interest",
            "open_interest_value",
            ratio,
            "top_trader_long_short_position_ratio",
            "long_short_account_ratio",
            "taker_long_short_volume_ratio",
            "open_interest_unit",
        ]
        cases = (
            ((ratio,), [0, 1, 2], [1.1, 1.2, 1.3], {}),
            ((ratio, "taker_long_short_volume_ratio"), [1], [1.2], {}),
            ((), [1], [1.2], {}),
            (
                (ratio,),
                [1],
                [1.2],
                {
                    "date_from": "2020-09-01T00:05:00",
                    "date_to": "2020-09-01T00:10:00",
                },
            ),
        )
        for fields, indices, ratios, bounds in cases:
            with (
                self.subTest(fields=fields, bounds=bounds),
                TemporaryDirectory() as temp_dir,
            ):
                output = Path(temp_dir) / "market-history.parquet"
                stdout = StringIO()
                args = [
                    "--code-name",
                    self.symbol.code_name,
                    "--frequency",
                    "5",
                    "--output",
                    str(output),
                ]
                if fields:
                    args.extend(["--field", *fields])
                with patch(
                    "quant_tick.management.commands.export_perpetual_stats."
                    "EXPORT_BATCH_SIZE",
                    1,
                ):
                    call_command(
                        "export_perpetual_stats", *args, stdout=stdout, **bounds
                    )

                frame = pd.read_parquet(output)
                self.assertEqual(
                    frame.columns.tolist(),
                    ["timestamp", *(fields or default_columns)],
                )
                self.assertEqual(
                    frame["timestamp"].tolist(),
                    [self.timestamp + timedelta(minutes=5 * i) for i in indices],
                )
                self.assertEqual(frame[ratio].tolist(), ratios)
                self.assertEqual(list(Path(temp_dir).iterdir()), [output])
                expected_message = f"Exported {len(indices)} perpetual stats rows"
                self.assertIn(expected_message, stdout.getvalue())

    def test_selected_field_exports_without_any_exchange_complete_rows(self):
        row = self.create_row(0, complete=False)
        row.top_trader_long_short_account_ratio = Decimal(0)
        row.open_interest_unit = ""
        row.save(
            update_fields=["top_trader_long_short_account_ratio", "open_interest_unit"]
        )

        with TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "partial-history.parquet"
            call_command(
                "export_perpetual_stats",
                "--code-name",
                self.symbol.code_name,
                "--frequency",
                "5",
                "--field",
                "top_trader_long_short_account_ratio",
                "open_interest_unit",
                "--output",
                str(output),
                stdout=StringIO(),
            )
            frame = pd.read_parquet(output)
            self.assertEqual(frame["timestamp"].tolist(), [self.timestamp])
            self.assertEqual(
                frame["top_trader_long_short_account_ratio"].tolist(), [0.0]
            )
            self.assertTrue(frame["open_interest_unit"].isna().all())

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

    def test_phoenix_default_export_uses_open_interest_fields(self):
        symbol = self.get_symbol(
            exchange=Exchange.PHOENIX,
            api_symbol="BTC",
            symbol_type=SymbolType.PERPETUAL,
        )
        PerpetualStatsData.objects.create(
            symbol=symbol,
            timestamp=self.timestamp,
            frequency=60,
            open_interest=Decimal("24.4222"),
            open_interest_value=Decimal("2065800.6314"),
            open_interest_unit="base_asset",
        )
        with TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "phoenix-stats.parquet"
            call_command(
                "export_perpetual_stats",
                "--code-name",
                symbol.code_name,
                "--output",
                str(output),
                stdout=StringIO(),
            )
            frame = pd.read_parquet(output)
        self.assertEqual(
            frame.columns.tolist(),
            ["timestamp", "open_interest", "open_interest_value", "open_interest_unit"],
        )
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.iloc[0].open_interest, 24.4222)
