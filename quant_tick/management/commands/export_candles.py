import json
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pyarrow as pa
import pyarrow.parquet as pq
from django.core.management.base import CommandParser
from django.db.models import Max, Min, QuerySet
from pandas import DataFrame

from quant_tick.models import Candle
from quant_tick.models.candles import CandleData

from ..base import BaseDateCommand


def _convert_decimals(val: object) -> object:
    """Recursively convert Decimal values to float."""
    if isinstance(val, Decimal):
        return float(val)
    if isinstance(val, dict):
        return {k: _convert_decimals(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_convert_decimals(item) for item in val]
    return val


class Command(BaseDateCommand):
    """Export candle data."""

    help = "Export candle data."

    def get_queryset(self) -> QuerySet:
        """Get queryset."""
        return Candle.objects.filter(is_active=True)

    def add_arguments(self, parser: CommandParser) -> None:
        """Add arguments."""
        super().add_arguments(parser)
        queryset = self.get_queryset()
        parser.add_argument(
            "--code-name",
            choices=queryset.values_list("code_name", flat=True),
            required=True,
        )

    def handle(self, *args, **options) -> None:
        """Handle."""
        code_name = options["code_name"]
        candle = self.get_queryset().get(code_name=code_name)
        date_from = (
            datetime.fromisoformat(options["date_from"])
            if options["date_from"]
            else None
        )
        date_to = (
            datetime.fromisoformat(options["date_to"]) if options["date_to"] else None
        )
        timestamp_from = date_from.replace(tzinfo=timezone.utc) if date_from else None
        timestamp_to = date_to.replace(tzinfo=timezone.utc) if date_to else None

        if timestamp_from and timestamp_to:
            ts_min = timestamp_from
            ts_max = timestamp_to
        else:
            qs = CandleData.objects.filter(candle=candle)
            if timestamp_from:
                qs = qs.filter(timestamp__gte=timestamp_from)
            if timestamp_to:
                qs = qs.filter(timestamp__lt=timestamp_to)
            bounds = qs.aggregate(ts_min=Min("timestamp"), ts_max=Max("timestamp"))
            if bounds["ts_min"] is None:
                self.stdout.write(self.style.WARNING("No data found."))
                return
            ts_min = timestamp_from or bounds["ts_min"]
            ts_max = timestamp_to or bounds["ts_max"] + timedelta(microseconds=1)

        today = datetime.now().strftime("%Y%m%d")
        output_path = f"{code_name}-candles-{today}.parquet"

        self.stdout.write(f"Exporting candles for '{code_name}'...")

        writer = None
        total_rows = 0
        cur = datetime(ts_min.year, 1, 1, tzinfo=timezone.utc)

        while cur < ts_max:
            next_year = datetime(cur.year + 1, 1, 1, tzinfo=timezone.utc)
            chunk_from = max(cur, ts_min)
            chunk_to = min(next_year, ts_max)
            rows = candle.get_candle_data(
                timestamp_from=chunk_from,
                timestamp_to=chunk_to,
            )
            df = DataFrame(rows)
            if not df.empty:
                df = candle.process_data_frame(df)
                for col in df.columns:
                    if df[col].dtype == object:
                        df[col] = df[col].apply(_convert_decimals)
                # Serialize variable-key dicts to JSON strings
                if "distribution" in df.columns:
                    df["distribution"] = df["distribution"].apply(json.dumps)
                table = pa.Table.from_pandas(df, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(output_path, table.schema)
                writer.write_table(table)
                total_rows += len(df)
                self.stdout.write(f"  {cur.year}: {len(df)} candles")
            cur = next_year

        if writer:
            writer.close()
            self.stdout.write(
                self.style.SUCCESS(f"Exported {total_rows} candles to {output_path}")
            )
        else:
            self.stdout.write(self.style.WARNING("No data found."))
