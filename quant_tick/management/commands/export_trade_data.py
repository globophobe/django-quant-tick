from datetime import datetime

import pyarrow as pa
import pyarrow.parquet as pq
from django.core.management.base import CommandParser
from django.db.models import QuerySet

from quant_tick.lib import parse_period_from_to
from quant_tick.models import Symbol, TradeData

from ..base import BaseDateTimeCommand


class Command(BaseDateTimeCommand):
    """Export trade data."""

    help = "Export trade data (filtered_data) to parquet."

    def get_queryset(self) -> QuerySet:
        """Get queryset."""
        return Symbol.objects.filter(is_active=True)

    def add_arguments(self, parser: CommandParser) -> None:
        """Add arguments."""
        super().add_arguments(parser)
        queryset = self.get_queryset()
        parser.add_argument(
            "--code-name",
            choices=queryset.values_list("code_name", flat=True),
            required=True,
        )
        parser.add_argument(
            "--source",
            choices=["raw_data", "aggregated_data", "filtered_data"],
            default="filtered_data",
        )

    def handle(self, *args, **options) -> None:
        """Handle."""
        code_name = options["code_name"]
        source = options["source"]
        symbol = self.get_queryset().get(code_name=code_name)

        timestamp_from, timestamp_to = parse_period_from_to(
            date_from=options["date_from"],
            time_from=options["time_from"],
            date_to=options["date_to"],
            time_to=options["time_to"],
        )

        self.stdout.write(f"Exporting {source} for '{code_name}'...")

        queryset = TradeData.objects.filter(symbol=symbol).order_by("timestamp")
        if timestamp_from:
            queryset = queryset.filter(timestamp__gte=timestamp_from)
        if timestamp_to:
            queryset = queryset.filter(timestamp__lt=timestamp_to)

        today = datetime.now().strftime("%Y%m%d")
        output_path = f"{code_name}-{source.replace('_', '-')}-{today}.parquet"

        writer = None
        total_rows = 0

        for obj in queryset:
            df = obj.get_data_frame(source)
            if df is not None and len(df):
                # Convert all numeric columns to float for consistent schema
                for col in df.columns:
                    if col not in ("timestamp", "uid"):
                        df[col] = df[col].astype(float)
                table = pa.Table.from_pandas(df, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(output_path, table.schema)
                writer.write_table(table)
                total_rows += len(df)
                self.stdout.write(f"  {obj.timestamp.date()}: {len(df)} rows")

        if writer:
            writer.close()
            self.stdout.write(
                self.style.SUCCESS(f"Exported {total_rows} rows to {output_path}")
            )
        else:
            self.stdout.write(self.style.WARNING("No data found."))
