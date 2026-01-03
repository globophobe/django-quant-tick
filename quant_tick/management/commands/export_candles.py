from datetime import datetime

from django.core.management.base import CommandParser
from django.db.models import QuerySet

from quant_tick.models import Candle

from ..base import BaseDateCommand


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
        timestamp_from = (
            date_from.replace(tzinfo=datetime.timezone.utc) if date_from else None
        )
        timestamp_to = (
            date_to.replace(tzinfo=datetime.timezone.utc) if date_to else None
        )

        self.stdout.write(f"Exporting candles for '{code_name}'...")

        data_frame = candle.get_candle_data(
            timestamp_from=timestamp_from,
            timestamp_to=timestamp_to,
            progress=True,
        )

        df = strategy.get_data_frame(df)

        today = datetime.now().strftime("%Y%m%d")
        output_path = f"{code_name}-candles-{today}.parquet"

        df.to_parquet(output_path, index=False)
        self.stdout.write(
            self.style.SUCCESS(f"Exported {len(df)} candles to {output_path}")
        )
