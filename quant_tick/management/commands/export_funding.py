from datetime import UTC, datetime
from decimal import Decimal
import re

import pyarrow as pa
import pyarrow.parquet as pq
from django.core.management.base import CommandParser
from django.db.models import Max, Min, QuerySet
from pandas import DataFrame

from quant_tick.models import FundingData, Symbol

from ..base import BaseDateCommand


def _token(value: object) -> str:
    token = str(value).strip().lower()
    token = token.replace("/", "-").replace("_", "-").replace(" ", "-").replace(".", "p")
    token = re.sub(r"[^a-z0-9-]+", "-", token)
    return re.sub(r"-+", "-", token).strip("-")


def _convert_decimals(value: object) -> object:
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, dict):
        return {key: _convert_decimals(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_convert_decimals(item) for item in value]
    return value


def get_output_path(symbol: Symbol, today: str) -> str:
    parts = [
        _token(symbol.exchange),
        _token(symbol.api_symbol),
        _token(symbol.symbol_type),
        "funding",
        today,
    ]
    return f"{'-'.join(parts)}.parquet"


class Command(BaseDateCommand):
    help = "Export funding data."

    def get_queryset(self) -> QuerySet:
        return Symbol.objects.filter(funding_data__isnull=False).distinct()

    def add_arguments(self, parser: CommandParser) -> None:
        super().add_arguments(parser)
        queryset = self.get_queryset()
        parser.add_argument(
            "--code-name",
            choices=queryset.values_list("code_name", flat=True),
        )

    def get_symbols(self, code_name: str | None) -> list[Symbol]:
        queryset = self.get_queryset().order_by("code_name")
        if code_name:
            return [queryset.get(code_name=code_name)]
        return list(queryset)

    def get_timestamp_bounds(
        self,
        symbol: Symbol,
        timestamp_from: datetime | None,
        timestamp_to: datetime | None,
    ) -> tuple[datetime, datetime] | None:
        if timestamp_from and timestamp_to:
            return timestamp_from, timestamp_to
        qs = FundingData.objects.filter(symbol=symbol)
        if timestamp_from:
            qs = qs.filter(timestamp__gte=timestamp_from)
        if timestamp_to:
            qs = qs.filter(timestamp__lt=timestamp_to)
        bounds = qs.aggregate(ts_min=Min("timestamp"), ts_max=Max("timestamp"))
        if bounds["ts_min"] is None:
            return None
        return timestamp_from or bounds["ts_min"], timestamp_to or bounds["ts_max"]

    def export_symbol(
        self,
        symbol: Symbol,
        timestamp_from: datetime | None,
        timestamp_to: datetime | None,
        today: str,
    ) -> None:
        bounds = self.get_timestamp_bounds(symbol, timestamp_from, timestamp_to)
        if bounds is None:
            self.stdout.write(
                self.style.WARNING(f"No funding data found for '{symbol.code_name}'.")
            )
            return
        ts_min, ts_max = bounds
        queryset = FundingData.objects.filter(symbol=symbol, timestamp__gte=ts_min)
        if timestamp_to:
            queryset = queryset.filter(timestamp__lt=ts_max)
        else:
            queryset = queryset.filter(timestamp__lte=ts_max)
        rows = [obj.to_row() for obj in queryset.order_by("timestamp")]
        df = DataFrame(rows)
        if df.empty:
            self.stdout.write(
                self.style.WARNING(f"No funding data found for '{symbol.code_name}'.")
            )
            return
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].apply(_convert_decimals)
        output_path = get_output_path(symbol, today)
        table = pa.Table.from_pandas(df, preserve_index=False)
        with pq.ParquetWriter(output_path, table.schema) as writer:
            writer.write_table(table)
        self.stdout.write(
            self.style.SUCCESS(f"Exported {len(df)} funding rows to {output_path}")
        )

    def handle(self, *args, **options) -> None:
        date_from = (
            datetime.fromisoformat(options["date_from"])
            if options["date_from"]
            else None
        )
        date_to = (
            datetime.fromisoformat(options["date_to"]) if options["date_to"] else None
        )
        timestamp_from = date_from.replace(tzinfo=UTC) if date_from else None
        timestamp_to = date_to.replace(tzinfo=UTC) if date_to else None
        today = datetime.now().strftime("%Y%m%d")
        for symbol in self.get_symbols(options["code_name"]):
            self.export_symbol(symbol, timestamp_from, timestamp_to, today)
