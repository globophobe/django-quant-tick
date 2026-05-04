from datetime import UTC, datetime, timedelta
from decimal import Decimal
import re

import pyarrow as pa
import pyarrow.parquet as pq
from django.core.management.base import CommandParser
from django.db.models import Max, Min, QuerySet
from pandas import DataFrame

from quant_tick.models import ExchangeCandleData, Symbol

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


def get_output_path(symbol: Symbol, frequency: int, today: str) -> str:
    parts = [
        _token(symbol.exchange),
        _token(symbol.api_symbol),
        _token(symbol.symbol_type),
        f"{frequency}m",
        "exchange-candles",
        today,
    ]
    return f"{'-'.join(parts)}.parquet"


class Command(BaseDateCommand):
    help = "Export direct exchange candle data."

    def get_queryset(self) -> QuerySet:
        return Symbol.objects.filter(exchange_candle_data__isnull=False).distinct()

    def add_arguments(self, parser: CommandParser) -> None:
        super().add_arguments(parser)
        queryset = self.get_queryset()
        parser.add_argument(
            "--code-name",
            choices=queryset.values_list("code_name", flat=True),
        )
        parser.add_argument("--frequency", type=int)

    def get_symbols(self, code_name: str | None) -> list[Symbol]:
        queryset = self.get_queryset().order_by("code_name")
        if code_name:
            return [queryset.get(code_name=code_name)]
        return list(queryset)

    def get_frequencies(self, symbol: Symbol, frequency: int | None) -> list[int]:
        if frequency:
            return [frequency]
        return list(
            ExchangeCandleData.objects.filter(symbol=symbol)
            .order_by("frequency")
            .values_list("frequency", flat=True)
            .distinct()
        )

    def get_timestamp_bounds(
        self,
        symbol: Symbol,
        frequency: int,
        timestamp_from: datetime | None,
        timestamp_to: datetime | None,
    ) -> tuple[datetime, datetime] | None:
        if timestamp_from and timestamp_to:
            return timestamp_from, timestamp_to
        qs = ExchangeCandleData.objects.filter(symbol=symbol, frequency=frequency)
        if timestamp_from:
            qs = qs.filter(timestamp__gte=timestamp_from)
        if timestamp_to:
            qs = qs.filter(timestamp__lt=timestamp_to)
        bounds = qs.aggregate(ts_min=Min("timestamp"), ts_max=Max("timestamp"))
        if bounds["ts_min"] is None:
            return None
        ts_min = timestamp_from or bounds["ts_min"]
        ts_max = timestamp_to or bounds["ts_max"] + timedelta(minutes=frequency)
        return ts_min, ts_max

    def export_symbol_frequency(
        self,
        symbol: Symbol,
        frequency: int,
        timestamp_from: datetime | None,
        timestamp_to: datetime | None,
        today: str,
    ) -> None:
        bounds = self.get_timestamp_bounds(
            symbol,
            frequency,
            timestamp_from,
            timestamp_to,
        )
        if bounds is None:
            self.stdout.write(
                self.style.WARNING(
                    f"No exchange candle data found for '{symbol.code_name}'."
                )
            )
            return
        ts_min, ts_max = bounds
        queryset = ExchangeCandleData.objects.filter(
            symbol=symbol,
            frequency=frequency,
            timestamp__gte=ts_min,
            timestamp__lt=ts_max,
        ).order_by("timestamp")
        include_volume = queryset.filter(volume__isnull=False).exists()
        include_notional = queryset.filter(notional__isnull=False).exists()
        rows = [
            obj.to_row(
                include_volume=include_volume,
                include_notional=include_notional,
            )
            for obj in queryset
        ]
        df = DataFrame(rows)
        if df.empty:
            self.stdout.write(
                self.style.WARNING(
                    f"No exchange candle data found for '{symbol.code_name}'."
                )
            )
            return
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].apply(_convert_decimals)
        output_path = get_output_path(symbol, frequency, today)
        table = pa.Table.from_pandas(df, preserve_index=False)
        with pq.ParquetWriter(output_path, table.schema) as writer:
            writer.write_table(table)
        self.stdout.write(
            self.style.SUCCESS(
                f"Exported {len(df)} exchange candles to {output_path}"
            )
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
            for frequency in self.get_frequencies(symbol, options["frequency"]):
                self.export_symbol_frequency(
                    symbol,
                    frequency,
                    timestamp_from,
                    timestamp_to,
                    today,
                )
