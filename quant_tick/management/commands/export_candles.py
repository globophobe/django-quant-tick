from datetime import UTC, datetime, timedelta
from decimal import Decimal
import re

import pyarrow as pa
import pyarrow.parquet as pq
from django.core.management.base import CommandParser
from django.db.models import Max, Min, QuerySet
from pandas import DataFrame

from quant_tick.constants import Frequency
from quant_tick.models import AdaptiveCandle, Candle, ConstantCandle, TimeBasedCandle
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


def _token(value: object) -> str:
    token = str(value).strip().lower().replace("_data", "")
    token = token.replace("/", "-").replace("_", "-").replace(" ", "-").replace(".", "p")
    token = re.sub(r"[^a-z0-9-]+", "-", token)
    token = re.sub(r"-+", "-", token).strip("-")
    return token


def _cache_reset_token(value: object) -> str | None:
    if value == Frequency.DAY:
        return "daily"
    if value == Frequency.WEEK:
        return "weekly"
    if value is None:
        return None
    return _token(value)


def _describe_candle(candle: Candle) -> list[str]:
    data = candle.json_data or {}
    source = data.get("source_data")

    if isinstance(candle, TimeBasedCandle):
        parts = ["timebased"]
        window = data.get("window")
        if window:
            parts.append(_token(window))
        if source:
            parts.append(_token(source))
        if (
            data.get("min_volume_exponent") == 1
            and data.get("min_notional_exponent") == 1
        ):
            parts.append("round")
        else:
            min_volume_exponent = data.get("min_volume_exponent")
            min_notional_exponent = data.get("min_notional_exponent")
            if min_volume_exponent is not None:
                parts.append(f"mv{min_volume_exponent}")
            if min_notional_exponent is not None:
                parts.append(f"mn{min_notional_exponent}")
        return parts

    if isinstance(candle, AdaptiveCandle):
        parts = ["adaptive"]
        if source:
            parts.append(_token(source))
        sample_type = data.get("sample_type")
        if sample_type:
            parts.append(_token(sample_type))
        target_candles_per_day = data.get("target_candles_per_day")
        if target_candles_per_day is not None:
            parts.append(f"{target_candles_per_day}cpd")
        moving_average_number_of_days = data.get("moving_average_number_of_days")
        if moving_average_number_of_days is not None:
            parts.append(f"ma{moving_average_number_of_days}d")
        return parts

    if isinstance(candle, ConstantCandle):
        parts = ["constant"]
        if source:
            parts.append(_token(source))
        sample_type = data.get("sample_type")
        if sample_type:
            parts.append(_token(sample_type))
        target_value = data.get("target_value")
        if target_value is not None:
            parts.append(f"target{_token(target_value)}")
        cache_reset = _cache_reset_token(data.get("cache_reset"))
        if cache_reset:
            parts.append(cache_reset)
        return parts

    return [_token(candle.__class__.__name__)]


def get_output_path(candle: Candle, today: str) -> str:
    parts = [
        _token(candle.symbol.exchange),
        _token(candle.symbol.api_symbol),
        *_describe_candle(candle),
        "candles",
        today,
    ]
    parts = [part for part in parts if part]
    return f"{'-'.join(parts)}.parquet"


class Command(BaseDateCommand):
    """Export candle data."""

    help = "Export candle data."

    def get_queryset(self) -> QuerySet:
        return Candle.objects.filter(is_active=True)

    def add_arguments(self, parser: CommandParser) -> None:
        super().add_arguments(parser)
        queryset = self.get_queryset()
        parser.add_argument(
            "--code-name",
            choices=queryset.values_list("code_name", flat=True),
        )

    def get_candles(self, code_name: str | None) -> list[Candle]:
        queryset = self.get_queryset().select_related("symbol").order_by("code_name")
        if code_name:
            return [queryset.get(code_name=code_name)]
        return list(queryset)

    def get_timestamp_bounds(
        self,
        candle: Candle,
        timestamp_from: datetime | None,
        timestamp_to: datetime | None,
    ) -> tuple[datetime, datetime] | None:
        if timestamp_from and timestamp_to:
            return timestamp_from, timestamp_to

        qs = CandleData.objects.filter(candle=candle)
        if timestamp_from:
            qs = qs.filter(timestamp__gte=timestamp_from)
        if timestamp_to:
            qs = qs.filter(timestamp__lt=timestamp_to)
        bounds = qs.aggregate(ts_min=Min("timestamp"), ts_max=Max("timestamp"))
        if bounds["ts_min"] is None:
            return None
        ts_min = timestamp_from or bounds["ts_min"]
        ts_max = timestamp_to or bounds["ts_max"] + timedelta(microseconds=1)
        return ts_min, ts_max

    def export_candle(
        self,
        candle: Candle,
        timestamp_from: datetime | None,
        timestamp_to: datetime | None,
        today: str,
    ) -> None:
        bounds = self.get_timestamp_bounds(candle, timestamp_from, timestamp_to)
        if bounds is None:
            self.stdout.write(
                self.style.WARNING(f"No data found for '{candle.code_name}'.")
            )
            return
        ts_min, ts_max = bounds
        output_path = get_output_path(candle, today)
        self.stdout.write(f"Exporting candles for '{candle.code_name}'...")

        writer = None
        total_rows = 0
        cur = datetime(ts_min.year, 1, 1, tzinfo=UTC)

        while cur < ts_max:
            next_year = datetime(cur.year + 1, 1, 1, tzinfo=UTC)
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
            self.stdout.write(
                self.style.WARNING(f"No data found for '{candle.code_name}'.")
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

        for candle in self.get_candles(options["code_name"]):
            self.export_candle(candle, timestamp_from, timestamp_to, today)
