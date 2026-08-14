import re
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from tempfile import NamedTemporaryFile

import pyarrow as pa
import pyarrow.parquet as pq
from django.core.management.base import CommandError, CommandParser
from django.db.models import Max, Min, QuerySet

from quant_tick.models import PerpetualStatsData, Symbol
from quant_tick.models.perpetual_stats import (
    PERPETUAL_STATS_VALUE_FIELDS,
    perpetual_stats_required_fields,
)

from ..base import BaseDateCommand

EXPORTABLE_FIELDS = (*PERPETUAL_STATS_VALUE_FIELDS, "open_interest_unit")
EXPORT_BATCH_SIZE = 50_000


def _token(value: object) -> str:
    token = str(value).strip().lower()
    token = (
        token.replace("/", "-").replace("_", "-").replace(" ", "-").replace(".", "p")
    )
    token = re.sub(r"[^a-z0-9-]+", "-", token)
    return re.sub(r"-+", "-", token).strip("-")


def get_output_path(
    symbol: Symbol,
    frequency: int,
    fields: tuple[str, ...],
    today: str,
) -> str:
    field_token = _token(fields[0]) if len(fields) == 1 else "perpetual-stats"
    parts = [
        _token(symbol.exchange),
        _token(symbol.api_symbol),
        _token(symbol.symbol_type),
        f"{frequency}m",
        field_token,
        today,
    ]
    return f"{'-'.join(parts)}.parquet"


def _iter_batches(
    rows: Iterable[dict[str, object]],
    *,
    batch_size: int,
) -> Iterator[list[dict[str, object]]]:
    batch = []
    for row in rows:
        batch.append(row)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _build_table(
    rows: list[dict[str, object]],
    fields: tuple[str, ...],
) -> pa.Table:
    arrays = [
        pa.array(
            [row["timestamp"] for row in rows],
            type=pa.timestamp("us", tz="UTC"),
        )
    ]
    for field in fields:
        values = [row[field] for row in rows]
        if field in PERPETUAL_STATS_VALUE_FIELDS:
            arrays.append(
                pa.array(
                    [
                        float(value) if isinstance(value, Decimal) else value
                        for value in values
                    ],
                    type=pa.float64(),
                )
            )
        else:
            arrays.append(
                pa.array(
                    [str(value) if value not in (None, "") else None for value in values],
                    type=pa.string(),
                )
            )
    return pa.Table.from_arrays(arrays, names=["timestamp", *fields])


class Command(BaseDateCommand):
    help = (
        "Export complete native-cadence PerpetualStatsData to a bounded-memory "
        "Parquet file for research and benchmark fixtures."
    )

    def get_queryset(self) -> QuerySet:
        return Symbol.objects.filter(perpetual_stats_data__isnull=False).distinct()

    def add_arguments(self, parser: CommandParser) -> None:
        super().add_arguments(parser)
        queryset = self.get_queryset()
        parser.add_argument(
            "--code-name",
            choices=queryset.values_list("code_name", flat=True),
        )
        parser.add_argument("--frequency", type=int)
        parser.add_argument(
            "--field",
            choices=EXPORTABLE_FIELDS,
            nargs="+",
            help=(
                "Columns to export after timestamp. Defaults to the exchange's "
                "required fields plus open_interest_unit."
            ),
        )
        parser.add_argument(
            "--output",
            type=Path,
            help="Explicit output path; requires exactly one symbol/frequency target.",
        )

    def get_symbols(self, code_name: str | None) -> list[Symbol]:
        queryset = self.get_queryset().order_by("code_name")
        if code_name:
            return [queryset.get(code_name=code_name)]
        return list(queryset)

    def get_frequencies(self, symbol: Symbol, frequency: int | None) -> list[int]:
        if frequency is not None:
            if frequency <= 0:
                raise CommandError("--frequency must be positive")
            return [frequency]
        return list(
            PerpetualStatsData.objects.filter(symbol=symbol)
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
        queryset = PerpetualStatsData.objects.filter(
            symbol=symbol,
            frequency=frequency,
        ).complete_for_exchange(symbol.exchange)
        if timestamp_from:
            queryset = queryset.filter(timestamp__gte=timestamp_from)
        if timestamp_to:
            queryset = queryset.filter(timestamp__lt=timestamp_to)
        bounds = queryset.aggregate(ts_min=Min("timestamp"), ts_max=Max("timestamp"))
        if bounds["ts_min"] is None:
            return None
        return (
            timestamp_from or bounds["ts_min"],
            timestamp_to or bounds["ts_max"] + timedelta(minutes=frequency),
        )

    def export_symbol_frequency(
        self,
        symbol: Symbol,
        frequency: int,
        fields: tuple[str, ...],
        timestamp_from: datetime | None,
        timestamp_to: datetime | None,
        output_path: Path,
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
                    f"No complete perpetual stats found for '{symbol.code_name}' "
                    f"at {frequency}m."
                )
            )
            return

        ts_min, ts_max = bounds
        queryset = (
            PerpetualStatsData.objects.filter(
                symbol=symbol,
                frequency=frequency,
                timestamp__gte=ts_min,
                timestamp__lt=ts_max,
            )
            .complete_for_exchange(symbol.exchange)
            .order_by("timestamp", "id")
            .values("timestamp", *fields)
            .iterator(chunk_size=EXPORT_BATCH_SIZE)
        )

        writer = None
        staged_path = None
        total_rows = 0
        try:
            try:
                for batch in _iter_batches(queryset, batch_size=EXPORT_BATCH_SIZE):
                    table = _build_table(batch, fields)
                    if writer is None:
                        with NamedTemporaryFile(
                            prefix=f".{output_path.name}.",
                            suffix=".tmp",
                            dir=output_path.parent,
                            delete=False,
                        ) as staged:
                            staged_path = Path(staged.name)
                        writer = pq.ParquetWriter(staged_path, table.schema)
                    writer.write_table(table)
                    total_rows += len(batch)
            finally:
                if writer is not None:
                    writer.close()
        except Exception:
            if staged_path is not None:
                staged_path.unlink(missing_ok=True)
            raise

        if staged_path is None:
            self.stdout.write(
                self.style.WARNING(
                    f"No complete perpetual stats found for '{symbol.code_name}' "
                    f"at {frequency}m."
                )
            )
            return
        try:
            staged_path.replace(output_path)
        except Exception:
            staged_path.unlink(missing_ok=True)
            raise

        self.stdout.write(
            self.style.SUCCESS(
                f"Exported {total_rows} perpetual stats rows to {output_path}"
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
        fields = tuple(
            dict.fromkeys(options["field"] or ())
        )
        today = datetime.now(tz=UTC).strftime("%Y%m%d")

        targets = [
            (symbol, frequency)
            for symbol in self.get_symbols(options["code_name"])
            for frequency in self.get_frequencies(symbol, options["frequency"])
        ]
        if options["output"] is not None and len(targets) != 1:
            raise CommandError(
                "--output requires exactly one symbol/frequency target; "
                "use --code-name and --frequency"
            )

        for symbol, frequency in targets:
            export_fields = fields or (
                *perpetual_stats_required_fields(symbol.exchange),
                "open_interest_unit",
            )
            output_path = options["output"] or Path(
                get_output_path(symbol, frequency, export_fields, today)
            )
            self.export_symbol_frequency(
                symbol,
                frequency,
                export_fields,
                timestamp_from,
                timestamp_to,
                output_path,
            )
