import os
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

import pyarrow as pa
import pyarrow.parquet as pq
from django.core.management.base import CommandError

from quant_tick.management.base import BaseDateCommand
from quant_tick.models import OrderBookData, Symbol


class Command(BaseDateCommand):
    help = "Export order book minute summaries to Parquet."

    def add_arguments(self, parser):
        super().add_arguments(parser)
        parser.add_argument("--code-name", required=True)
        parser.add_argument("--output", type=Path, required=True)

    def handle(self, *args, **options):
        try:
            symbol = Symbol.objects.get(code_name=options["code_name"])
        except Symbol.DoesNotExist as error:
            raise CommandError("Unknown symbol code name") from error
        queryset = OrderBookData.objects.filter(symbol=symbol).order_by("timestamp")
        for name, lookup in (
            ("date_from", "timestamp__gte"),
            ("date_to", "timestamp__lt"),
        ):
            if options[name]:
                queryset = queryset.filter(
                    **{
                        lookup: datetime.fromisoformat(options[name]).replace(
                            tzinfo=UTC
                        )
                    }
                )
        output = options["output"]
        temporary = None
        writer = None
        count = 0
        try:
            batch = []
            for row in queryset.iterator(chunk_size=5000):
                batch.append(
                    {
                        "timestamp": row.timestamp,
                        **{key: float(value) for key, value in row.json_data.items()},
                    }
                )
                if len(batch) == 5000:
                    if writer is None:
                        with NamedTemporaryFile(
                            dir=output.parent, delete=False
                        ) as handle:
                            temporary = handle.name
                        writer = pq.ParquetWriter(
                            temporary, pa.Table.from_pylist(batch).schema
                        )
                    writer.write_table(
                        pa.Table.from_pylist(batch, schema=writer.schema)
                    )
                    count += len(batch)
                    batch = []
            if batch:
                if writer is None:
                    with NamedTemporaryFile(dir=output.parent, delete=False) as handle:
                        temporary = handle.name
                    writer = pq.ParquetWriter(
                        temporary, pa.Table.from_pylist(batch).schema
                    )
                writer.write_table(pa.Table.from_pylist(batch, schema=writer.schema))
                count += len(batch)
            if writer is not None:
                writer.close()
                writer = None
                os.replace(temporary, output)
                temporary = None
        finally:
            if writer is not None:
                writer.close()
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)
        if count:
            self.stdout.write(f"Exported {count} order book rows to {output}")
        else:
            self.stdout.write("No matching order book rows; no file written")
