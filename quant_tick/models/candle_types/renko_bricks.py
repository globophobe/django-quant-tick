import copy
import math
from datetime import datetime
from decimal import Decimal

from django.db import models
from pandas import DataFrame

from quant_tick.constants import Direction, RenkoKind
from quant_tick.lib import aggregate_candle, merge_cache
from quant_tick.utils import gettext_lazy as _

from ..candles import CandleCache, CandleData
from .constant_candles import ConstantCandle


class RenkoBrick(ConstantCandle):
    """Renko brick.

    Using multiplicative percent scaling, with relative normalization.
    """

    def get_initial_cache(self, timestamp: datetime) -> dict:
        """Get initial cache."""
        cache = {
            "level": None,
            "entry_direction": None,  # +1 from below, -1 from above
            "aggregates": None,
            "timestamp": None,
            "sequence": 0,
            "pending_wicks": [],  # list of wick dicts waiting for parent body
        }
        if "cache_reset" in self.json_data:
            cache["date"] = timestamp.date()
        return cache

    @property
    def origin_price(self) -> Decimal:
        """Origin price."""
        return self.json_data.get("origin_price", Decimal("1"))

    @property
    def target_change(self) -> Decimal:
        """Target change."""
        return self.json_data.get("target_percentage_change", Decimal("0.01"))

    def _get_level(self, price: Decimal) -> int:
        """Get level, with boundary correction."""
        log_step = math.log1p(float(self.target_change))
        log_ratio = math.log(float(price / self.origin_price))
        level = math.floor(log_ratio / log_step)
        # Boundary correction
        while True:
            upper = self.origin_price * (Decimal(1) + self.target_change) ** (level + 1)
            lower = self.origin_price * (Decimal(1) + self.target_change) ** level
            if price < lower:
                level -= 1
            elif price >= upper:
                level += 1
            else:
                break
        return level

    _METADATA_FIELDS = frozenset(["level", "kind", "direction", "bar_index"])

    def _merge_ohlcv(self, target_row: dict, source_row: dict) -> None:
        """Merge OHLCV data from source into target row."""
        target_ohlcv = {
            k: v for k, v in target_row.items() if k not in self._METADATA_FIELDS
        }
        source_ohlcv = {
            k: v for k, v in source_row.items() if k not in self._METADATA_FIELDS
        }
        merged = merge_cache(target_ohlcv, source_ohlcv)
        for k, v in merged.items():
            target_row[k] = v

    def _merge_return_body(self, last_body_row: dict, current_row: dict) -> None:
        """Merge a return-to-previous-level body into the last body row."""
        self._merge_ohlcv(last_body_row, current_row)

    def _merge_wick_into_body(self, body_row: dict, wick_row: dict) -> None:
        """Merge wick OHLCV data into its parent body row."""
        self._merge_ohlcv(body_row, wick_row)

    def _synthesize_incomplete_brick(
        self,
        timestamp_from: datetime | None = None,
        timestamp_to: datetime | None = None,
    ) -> dict | None:
        """Synthesize incomplete brick from cache."""
        queryset = CandleCache.objects.filter(candle=self)
        if timestamp_from is not None:
            queryset = queryset.filter(timestamp__gte=timestamp_from)
        if timestamp_to is not None:
            queryset = queryset.filter(timestamp__lt=timestamp_to)
        cache_obj = queryset.order_by("-timestamp").first()
        if not cache_obj or not cache_obj.json_data:
            return None

        cache = cache_obj.json_data
        level = cache.get("level")
        if level is None:
            return None

        aggregates = cache.get("aggregates")
        if not aggregates:
            return None

        # Determine direction from entry_direction
        entry_dir = cache.get("entry_direction")
        if entry_dir == 1:
            direction = 1
        elif entry_dir == -1:
            direction = -1
        else:
            direction = 1  # Seed defaults to up

        # Build synthetic incomplete brick row
        aggregates = copy.deepcopy(aggregates)
        active_close = aggregates.get("close")

        # Merge pending wick aggregates (for plotting)
        for wick in cache.get("pending_wicks", []):
            wick_agg = wick.get("aggregates")
            if wick_agg:
                aggregates = merge_cache(aggregates, copy.deepcopy(wick_agg))

        # Remove timestamp from aggregates to preserve brick open time invariant
        aggregates.pop("timestamp", None)

        # Restore active level's close (latest trade price)
        if active_close is not None:
            aggregates["close"] = active_close

        return {
            "timestamp": cache.get("timestamp"),
            "level": level,
            "kind": RenkoKind.BODY,
            "direction": direction,
            "bar_index": cache.get("sequence"),
            **aggregates,
        }

    def aggregate(
        self,
        timestamp_from: datetime,
        timestamp_to: datetime,
        data_frame: DataFrame,
        cache_data: dict,
    ) -> tuple[list, dict | None]:
        """Aggregate."""
        candle_data = []

        for index, row in data_frame.iterrows():
            bricks = self.should_aggregate_candle(row, cache_data)

            for brick in bricks:
                # All bricks have trades (no empty bricks for jumped levels)
                candle = brick["aggregates"].copy()
                candle["timestamp"] = brick["timestamp"]

                # Store brick metadata for RenkoData creation in write_data()
                candle["_brick"] = {
                    "level": brick["level"],
                    "direction": brick["direction"],
                    "kind": brick["kind"],
                    "sequence": brick["sequence"],
                }
                candle_data.append(candle)

        # Handle incomplete brick (not yet emitted)
        data, cache_data = self.get_incomplete_candle(
            timestamp_to, candle_data, cache_data
        )
        return data, cache_data

    def get_candle_data(
        self,
        timestamp_from: datetime | None = None,
        timestamp_to: datetime | None = None,
        is_complete: bool = True,
    ) -> DataFrame:
        """Get candle data.

        Derives parent relationship by level adjacency.
        Wicks are merged into their parent.
        """
        queryset = CandleData.objects.filter(candle=self, renko_data__isnull=False)
        if timestamp_from is not None:
            queryset = queryset.filter(timestamp__gte=timestamp_from)
        if timestamp_to is not None:
            queryset = queryset.filter(timestamp__lt=timestamp_to)
        queryset = queryset.select_related("renko_data").order_by(
            "renko_data__sequence"
        )

        last_body_row = None
        output_rows = []
        body_by_level = {}  # level -> most recent body row at that level

        for obj in queryset.iterator(chunk_size=10000):
            renko = obj.renko_data
            row = obj.json_data.copy()
            row["timestamp"] = obj.timestamp
            row["candle_data_id"] = obj.pk
            row["level"] = renko.level
            row["kind"] = renko.kind
            row["direction"] = renko.direction
            row["bar_index"] = renko.sequence

            if row["kind"] == RenkoKind.WICK:
                # Find parent by direction: ↑ attaches to level+1, ↓ to level-1
                wick_level = row["level"]
                parent_level = wick_level + row["direction"]
                parent_row = body_by_level.get(parent_level)
                if parent_row:
                    self._merge_wick_into_body(parent_row, row)

            elif row["kind"] == RenkoKind.BODY:
                body_by_level[row["level"]] = row

                # Merge consecutive same-level bodies
                is_same_level = (
                    last_body_row is not None
                    and row["level"] == last_body_row["level"]
                )
                if is_same_level:
                    self._merge_return_body(last_body_row, row)
                    # Update body_by_level to point to survivor
                    body_by_level[row["level"]] = last_body_row
                else:
                    output_rows.append(row)
                    last_body_row = row

        # Synthesize incomplete brick from cache
        if not is_complete:
            incomplete_row = self._synthesize_incomplete_brick(
                timestamp_from, timestamp_to
            )
            if incomplete_row:
                output_rows.append(incomplete_row)

        # Filter incomplete (last row is always a body since wicks are merged)
        if is_complete and output_rows:
            output_rows.pop()

        return DataFrame(output_rows)

    def should_aggregate_candle(self, row: dict, cache: dict) -> list[dict]:
        """Should aggregate candle.

        Emit bricks on boundary breach. Bodies emit immediately with pending wicks.
        Wicks accumulate in cache until parent body emits.
        """
        level = self._get_level(row["price"])

        # Initialize seed level
        if cache["level"] is None:
            cache["level"] = level
            cache["entry_direction"] = None
            cache["aggregates"] = aggregate_candle(DataFrame([row.to_dict()]))
            cache["timestamp"] = row["timestamp"]
            cache["pending_wicks"] = []
            return []

        bricks = []

        if level != cache["level"]:
            exit_dir = +1 if level > cache["level"] else -1
            levels_moved = abs(level - cache["level"])

            # Body if: seed OR continuation OR multi-level reversal (2+ levels)
            is_body = (
                cache["entry_direction"] is None
                or cache["entry_direction"] == exit_dir
                or levels_moved >= 2
            )

            brick = {
                "level": cache["level"],
                "direction": exit_dir,
                "kind": RenkoKind.BODY if is_body else RenkoKind.WICK,
                "aggregates": cache["aggregates"],
                "timestamp": cache["timestamp"],
            }

            if is_body:
                # Emit body with sequence
                brick["sequence"] = cache["sequence"]
                cache["sequence"] += 1
                bricks.append(brick)

                # Emit pending wicks (they follow parent body)
                for wick in cache["pending_wicks"]:
                    wick["sequence"] = cache["sequence"]
                    cache["sequence"] += 1
                    bricks.append(wick)
                cache["pending_wicks"] = []
            else:
                # Wick: accumulate in cache, don't save yet
                cache["pending_wicks"].append(brick)

            # Move to new level
            cache["level"] = level
            cache["entry_direction"] = exit_dir
            cache["aggregates"] = None
            cache["timestamp"] = row["timestamp"]

        # Accumulate trade into current level
        new_agg = aggregate_candle(DataFrame([row.to_dict()]))
        if cache["aggregates"] is not None:
            cache["aggregates"] = merge_cache(cache["aggregates"], new_agg)
        else:
            cache["aggregates"] = new_agg

        return bricks

    def write_data(
        self, timestamp_from: datetime, timestamp_to: datetime, json_data: list[dict]
    ) -> None:
        """Write data."""
        candle_data_list = []
        brick_metadata = []

        for j in json_data:
            timestamp = j.pop("timestamp")
            brick = j.pop("_brick", None)
            kwargs = {"timestamp": timestamp, "json_data": j}
            candle_data_list.append(CandleData(candle=self, **kwargs))
            brick_metadata.append(brick)

        # Bulk create CandleData
        created_candle_data = CandleData.objects.bulk_create(candle_data_list)

        # Bulk create RenkoData (no parent tracking - derived in get_candle_data)
        renko_data_list = []
        for candle_data, brick in zip(created_candle_data, brick_metadata, strict=True):
            if brick:
                renko_data_list.append(
                    RenkoData(
                        candle_data=candle_data,
                        level=brick["level"],
                        direction=brick["direction"],
                        kind=brick["kind"],
                        sequence=brick["sequence"],
                    )
                )

        if renko_data_list:
            RenkoData.objects.bulk_create(renko_data_list)

    class Meta:
        proxy = True
        verbose_name = _("renko brick")
        verbose_name_plural = _("renko bricks")


class RenkoData(models.Model):
    """Renko data."""

    candle_data = models.OneToOneField(
        "quant_tick.CandleData",
        on_delete=models.CASCADE,
        primary_key=True,
        related_name="renko_data",
        verbose_name=_("candle data"),
    )
    level = models.IntegerField(_("level"), db_index=True)
    sequence = models.BigIntegerField(_("sequence"), db_index=True)
    direction = models.SmallIntegerField(_("direction"), choices=Direction.choices)
    kind = models.CharField(
        _("kind"),
        max_length=10,
        choices=RenkoKind.choices,
        default=RenkoKind.BODY,
        db_index=True,
    )

    class Meta:
        db_table = "quant_tick_renko_data"
        indexes = [
            models.Index(fields=["level", "sequence"]),
        ]
        ordering = ["sequence"]
        verbose_name = verbose_name_plural = _("renko data")
