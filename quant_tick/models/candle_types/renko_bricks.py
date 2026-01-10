import copy
from collections.abc import Iterator
from datetime import datetime
from decimal import Decimal

import numpy as np
import pandas as pd
from django.db import models
from pandas import DataFrame

from quant_tick.constants import Direction, RenkoKind
from quant_tick.lib import aggregate_candle, merge_cache
from quant_tick.utils import gettext_lazy as _

from ..candles import CacheResetMixin, Candle, CandleCache, CandleData


class RenkoBrick(CacheResetMixin, Candle):
    """Renko brick.

    Using multiplicative percent scaling, with relative normalization.
    """

    def get_initial_cache(self, timestamp: datetime) -> dict:
        """Get initial cache."""
        cache = {
            "timestamp": None,
            "level": None,
            "direction": None,
            "ohlcv": None,
            "sequence": 0,
            "wicks": [],
        }
        if "cache_reset" in self.json_data:
            cache["date"] = timestamp.date()
        return cache

    def get_incomplete_candle(
        self, timestamp: datetime, data: list, cache_data: dict
    ) -> tuple[list, dict]:
        """Get incomplete candle.

        Emit incomplete brick if cache resets next iteration.
        """
        ts = timestamp + pd.Timedelta("1us")
        if self.should_reset_cache(ts, cache_data):
            brick = self._build_incomplete_brick(cache_data)
            if brick:
                brick["incomplete"] = True
                data.append(brick)
        return data, cache_data

    def _build_incomplete_brick(self, cache: dict) -> dict | None:
        """Build incomplete brick from cache."""
        level = cache.get("level")
        if level is None:
            return None

        ohlcv = cache.get("ohlcv")
        if not ohlcv:
            return None

        ohlcv = copy.deepcopy(ohlcv)
        active_close = ohlcv.get("close")

        # Merge pending wick data
        for wick in cache.get("wicks", []):
            wick_ohlcv = wick.get("ohlcv")
            if wick_ohlcv:
                ohlcv = merge_cache(ohlcv, copy.deepcopy(wick_ohlcv))

        ohlcv.pop("timestamp", None)

        if active_close is not None:
            ohlcv["close"] = active_close

        return {
            "timestamp": cache.get("timestamp"),
            "level": level,
            "kind": RenkoKind.BODY,
            "direction": cache.get("direction"),
            "bar_index": cache.get("sequence"),
            **ohlcv,
        }

    def _compute_levels(self, prices: np.ndarray) -> np.ndarray:
        """Compute levels."""
        origin_price = self.json_data.get("origin_price", Decimal("1"))
        target_change = self.json_data.get("target_percentage_change", Decimal("0.01"))
        origin = float(origin_price)
        log_step = np.log1p(float(target_change))
        log_ratios = np.log(prices / origin)
        return np.floor(log_ratios / log_step).astype(np.int64)

    def _find_level_boundaries(self, levels: np.ndarray) -> np.ndarray:
        """Find indices where level changes."""
        if len(levels) <= 1:
            return np.array([0, len(levels)])
        change_indices = np.nonzero(np.diff(levels))[0] + 1
        return np.concatenate([[0], change_indices, [len(levels)]])

    def _process_level_transition(
        self, chunk_level: int, chunk_agg: dict, chunk_timestamp: datetime, cache: dict
    ) -> list[dict]:
        """Process chunk of trades within a single level."""
        # Initialize
        if cache["level"] is None:
            cache["timestamp"] = chunk_timestamp
            cache["level"] = chunk_level
            cache["direction"] = None
            cache["ohlcv"] = chunk_agg
            cache["wicks"] = []
            return []

        bricks = []

        # Level changed from cache level
        if chunk_level != cache["level"]:
            exit_dir = +1 if chunk_level > cache["level"] else -1
            levels_moved = abs(chunk_level - cache["level"])

            is_body = (
                cache["direction"] is None
                or cache["direction"] == exit_dir
                or levels_moved >= 2
            )

            brick = {
                "timestamp": cache["timestamp"],
                "level": cache["level"],
                "direction": exit_dir,
                "kind": RenkoKind.BODY if is_body else RenkoKind.WICK,
                "ohlcv": cache["ohlcv"],
            }

            if is_body:
                brick["sequence"] = cache["sequence"]
                cache["sequence"] += 1
                bricks.append(brick)

                for wick in cache["wicks"]:
                    wick["sequence"] = cache["sequence"]
                    cache["sequence"] += 1
                    bricks.append(wick)
                cache["wicks"] = []
            else:
                cache["wicks"].append(brick)

            cache["timestamp"] = chunk_timestamp
            cache["level"] = chunk_level
            cache["direction"] = exit_dir
            cache["ohlcv"] = None

        # Accumulate chunk data
        if cache["ohlcv"] is not None:
            cache["ohlcv"] = merge_cache(cache["ohlcv"], chunk_agg)
        else:
            cache["ohlcv"] = chunk_agg

        return bricks

    def _merge_rows(self, target: dict, source: dict) -> None:
        """Merge rows, preserving target's metadata and candle_data_id."""
        exclude = {"level", "kind", "direction", "bar_index", "candle_data_id"}
        target_ohlcv = {k: v for k, v in target.items() if k not in exclude}
        source_ohlcv = {k: v for k, v in source.items() if k not in exclude}
        target.update(merge_cache(target_ohlcv, source_ohlcv))

    def aggregate(
        self,
        timestamp_from: datetime,
        timestamp_to: datetime,
        data_frame: DataFrame,
        cache_data: dict,
    ) -> tuple[list, dict | None]:
        """Aggregate using chunked vectorized processing."""
        if data_frame.empty:
            return self.get_incomplete_candle(timestamp_to, [], cache_data)

        # Vectorized level computation
        prices = data_frame["price"].values.astype(float)
        levels = self._compute_levels(prices)
        boundaries = self._find_level_boundaries(levels)

        candle_data = []
        for i in range(len(boundaries) - 1):
            start_idx = boundaries[i]
            end_idx = boundaries[i + 1]
            chunk_df = data_frame.iloc[start_idx:end_idx]
            chunk_level = int(levels[start_idx])
            chunk_agg = aggregate_candle(chunk_df)
            chunk_timestamp = chunk_df.iloc[0]["timestamp"]

            bricks = self._process_level_transition(
                chunk_level, chunk_agg, chunk_timestamp, cache_data
            )

            for brick in bricks:
                candle = brick["ohlcv"].copy()
                candle["timestamp"] = brick["timestamp"]
                candle["_brick"] = {
                    "level": brick["level"],
                    "direction": brick["direction"],
                    "kind": brick["kind"],
                    "sequence": brick["sequence"],
                }
                candle_data.append(candle)

        return self.get_incomplete_candle(timestamp_to, candle_data, cache_data)

    def get_candle_data(
        self,
        timestamp_from: datetime | None = None,
        timestamp_to: datetime | None = None,
        is_complete: bool = True,
    ) -> Iterator[dict]:
        """Get candle data as iterator of dicts.

        Derives parent relationship by level adjacency.
        Wicks are merged into their parent.

        When is_complete=True, excludes the last brick because it may still
        accumulate pending wicks. A brick is only "complete" once the next
        brick emits, resolving any wick state. This 1-brick lag ensures
        stability for backtesting.
        """
        queryset = CandleData.objects.filter(candle=self, renko_data__isnull=False)
        if timestamp_from is not None:
            queryset = queryset.filter(timestamp__gte=timestamp_from)
        if timestamp_to is not None:
            queryset = queryset.filter(timestamp__lt=timestamp_to)
        queryset = queryset.select_related("renko_data").order_by(
            "renko_data__sequence"
        )

        pending_row = None
        body_by_level = {}

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
                wick_level = row["level"]
                parent_level = wick_level + row["direction"]
                parent_row = body_by_level.get(parent_level)
                if parent_row:
                    self._merge_rows(parent_row, row)

            elif row["kind"] == RenkoKind.BODY:
                body_by_level[row["level"]] = row

                is_same_level = (
                    pending_row is not None and row["level"] == pending_row["level"]
                )
                if is_same_level:
                    self._merge_rows(pending_row, row)
                    pending_row["candle_data_id"] = row["candle_data_id"]
                    pending_row["bar_index"] = row["bar_index"]
                    body_by_level[row["level"]] = pending_row
                else:
                    if pending_row is not None:
                        yield pending_row
                    pending_row = row

        # Handle final pending row
        if not is_complete:
            if pending_row is not None:
                yield pending_row
            cache_qs = CandleCache.objects.filter(candle=self)
            if timestamp_from is not None:
                cache_qs = cache_qs.filter(timestamp__gte=timestamp_from)
            if timestamp_to is not None:
                cache_qs = cache_qs.filter(timestamp__lt=timestamp_to)
            cache_obj = cache_qs.order_by("-timestamp").first()
            if cache_obj and cache_obj.json_data:
                incomplete_row = self._build_incomplete_brick(cache_obj.json_data)
                if incomplete_row:
                    yield incomplete_row
        # is_complete=True: don't yield pending_row (1-brick lag)

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
