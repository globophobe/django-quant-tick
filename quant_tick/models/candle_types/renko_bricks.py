import copy
import logging
import math
import sys
from datetime import datetime
from decimal import Decimal

from django.db import models
from pandas import DataFrame
from tqdm import tqdm

from quant_tick.constants import Direction, RenkoKind
from quant_tick.lib import aggregate_candle, merge_cache
from quant_tick.utils import gettext_lazy as _

from ..candles import CandleData
from .constant_candles import ConstantCandle

logger = logging.getLogger(__name__)


class EntrySide(models.TextChoices):
    """Entry side."""

    FROM_BELOW = "from_below", _("from below")
    FROM_ABOVE = "from_above", _("from above")


class ExitSide(models.TextChoices):
    """Exit side."""

    TO_ABOVE = "to_above", _("to above")
    TO_BELOW = "to_below", _("to below")


class RenkoBrick(ConstantCandle):
    """Renko brick.

    Using multiplicative percent scaling, with relative normalization.
    """

    def get_initial_cache(self, timestamp: datetime) -> dict:
        """Get initial cache with level_states for body/wick tracking."""
        cache = {
            "active_level": None,
            "last_body_direction": None,
            "level_states": {},
            "brick_sequence": 0,
            "last_price_by_exchange": {},
        }
        if "cache_reset" in self.json_data:
            cache["date"] = timestamp.date()
        return cache

    def _get_composite_price(self, row: dict, cache_data: dict) -> Decimal:
        """Calculate composite price from multi-exchange data.

        For multi-exchange: median of last prices across exchanges
        For single-exchange: just the price
        """
        # Check if this is multi-exchange data
        # Use row["exchange"], not row.exchange (attribute access can break)
        if "exchange" in row.index:
            # Update last price for this exchange
            cache_data["last_price_by_exchange"][row["exchange"]] = row["price"]

            # Calculate median of all last prices
            prices = list(cache_data["last_price_by_exchange"].values())
            if len(prices) == 0:
                return row["price"]

            # Median (robust to outliers)
            sorted_prices = sorted(prices)
            n = len(sorted_prices)
            if n % 2 == 0:
                return (sorted_prices[n // 2 - 1] + sorted_prices[n // 2]) / 2
            else:
                return sorted_prices[n // 2]
        else:
            # Single exchange: just use price
            return row["price"]

    def _emit_wick(
        self,
        cache: dict,
        level_state: dict,
        wick_key: str,
        direction: int,
    ) -> dict | None:
        """Emit pending wick and reset cache."""
        wick_level = level_state.get(wick_key)
        if wick_level is None:
            return None

        wick_state = cache["level_states"][wick_level]
        wick_brick = {
            "level": wick_level,
            "direction": direction,
            "kind": RenkoKind.WICK,
            "entry_side": wick_state["entry_side"],
            "exit_side": wick_state.get("exit_side"),
            "aggregates": wick_state["aggregates"],
            "has_trades": True,
            "timestamp": wick_state["timestamp"],
            "sequence": cache["brick_sequence"],
        }
        cache["brick_sequence"] += 1
        del cache["level_states"][wick_level]
        level_state[wick_key] = None
        return wick_brick

    def _emit_pending_wicks(
        self,
        cache: dict,
        level_state: dict,
        bricks: list,
    ) -> None:
        """Emit pending upper and lower wicks."""
        upper = self._emit_wick(cache, level_state, "pending_upper_wick_level", +1)
        if upper:
            bricks.append(upper)

        lower = self._emit_wick(cache, level_state, "pending_lower_wick_level", -1)
        if lower:
            bricks.append(lower)

    def _calculate_trade_level(
        self,
        composite_price: Decimal,
        target_change: Decimal,
        origin_price: Decimal,
    ) -> int:
        """Calculate which Renko level a trade belongs to.

        Uses multiplicative percent scaling with boundary correction
        to handle float rounding errors.
        """
        log_step = math.log1p(float(target_change))
        log_ratio = math.log(float(composite_price / origin_price))
        trade_level = math.floor(log_ratio / log_step)

        # Boundary correction loop
        while True:
            boundary_low = origin_price * (Decimal(1) + target_change) ** trade_level
            boundary_high = origin_price * (Decimal(1) + target_change) ** (
                trade_level + 1
            )
            if composite_price < boundary_low:
                trade_level -= 1
            elif composite_price >= boundary_high:
                trade_level += 1
            else:
                break
        return trade_level

    def _is_body_brick(self, entry_side: str | None, exit_side: str) -> bool:
        """Determine if a brick is a body (opposite boundaries) or wick (same side)."""
        if entry_side is None:
            return True  # Seed level is always body
        return (
            entry_side == EntrySide.FROM_BELOW and exit_side == ExitSide.TO_ABOVE
        ) or (entry_side == EntrySide.FROM_ABOVE and exit_side == ExitSide.TO_BELOW)

    # Fields to exclude when extracting OHLCV data from renko rows
    _RENKO_METADATA_FIELDS = frozenset(
        [
            "renko_level",
            "renko_kind",
            "renko_direction",
            "renko_sequence",
            "timestamp_end",
            "renko_sequence_end",
        ]
    )

    def _merge_return_body(self, last_body_row: dict, current_row: dict) -> None:
        """Merge a return-to-previous-level body into the last body row."""
        # Extract OHLCV fields (exclude renko metadata)
        ohlcv_fields = {
            k: v
            for k, v in last_body_row.items()
            if k not in self._RENKO_METADATA_FIELDS
        }

        # Extract only json_data fields from current row
        curr_json_data = {
            k: v for k, v in current_row.items() if k not in self._RENKO_METADATA_FIELDS
        }

        # Merge candle payloads
        merged_candle = merge_cache(ohlcv_fields, curr_json_data)

        # Update last_body_row's OHLCV fields only (preserve renko fields)
        brick_open_timestamp = last_body_row["timestamp"]
        for k, v in merged_candle.items():
            if k not in self._RENKO_METADATA_FIELDS:
                last_body_row[k] = v
        last_body_row["timestamp"] = brick_open_timestamp

        # Add finalization metadata
        last_body_row["timestamp_end"] = current_row["timestamp"]
        last_body_row["renko_sequence_end"] = current_row["renko_sequence"]

    def _synthesize_incomplete_brick(
        self,
        timestamp_from: datetime,
        timestamp_to: datetime,
    ) -> dict | None:
        """Synthesize incomplete brick from cache."""
        from quant_tick.models.candles import CandleCache

        cache_obj = (
            CandleCache.objects.filter(
                candle=self,
                timestamp__gte=timestamp_from,
                timestamp__lt=timestamp_to,
            )
            .order_by("-timestamp")
            .first()
        )
        if not cache_obj or not cache_obj.json_data:
            return None

        cache = cache_obj.json_data
        active_level = cache.get("active_level")
        if active_level is None:
            return None

        # Keys in level_states are STRINGS (JSON serialization)
        level_states = cache.get("level_states", {})
        level_state = level_states.get(str(active_level))
        if not level_state or not level_state.get("initialized"):
            return None

        # Determine direction from entry_side
        entry_side = level_state.get("entry_side")
        if entry_side == EntrySide.FROM_BELOW:
            direction = 1
        elif entry_side == EntrySide.FROM_ABOVE:
            direction = -1
        else:
            direction = cache.get("last_body_direction", 1)

        # Build synthetic incomplete brick row
        aggregates = copy.deepcopy(level_state.get("aggregates", {}))
        active_close = aggregates.get("close")

        # Merge pending wick aggregates (for plotting)
        for wick_key in ("pending_upper_wick_level", "pending_lower_wick_level"):
            pending_level = level_state.get(wick_key)
            if pending_level is not None:
                wick_state = level_states.get(str(pending_level))
                if wick_state and wick_state.get("aggregates"):
                    aggregates = merge_cache(
                        aggregates,
                        copy.deepcopy(wick_state["aggregates"]),
                    )

        # Remove timestamp from aggregates to preserve brick open time invariant
        aggregates.pop("timestamp", None)

        # Restore active level's close (latest trade price)
        if active_close is not None:
            aggregates["close"] = active_close

        return {
            "timestamp": level_state.get("timestamp"),
            "renko_level": active_level,
            "renko_kind": RenkoKind.BODY,
            "renko_direction": direction,
            "renko_sequence": cache.get("brick_sequence"),
            "bar_idx": cache.get("brick_sequence"),
            **aggregates,
        }

    def aggregate(
        self,
        timestamp_from: datetime,
        timestamp_to: datetime,
        data_frame: DataFrame,
        cache_data: dict,
    ) -> tuple[list, dict | None]:
        """Aggregate with emit-on-exit semantics and body/wick distinction.

        For multi-exchange: composite price determines brick boundaries,
        each exchange's OHLCV is aggregated separately in the output structure.
        """
        target_change = Decimal(str(self.json_data["target_percentage_change"]))
        candle_data = []

        for index, row in data_frame.iterrows():
            # Returns list of bricks to emit, each with its own trade list
            bricks = self.should_aggregate_candle(row, cache_data)

            for brick in bricks:
                # brick already contains aggregates from cache
                if brick["has_trades"]:
                    # Normal brick: use pre-computed aggregates
                    candle = brick["aggregates"].copy()
                    candle["timestamp"] = brick[
                        "timestamp"
                    ]  # Use first trade timestamp
                else:
                    # Empty brick: standardized schema with all expected fields
                    origin_price = Decimal(str(self.json_data.get("origin_price", "1")))
                    boundary = (
                        origin_price * (Decimal(1) + target_change) ** brick["level"]
                    )
                    candle = {
                        "timestamp": brick["timestamp"],
                        "open": boundary,
                        "high": boundary,
                        "low": boundary,
                        "close": boundary,
                        "volume": Decimal(0),
                        "buyVolume": Decimal(0),
                        "notional": Decimal(0),
                        "buyNotional": Decimal(0),
                        "ticks": 0,
                        "buyTicks": 0,
                        "roundVolume": Decimal(0),
                        "roundBuyVolume": Decimal(0),
                        "roundNotional": Decimal(0),
                        "roundBuyNotional": Decimal(0),
                        "realizedVariance": Decimal(0),
                        "exchanges": {},
                    }

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

    @staticmethod
    def _get_trades_count(brick: dict) -> int:
        """Get total trade count from aggregates."""
        if not brick["has_trades"]:
            return 0
        agg = brick["aggregates"]
        if "exchanges" in agg:
            return sum(ex["ticks"] for ex in agg["exchanges"].values())
        return agg.get("ticks", 0)

    def write_data(
        self, timestamp_from: datetime, timestamp_to: datetime, json_data: list[dict]
    ) -> None:
        """Write data with RenkoData creation."""
        from quant_tick.models.candles import CandleData

        # Delete existing data
        CandleData.objects.filter(
            candle=self, timestamp__gte=timestamp_from, timestamp__lt=timestamp_to
        ).delete()

        # Prepare CandleData and extract brick metadata
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

        # Bulk create RenkoData
        renko_data_list = []
        for candle_data, brick in zip(created_candle_data, brick_metadata, strict=True):
            if brick:
                renko_data_list.append(
                    RenkoData(
                        candle_data=candle_data,
                        level=brick["level"],
                        kind=brick["kind"],
                        direction=brick["direction"],
                        sequence=brick["sequence"],
                    )
                )

        if renko_data_list:
            RenkoData.objects.bulk_create(renko_data_list)

    def get_candle_data(
        self,
        timestamp_from: datetime,
        timestamp_to: datetime,
        is_complete: bool = True,
        progress: bool = False,
    ) -> DataFrame:
        """Get candle data."""
        queryset = (
            CandleData.objects.filter(
                candle=self,
                timestamp__gte=timestamp_from,
                timestamp__lt=timestamp_to,
                renko_data__isnull=False,
            )
            .select_related("renko_data")
            .order_by("renko_data__sequence")
        )

        last_body_row = None
        pending_wicks = []
        output_rows = []

        progress_bar = None
        if progress and sys.stderr.isatty() and "test" not in sys.argv:
            total = queryset.count()
            progress_bar = tqdm(total=total, desc="Renko bricks", unit="row")
            iterator = queryset.iterator(chunk_size=10000)
        else:
            iterator = queryset.iterator(chunk_size=10000)

        try:
            for obj in iterator:
                renko = obj.renko_data
                row = obj.json_data.copy()
                row["timestamp"] = obj.timestamp
                row["obj"] = obj
                row["renko_level"] = renko.level
                row["renko_kind"] = renko.kind
                row["renko_direction"] = renko.direction
                row["renko_sequence"] = renko.sequence
                row["bar_idx"] = renko.sequence

                if row["renko_kind"] == RenkoKind.WICK:
                    if last_body_row is None:
                        raise ValueError("Wick row without parent body")

                    # Wick must be exactly 1 level from parent
                    if abs(row["renko_level"] - last_body_row["renko_level"]) != 1:
                        raise ValueError(
                            f"Wick invariant violation: wick at level {row['renko_level']} "
                            f"is not exactly 1 level from parent {last_body_row['renko_level']}"
                        )

                    row["timestamp"] = last_body_row["timestamp"]
                    output_rows.append(row)
                    pending_wicks.append(row)

                elif row["renko_kind"] == RenkoKind.BODY:
                    is_return_to_prev = (
                        last_body_row is not None
                        and row["renko_level"] == last_body_row["renko_level"]
                        and row["renko_direction"] == -last_body_row["renko_direction"]
                        and len(pending_wicks) > 0
                    )
                    if is_return_to_prev:
                        self._merge_return_body(last_body_row, row)
                        pending_wicks.clear()
                    else:
                        output_rows.append(row)
                        last_body_row = row
                        pending_wicks.clear()

                if progress_bar:
                    progress_bar.update(1)

            # Synthesize incomplete brick from cache
            if not is_complete:
                incomplete_row = self._synthesize_incomplete_brick(
                    timestamp_from, timestamp_to
                )
                if incomplete_row:
                    output_rows.append(incomplete_row)

            # Filter incomplete
            if is_complete and output_rows:
                # Find and remove the last body brick
                last_body_idx = None
                for i in range(len(output_rows) - 1, -1, -1):
                    if output_rows[i]["renko_kind"] == RenkoKind.BODY:
                        last_body_idx = i
                        break

                if last_body_idx is not None:
                    output_rows.pop(last_body_idx)
        finally:
            if progress_bar:
                progress_bar.close()

        return DataFrame(output_rows)

    def should_aggregate_candle(self, row: dict, cache: dict) -> list[dict]:
        """Should aggregate candle."""
        target_change = Decimal(str(self.json_data["target_percentage_change"]))
        origin_price = Decimal(str(self.json_data.get("origin_price", "1")))
        composite_price = self._get_composite_price(row, cache)
        trade_level = self._calculate_trade_level(
            composite_price, target_change, origin_price
        )

        # Initialize on first trade
        if cache["active_level"] is None:
            cache["active_level"] = trade_level
            cache["level_states"] = {
                trade_level: {
                    "entry_side": None,  # Seed level has no entry side
                    "aggregates": aggregate_candle(DataFrame([row.to_dict()])),
                    "initialized": True,
                    "timestamp": row["timestamp"],
                }
            }
            return []

        bricks = []
        old_level = cache["active_level"]
        if trade_level != old_level:
            # Emit bricks for levels we're exiting

            if trade_level > old_level:
                direction = +1
                levels_to_close = range(old_level, trade_level)
                new_entry_side = EntrySide.FROM_BELOW
                exit_side = ExitSide.TO_ABOVE
            else:
                direction = -1
                levels_to_close = range(old_level, trade_level, -1)
                new_entry_side = EntrySide.FROM_ABOVE
                exit_side = ExitSide.TO_BELOW

            # Close each level in sequence
            for level_idx, level in enumerate(levels_to_close):
                level_state = cache["level_states"].get(
                    level,
                    {
                        "entry_side": None,
                        "trades": [],
                        "timestamp": row["timestamp"],
                    },
                )

                # Determine body vs wick
                is_body = self._is_body_brick(level_state["entry_side"], exit_side)

                if level_idx == 0:
                    if is_body:
                        # Emit pending wicks first, then body
                        self._emit_pending_wicks(cache, level_state, bricks)

                        brick = {
                            "level": level,
                            "direction": direction,
                            "kind": RenkoKind.BODY,
                            "entry_side": level_state["entry_side"],
                            "exit_side": exit_side,
                            "aggregates": level_state.get("aggregates"),
                            "has_trades": level_state.get("initialized", False),
                            "timestamp": level_state["timestamp"],
                            "sequence": cache["brick_sequence"],
                        }

                        cache["brick_sequence"] += 1
                        bricks.append(brick)
                        cache["last_body_direction"] = direction
                        if level in cache["level_states"]:
                            del cache["level_states"][level]
                    else:
                        # Wick: same-side exit at this level
                        parent_level = (
                            level + direction
                        )  # Adjacent level in direction of travel
                        parent_state = cache["level_states"].get(parent_level)

                        if parent_state is None:
                            cache["level_states"][parent_level] = {
                                "entry_side": new_entry_side,
                                "timestamp": row["timestamp"],
                            }
                            parent_state = cache["level_states"][parent_level]

                        # Wicks must be exactly 1 level from parent
                        if abs(level - parent_level) != 1:
                            raise ValueError(
                                f"Wick invariant violation: wick at level {level} is "
                                f"{abs(level - parent_level)} levels away from parent {parent_level}. "
                                f"Wicks must be exactly 1 level away (2-brick reversal semantics)."
                            )

                        # Determine if wick is above or below
                        if level > parent_level:
                            parent_state["pending_upper_wick_level"] = level
                        else:
                            parent_state["pending_lower_wick_level"] = level
                else:
                    # Flush pending wicks for this level before emitting the body
                    level_state = cache["level_states"].get(level)
                    if level_state:
                        self._emit_pending_wicks(cache, level_state, bricks)

                    # Now emit the intermediate body brick
                    brick = {
                        "level": level,
                        "direction": direction,
                        "kind": RenkoKind.BODY,  # Jumped levels are always bodies
                        "entry_side": (
                            new_entry_side if direction == +1 else EntrySide.FROM_ABOVE
                        ),
                        "exit_side": exit_side,
                        "aggregates": None,
                        "has_trades": False,
                        "timestamp": row[
                            "timestamp"
                        ],  # Boundary-crossing trade timestamp
                        "sequence": cache["brick_sequence"],
                    }
                    cache["brick_sequence"] += 1
                    bricks.append(brick)

            # Initialize state for NEW level we're entering
            cache["active_level"] = trade_level
            if trade_level not in cache["level_states"]:
                cache["level_states"][trade_level] = {
                    "entry_side": new_entry_side,
                    "timestamp": row["timestamp"],
                }

        # Accumulate trade in current active level's state
        level_state = cache["level_states"][cache["active_level"]]
        new_agg = aggregate_candle(DataFrame([row.to_dict()]))
        if level_state.get("initialized"):
            level_state["aggregates"] = merge_cache(level_state["aggregates"], new_agg)
        else:
            level_state["aggregates"] = new_agg
            level_state["initialized"] = True

        return bricks

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
    kind = models.CharField(_("kind"), max_length=4, choices=RenkoKind.choices)
    direction = models.SmallIntegerField(_("direction"), choices=Direction.choices)

    class Meta:
        db_table = "quant_tick_renko_data"
        indexes = [
            models.Index(fields=["level", "sequence"]),
        ]
        ordering = ["sequence"]
        verbose_name = verbose_name_plural = _("renko data")
