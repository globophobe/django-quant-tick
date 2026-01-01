import copy
import logging
import math
from datetime import datetime
from decimal import Decimal

from django.db import models
from pandas import DataFrame

from quant_tick.constants import Direction, RenkoKind
from quant_tick.lib import aggregate_candle, merge_cache
from quant_tick.utils import gettext_lazy as _

from ..candles import CandleData
from .constant_candles import ConstantCandle

logger = logging.getLogger(__name__)


class RenkoBrick(ConstantCandle):
    """Renko brick.

    Using multiplicative percent scaling, with relative normalization.
    """

    def get_initial_cache(self, timestamp: datetime) -> dict:
        """Get initial cache with level_states for body/wick tracking."""
        cache = {
            "active_level": None,
            "last_body_direction": None,  # NEW: track last body direction (+1, -1, or None)
            "level_states": {},  # Per-level state: {level: {entry_side, aggregates, timestamp, pending_wicks}}
            "brick_sequence": 0,  # Global sequence counter for deterministic ordering
            "last_price_by_exchange": {},
        }
        # Include "date" field if cache_reset is configured
        # Required for ConstantCandle.get_incomplete_candle() to work correctly
        if "cache_reset" in self.json_data:
            cache["date"] = timestamp.date()
        return cache

    def _get_composite_price(self, row, cache_data: dict) -> Decimal:
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
        for candle_data, brick in zip(created_candle_data, brick_metadata):
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
        limit: int | None = None,
        is_complete: bool = True,
    ) -> DataFrame:
        """Get candle data with canonicalization (2-brick reversal semantics).

        Args:
            timestamp_from: Start timestamp
            timestamp_to: End timestamp
            limit: Max rows to return (applied after canonicalization)
            is_complete: If True, exclude last body brick (for ML). If False, include (for plotting).

        Returns:
            DataFrame with canonicalized body stream and wick rows
        """
        qs = (
            CandleData.objects.filter(
                candle=self, timestamp__gte=timestamp_from, timestamp__lt=timestamp_to
            )
            .select_related("renko_data")
            .order_by("renko_data__sequence")
        )

        # Canonicalization state
        last_body_row = None  # Most recent kept body row
        pending_wicks = []  # Wicks since last body
        output_rows = []  # Final canonicalized dataset

        # Iterate rows in sequence order
        for cd in qs:
            row = cd.json_data.copy()
            row["timestamp"] = cd.timestamp
            row["obj"] = cd

            try:
                renko = cd.renko_data
                row["renko_level"] = renko.level
                row["renko_kind"] = renko.kind
                row["renko_direction"] = renko.direction
                row["renko_sequence"] = renko.sequence
                row["bar_idx"] = renko.sequence
            except RenkoData.DoesNotExist:
                # Skip rows without renko_data
                continue

            if row["renko_kind"] == RenkoKind.WICK:
                # Wick row: normalize timestamp to parent body, enforce invariant
                if last_body_row is None:
                    raise ValueError("Wick row without parent body")

                # Enforce invariant: wick must be exactly 1 level from parent
                if abs(row["renko_level"] - last_body_row["renko_level"]) != 1:
                    raise ValueError(
                        f"Wick invariant violation: wick at level {row['renko_level']} "
                        f"is not exactly 1 level from parent {last_body_row['renko_level']}"
                    )

                # Set wick timestamp to parent body open timestamp
                row["timestamp"] = last_body_row["timestamp"]

                # Append wick row to output
                output_rows.append(row)
                pending_wicks.append(row)

            elif row["renko_kind"] == RenkoKind.BODY:
                # Body row: check if "return-to-previous-level" pattern
                is_return_to_prev = (
                    last_body_row is not None
                    and row["renko_level"] == last_body_row["renko_level"]
                    and row["renko_direction"] == -last_body_row["renko_direction"]
                    and len(pending_wicks) > 0
                )

                if is_return_to_prev:
                    # Merge OHLCV into last_body_row
                    # Extract OHLCV fields (exclude renko metadata)
                    ohlcv_fields = {
                        k: v
                        for k, v in last_body_row.items()
                        if k
                        not in [
                            "renko_level",
                            "renko_kind",
                            "renko_direction",
                            "renko_sequence",
                            "timestamp_end",
                            "renko_sequence_end",
                        ]
                    }

                    # Prepare dicts for merge_cache (both need timestamp)
                    prev_candle = ohlcv_fields  # Has timestamp from brick open
                    # Extract only json_data fields from current row (no renko fields)
                    curr_json_data = {
                        k: v
                        for k, v in row.items()
                        if k
                        not in [
                            "renko_level",
                            "renko_kind",
                            "renko_direction",
                            "renko_sequence",
                        ]
                    }
                    curr_candle = curr_json_data  # Already has timestamp

                    # Merge candle payloads
                    merged_candle = merge_cache(prev_candle, curr_candle)

                    # Update last_body_row's OHLCV fields only (preserve renko fields)
                    brick_open_timestamp = last_body_row["timestamp"]
                    # Only update OHLCV fields, not renko metadata
                    for k, v in merged_candle.items():
                        if k not in [
                            "renko_level",
                            "renko_kind",
                            "renko_direction",
                            "renko_sequence",
                        ]:
                            last_body_row[k] = v
                    last_body_row["timestamp"] = (
                        brick_open_timestamp  # Restore brick open
                    )

                    # Add finalization metadata
                    last_body_row["timestamp_end"] = row["timestamp"]
                    last_body_row["renko_sequence_end"] = row["renko_sequence"]

                    # renko_direction, renko_level, renko_kind, renko_sequence unchanged
                    # Clear pending wicks
                    pending_wicks.clear()

                    # Do NOT append this row (merged into last_body_row)

                else:
                    # Normal progression: new body row
                    output_rows.append(row)
                    last_body_row = row
                    pending_wicks.clear()

        # Synthesize incomplete brick from cache (if requested)
        if not is_complete:
            from quant_tick.models.candles import CandleCache

            # Get cache for this time window (not latest overall)
            cache_obj = (
                CandleCache.objects.filter(
                    candle=self,
                    timestamp__gte=timestamp_from,
                    timestamp__lt=timestamp_to,
                )
                .order_by("-timestamp")
                .first()
            )
            if cache_obj and cache_obj.json_data:
                cache = cache_obj.json_data
                active_level = cache.get("active_level")

                if active_level is not None:
                    # Keys in level_states are STRINGS (JSON serialization)
                    level_states = cache.get("level_states", {})
                    level_state = level_states.get(str(active_level))

                    if level_state and level_state.get("initialized"):
                        # Determine direction from entry_side
                        entry_side = level_state.get("entry_side")
                        if entry_side == "from_below":
                            direction = 1
                        elif entry_side == "from_above":
                            direction = -1
                        else:
                            # Seed level (entry_side=None): use last_body_direction from cache
                            direction = cache.get("last_body_direction", 1)

                        # Build synthetic incomplete brick row
                        aggregates = copy.deepcopy(level_state.get("aggregates", {}))

                        # Save active level's close price (latest trade in this level)
                        active_close = aggregates.get("close")

                        # Merge pending wick aggregates (for plotting)
                        pending_upper = level_state.get("pending_upper_wick_level")
                        pending_lower = level_state.get("pending_lower_wick_level")

                        if pending_upper is not None:
                            upper_wick_state = level_states.get(str(pending_upper))
                            if upper_wick_state and upper_wick_state.get("aggregates"):
                                # Deepcopy to avoid mutating cache JSON (_merge_multi_exchange mutates nested dicts)
                                aggregates = merge_cache(
                                    aggregates,
                                    copy.deepcopy(upper_wick_state["aggregates"]),
                                )

                        if pending_lower is not None:
                            lower_wick_state = level_states.get(str(pending_lower))
                            if lower_wick_state and lower_wick_state.get("aggregates"):
                                # Deepcopy to avoid mutating cache JSON (_merge_multi_exchange mutates nested dicts)
                                aggregates = merge_cache(
                                    aggregates,
                                    copy.deepcopy(lower_wick_state["aggregates"]),
                                )

                        # Remove timestamp from aggregates to preserve brick open time invariant
                        aggregates.pop("timestamp", None)

                        # Restore active level's close (latest trade price)
                        if active_close is not None:
                            aggregates["close"] = active_close

                        incomplete_row = {
                            "timestamp": level_state.get("timestamp"),
                            "renko_level": active_level,
                            "renko_kind": RenkoKind.BODY,
                            "renko_direction": direction,
                            "renko_sequence": cache.get(
                                "brick_sequence"
                            ),  # Next sequence (sorts last)
                            "bar_idx": cache.get("brick_sequence"),
                            **aggregates,  # Include merged OHLCV fields
                        }
                        output_rows.append(incomplete_row)

        # Filter incomplete (if requested)
        if is_complete and output_rows:
            # Find and remove the last body brick
            last_body_idx = None
            for i in range(len(output_rows) - 1, -1, -1):
                if output_rows[i]["renko_kind"] == RenkoKind.BODY:
                    last_body_idx = i
                    break

            if last_body_idx is not None:
                output_rows.pop(last_body_idx)

        # Apply limit to final output (after all processing)
        if limit and len(output_rows) > limit:
            output_rows = output_rows[:limit]

        return DataFrame(output_rows)

    def should_aggregate_candle(self, row, cache: dict) -> list[dict]:
        """Emit bricks when exiting levels, with body/wick distinction.

        Returns list of brick dicts to emit (0, 1, or many for multi-level jumps).
        Each brick dict contains trades, metadata, kind (body/wick), etc.
        """
        target_change = Decimal(str(self.json_data["target_percentage_change"]))
        composite_price = self._get_composite_price(row, cache)

        # Get origin price from config (universal grid anchor)
        origin_price = Decimal(str(self.json_data.get("origin_price", "1")))

        # Calculate which level this trade belongs to
        log_step = math.log1p(float(target_change))
        log_ratio = math.log(float(composite_price / origin_price))
        trade_level = math.floor(log_ratio / log_step)  # Use floor for symmetric grid

        # Boundary correction (prevent float rounding errors)
        # Verify: origin*(1+p)**level <= price < origin*(1+p)**(level+1)
        # Use while loop to handle extreme float drift (>1 level)
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
                break  # Correct level found

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
            # Don't emit yet - will emit when exiting level
            return []

        old_level = cache["active_level"]

        # Check if we need to emit any bricks
        bricks = []

        if trade_level != old_level:
            # Level changed: emit brick(s) for levels we're exiting

            # Determine direction and entry side for NEW level
            if trade_level > old_level:
                direction = +1
                levels_to_close = range(old_level, trade_level)
                new_entry_side = "from_below"
                exit_side = "to_above"
            else:
                direction = -1
                levels_to_close = range(old_level, trade_level, -1)
                new_entry_side = "from_above"
                exit_side = "to_below"

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
                if level_state["entry_side"] is None:
                    # Seed level (level 0 at start)
                    is_body = True  # Treat seed as body
                else:
                    # Check if opposite boundaries
                    is_body = (
                        level_state["entry_side"] == "from_below"
                        and exit_side == "to_above"
                    ) or (
                        level_state["entry_side"] == "from_above"
                        and exit_side == "to_below"
                    )

                if level_idx == 0:
                    # First level: has accumulated aggregates from this level's state
                    if is_body:
                        # Body brick: emit pending wicks first, then body

                        # Emit upper wick (if exists for this level)
                        upper_wick_level = level_state.get("pending_upper_wick_level")
                        if upper_wick_level is not None:
                            upper_wick_state = cache["level_states"][upper_wick_level]
                            upper_wick_brick = {
                                "level": upper_wick_level,
                                "direction": +1,  # Upper wick, side-based
                                "kind": RenkoKind.WICK,
                                "entry_side": upper_wick_state["entry_side"],
                                "exit_side": upper_wick_state.get("exit_side"),
                                "aggregates": upper_wick_state["aggregates"],
                                "has_trades": True,
                                "timestamp": upper_wick_state["timestamp"],
                                "sequence": cache["brick_sequence"],
                            }
                            cache["brick_sequence"] += 1
                            bricks.append(upper_wick_brick)
                            del cache["level_states"][upper_wick_level]
                            level_state["pending_upper_wick_level"] = None

                        # Emit lower wick (if exists for this level)
                        lower_wick_level = level_state.get("pending_lower_wick_level")
                        if lower_wick_level is not None:
                            lower_wick_state = cache["level_states"][lower_wick_level]
                            lower_wick_brick = {
                                "level": lower_wick_level,
                                "direction": -1,  # Lower wick, side-based
                                "kind": RenkoKind.WICK,
                                "entry_side": lower_wick_state["entry_side"],
                                "exit_side": lower_wick_state.get("exit_side"),
                                "aggregates": lower_wick_state["aggregates"],
                                "has_trades": True,
                                "timestamp": lower_wick_state["timestamp"],
                                "sequence": cache["brick_sequence"],
                            }
                            cache["brick_sequence"] += 1
                            bricks.append(lower_wick_brick)
                            del cache["level_states"][lower_wick_level]
                            level_state["pending_lower_wick_level"] = None

                        # Now emit the body brick
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

                        # Remove this level's state (body bricks are emitted)
                        if level in cache["level_states"]:
                            del cache["level_states"][level]
                    else:
                        # Wick: same-side exit at this level
                        # Assign this wick to the ADJACENT parent level (not final destination!)
                        # CRITICAL: For multi-level jumps (1→-1), assign to level 0, not level -1
                        parent_level = (
                            level + direction
                        )  # Adjacent level in direction of travel
                        parent_state = cache["level_states"].get(parent_level)

                        # Initialize parent state if it doesn't exist yet
                        if parent_state is None:
                            cache["level_states"][parent_level] = {
                                "entry_side": new_entry_side,
                                "timestamp": row["timestamp"],
                            }
                            parent_state = cache["level_states"][parent_level]

                        # Validate invariant: wicks must be exactly 1 level from parent
                        if abs(level - parent_level) != 1:
                            raise ValueError(
                                f"Wick invariant violation: wick at level {level} is "
                                f"{abs(level - parent_level)} levels away from parent {parent_level}. "
                                f"Wicks must be exactly 1 level away (2-brick reversal semantics)."
                            )

                        # Determine if this wick level is above or below parent
                        if level > parent_level:  # Upper wick
                            parent_state["pending_upper_wick_level"] = level
                        else:  # Lower wick
                            parent_state["pending_lower_wick_level"] = level
                        # DON'T delete level state - keep it for potential re-entry
                else:
                    # Intermediate level: empty (jumped over)
                    # Jumped-over levels are always body bricks (full traversal implied)

                    # CRITICAL: Flush pending wicks for THIS level before emitting the body
                    # (same logic as level_idx==0 case)
                    level_state = cache["level_states"].get(level)
                    if level_state:
                        # Emit upper wick (if exists for this level)
                        upper_wick_level = level_state.get("pending_upper_wick_level")
                        if upper_wick_level is not None:
                            upper_wick_state = cache["level_states"][upper_wick_level]
                            upper_wick_brick = {
                                "level": upper_wick_level,
                                "direction": +1,  # Upper wick, side-based
                                "kind": RenkoKind.WICK,
                                "entry_side": upper_wick_state["entry_side"],
                                "exit_side": upper_wick_state.get("exit_side"),
                                "aggregates": upper_wick_state["aggregates"],
                                "has_trades": True,
                                "timestamp": upper_wick_state["timestamp"],
                                "sequence": cache["brick_sequence"],
                            }
                            cache["brick_sequence"] += 1
                            bricks.append(upper_wick_brick)
                            del cache["level_states"][upper_wick_level]
                            level_state["pending_upper_wick_level"] = None

                        # Emit lower wick (if exists for this level)
                        lower_wick_level = level_state.get("pending_lower_wick_level")
                        if lower_wick_level is not None:
                            lower_wick_state = cache["level_states"][lower_wick_level]
                            lower_wick_brick = {
                                "level": lower_wick_level,
                                "direction": -1,  # Lower wick, side-based
                                "kind": RenkoKind.WICK,
                                "entry_side": lower_wick_state["entry_side"],
                                "exit_side": lower_wick_state.get("exit_side"),
                                "aggregates": lower_wick_state["aggregates"],
                                "has_trades": True,
                                "timestamp": lower_wick_state["timestamp"],
                                "sequence": cache["brick_sequence"],
                            }
                            cache["brick_sequence"] += 1
                            bricks.append(lower_wick_brick)
                            del cache["level_states"][lower_wick_level]
                            level_state["pending_lower_wick_level"] = None

                    # Now emit the intermediate body brick
                    brick = {
                        "level": level,
                        "direction": direction,
                        "kind": RenkoKind.BODY,  # Jumped levels are always bodies
                        "entry_side": (
                            new_entry_side if direction == +1 else "from_above"
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

        # CRITICAL: Accumulate trade in CURRENT active level's state (after level updates)
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
