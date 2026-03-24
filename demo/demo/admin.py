from datetime import datetime, timezone

from django.contrib import admin
from django.contrib.auth.models import Group, User
from django.http import HttpRequest, JsonResponse
from django.template.response import TemplateResponse
from django.urls import path
from django.utils.dateparse import parse_datetime
from semantic_admin import SemanticModelAdmin, SemanticTabularInline

from quant_tick.models import Candle, GlobalSymbol, Symbol

admin.site.unregister(User)
admin.site.unregister(Group)


class SymbolInline(SemanticTabularInline):
    """Symbol inline."""

    model = Symbol
    extra = 0


@admin.register(GlobalSymbol)
class GlobalSymbolAdmin(SemanticModelAdmin):
    """Global symbol admin."""

    list_display = ("__str__",)
    fields = ("name",)
    inlines = (SymbolInline,)


@admin.register(Symbol)
class SymbolAdmin(SemanticModelAdmin):
    """Symbol admin."""

    list_display = ("__str__",)
    fields = (
        ("global_symbol", "exchange"),
        ("symbol_type", "api_symbol"),
        ("aggregate_trades", "significant_trade_filter"),
    )


@admin.register(Candle)
class CandleAdmin(SemanticModelAdmin):
    """Candle admin."""

    list_display = ("code_name", "symbol", "is_active")
    list_filter = ("is_active", "symbol")

    def get_urls(self) -> list:
        """Return custom admin URLs."""
        urls = super().get_urls()
        custom_urls = [
            path(
                "chart/<str:code_name>/",
                self.admin_site.admin_view(self.chart_view),
                name="candle_chart",
            ),
            path(
                "chart/<str:code_name>/data/",
                self.admin_site.admin_view(self.chart_data_api),
                name="candle_chart_data",
            ),
        ]
        return custom_urls + urls

    def chart_view(self, request: HttpRequest, code_name: str) -> TemplateResponse:
        """Render interactive order flow chart."""
        import json

        from quant_tick.models import CandleData

        # Fetch latest 100 candles for display
        candles_qs = (
            CandleData.objects.filter(candle__code_name=code_name)
            .order_by("-timestamp")[:100]
        )

        display_candles = list(reversed(list(candles_qs)))

        # Calculate actual year highs/lows from full dataset
        if display_candles:
            # Find the actual latest year with data (not calendar year)
            latest_candle = CandleData.objects.filter(
                candle__code_name=code_name
            ).order_by("-timestamp").first()

            if latest_candle:
                current_year = latest_candle.timestamp.year
                last_year = current_year - 1
                two_years_ago = current_year - 2
            else:
                current_year = display_candles[-1].timestamp.year
                last_year = current_year - 1
                two_years_ago = current_year - 2

            # Get current year extremes
            current_year_start = datetime(current_year, 1, 1, tzinfo=timezone.utc)
            current_year_end = datetime(current_year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

            yearly_anchors = []

            # Find actual max high in current year from all candles
            current_year_candles = list(
                CandleData.objects.filter(
                    candle__code_name=code_name,
                    timestamp__gte=current_year_start,
                    timestamp__lte=current_year_end,
                )
            )

            current_high_candle = None
            max_high = -float("inf")
            for candle in current_year_candles:
                high_price = float(candle.json_data.get("high", 0))
                if high_price > max_high:
                    max_high = high_price
                    current_high_candle = candle

            # Find actual min low in current year
            current_low_candle = None
            min_low = float("inf")
            for candle in current_year_candles:
                low_price = float(candle.json_data.get("low", 0))
                if low_price > 0 and low_price < min_low:
                    min_low = low_price
                    current_low_candle = candle

            if current_high_candle:
                # Calculate cumulative values from anchor to first visible candle
                anchor_timestamp = current_high_candle.timestamp
                first_visible_timestamp = display_candles[0].timestamp

                cumulative_tpv = 0
                cumulative_volume = 0
                initial_avwap = None

                if anchor_timestamp < first_visible_timestamp:
                    # Get all candles from anchor to first visible
                    candles_before = CandleData.objects.filter(
                        candle__code_name=code_name,
                        timestamp__gte=anchor_timestamp,
                        timestamp__lt=first_visible_timestamp,
                    ).order_by("timestamp")

                    for candle in candles_before:
                        high = float(candle.json_data.get("high", 0))
                        low = float(candle.json_data.get("low", 0))
                        close = float(candle.json_data.get("close", 0))
                        volume = float(candle.json_data.get("volume", 0))

                        if volume > 0:
                            typical_price = (high + low + close) / 3
                            cumulative_tpv += typical_price * volume
                            cumulative_volume += volume

                    # Calculate AVWAP value at first visible candle
                    if cumulative_volume > 0:
                        initial_avwap = cumulative_tpv / cumulative_volume

                yearly_anchors.append(
                    {
                        "type": "current_high",
                        "label": f"{current_year} High",
                        "price": float(current_high_candle.json_data.get("high", 0)),
                        "timestamp": current_high_candle.timestamp.isoformat(),
                        "initialAVWAP": initial_avwap,
                        "cumulativeTPV": cumulative_tpv,
                        "cumulativeVolume": cumulative_volume,
                    }
                )

            if current_low_candle:
                # Calculate cumulative values from anchor to first visible candle
                anchor_timestamp = current_low_candle.timestamp
                first_visible_timestamp = display_candles[0].timestamp

                cumulative_tpv = 0
                cumulative_volume = 0
                initial_avwap = None

                if anchor_timestamp < first_visible_timestamp:
                    # Get all candles from anchor to first visible
                    candles_before = CandleData.objects.filter(
                        candle__code_name=code_name,
                        timestamp__gte=anchor_timestamp,
                        timestamp__lt=first_visible_timestamp,
                    ).order_by("timestamp")

                    for candle in candles_before:
                        high = float(candle.json_data.get("high", 0))
                        low = float(candle.json_data.get("low", 0))
                        close = float(candle.json_data.get("close", 0))
                        volume = float(candle.json_data.get("volume", 0))

                        if volume > 0:
                            typical_price = (high + low + close) / 3
                            cumulative_tpv += typical_price * volume
                            cumulative_volume += volume

                    # Calculate AVWAP value at first visible candle
                    if cumulative_volume > 0:
                        initial_avwap = cumulative_tpv / cumulative_volume

                yearly_anchors.append(
                    {
                        "type": "current_low",
                        "label": f"{current_year} Low",
                        "price": float(current_low_candle.json_data.get("low", 0)),
                        "timestamp": current_low_candle.timestamp.isoformat(),
                        "initialAVWAP": initial_avwap,
                        "cumulativeTPV": cumulative_tpv,
                        "cumulativeVolume": cumulative_volume,
                    }
                )

            # Get last year extremes
            last_year_start = datetime(last_year, 1, 1, tzinfo=timezone.utc)
            last_year_end = datetime(last_year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

            # Find actual max high and min low in last year from all candles
            last_year_candles = list(
                CandleData.objects.filter(
                    candle__code_name=code_name,
                    timestamp__gte=last_year_start,
                    timestamp__lte=last_year_end,
                )
            )

            last_high_candle = None
            max_high = -float("inf")
            for candle in last_year_candles:
                high_price = float(candle.json_data.get("high", 0))
                if high_price > max_high:
                    max_high = high_price
                    last_high_candle = candle

            last_low_candle = None
            min_low = float("inf")
            for candle in last_year_candles:
                low_price = float(candle.json_data.get("low", 0))
                if low_price > 0 and low_price < min_low:
                    min_low = low_price
                    last_low_candle = candle

            if last_high_candle:
                # Calculate cumulative values from anchor to first visible candle
                anchor_timestamp = last_high_candle.timestamp
                first_visible_timestamp = display_candles[0].timestamp

                cumulative_tpv = 0
                cumulative_volume = 0
                initial_avwap = None

                if anchor_timestamp < first_visible_timestamp:
                    # Get all candles from anchor to first visible
                    candles_before = CandleData.objects.filter(
                        candle__code_name=code_name,
                        timestamp__gte=anchor_timestamp,
                        timestamp__lt=first_visible_timestamp,
                    ).order_by("timestamp")

                    for candle in candles_before:
                        high = float(candle.json_data.get("high", 0))
                        low = float(candle.json_data.get("low", 0))
                        close = float(candle.json_data.get("close", 0))
                        volume = float(candle.json_data.get("volume", 0))

                        if volume > 0:
                            typical_price = (high + low + close) / 3
                            cumulative_tpv += typical_price * volume
                            cumulative_volume += volume

                    # Calculate AVWAP value at first visible candle
                    if cumulative_volume > 0:
                        initial_avwap = cumulative_tpv / cumulative_volume

                yearly_anchors.append(
                    {
                        "type": "last_high",
                        "label": f"{last_year} High",
                        "price": float(last_high_candle.json_data.get("high", 0)),
                        "timestamp": last_high_candle.timestamp.isoformat(),
                        "initialAVWAP": initial_avwap,
                        "cumulativeTPV": cumulative_tpv,
                        "cumulativeVolume": cumulative_volume,
                    }
                )

            if last_low_candle:
                # Calculate cumulative values from anchor to first visible candle
                anchor_timestamp = last_low_candle.timestamp
                first_visible_timestamp = display_candles[0].timestamp

                cumulative_tpv = 0
                cumulative_volume = 0
                initial_avwap = None

                if anchor_timestamp < first_visible_timestamp:
                    # Get all candles from anchor to first visible
                    candles_before = CandleData.objects.filter(
                        candle__code_name=code_name,
                        timestamp__gte=anchor_timestamp,
                        timestamp__lt=first_visible_timestamp,
                    ).order_by("timestamp")

                    for candle in candles_before:
                        high = float(candle.json_data.get("high", 0))
                        low = float(candle.json_data.get("low", 0))
                        close = float(candle.json_data.get("close", 0))
                        volume = float(candle.json_data.get("volume", 0))

                        if volume > 0:
                            typical_price = (high + low + close) / 3
                            cumulative_tpv += typical_price * volume
                            cumulative_volume += volume

                    # Calculate AVWAP value at first visible candle
                    if cumulative_volume > 0:
                        initial_avwap = cumulative_tpv / cumulative_volume

                yearly_anchors.append(
                    {
                        "type": "last_low",
                        "label": f"{last_year} Low",
                        "price": float(last_low_candle.json_data.get("low", 0)),
                        "timestamp": last_low_candle.timestamp.isoformat(),
                        "initialAVWAP": initial_avwap,
                        "cumulativeTPV": cumulative_tpv,
                        "cumulativeVolume": cumulative_volume,
                    }
                )

            # Get two years ago extremes
            two_years_ago_start = datetime(two_years_ago, 1, 1, tzinfo=timezone.utc)
            two_years_ago_end = datetime(two_years_ago, 12, 31, 23, 59, 59, tzinfo=timezone.utc)

            # Find actual max high and min low in two years ago from all candles
            two_years_ago_candles = list(
                CandleData.objects.filter(
                    candle__code_name=code_name,
                    timestamp__gte=two_years_ago_start,
                    timestamp__lte=two_years_ago_end,
                )
            )

            two_years_ago_high_candle = None
            max_high = -float("inf")
            for candle in two_years_ago_candles:
                high_price = float(candle.json_data.get("high", 0))
                if high_price > max_high:
                    max_high = high_price
                    two_years_ago_high_candle = candle

            two_years_ago_low_candle = None
            min_low = float("inf")
            for candle in two_years_ago_candles:
                low_price = float(candle.json_data.get("low", 0))
                if low_price > 0 and low_price < min_low:
                    min_low = low_price
                    two_years_ago_low_candle = candle

            if two_years_ago_high_candle:
                # Calculate cumulative values from anchor to first visible candle
                anchor_timestamp = two_years_ago_high_candle.timestamp
                first_visible_timestamp = display_candles[0].timestamp

                cumulative_tpv = 0
                cumulative_volume = 0
                initial_avwap = None

                if anchor_timestamp < first_visible_timestamp:
                    # Get all candles from anchor to first visible
                    candles_before = CandleData.objects.filter(
                        candle__code_name=code_name,
                        timestamp__gte=anchor_timestamp,
                        timestamp__lt=first_visible_timestamp,
                    ).order_by("timestamp")

                    for candle in candles_before:
                        high = float(candle.json_data.get("high", 0))
                        low = float(candle.json_data.get("low", 0))
                        close = float(candle.json_data.get("close", 0))
                        volume = float(candle.json_data.get("volume", 0))

                        if volume > 0:
                            typical_price = (high + low + close) / 3
                            cumulative_tpv += typical_price * volume
                            cumulative_volume += volume

                    # Calculate AVWAP value at first visible candle
                    if cumulative_volume > 0:
                        initial_avwap = cumulative_tpv / cumulative_volume

                yearly_anchors.append(
                    {
                        "type": "two_years_ago_high",
                        "label": f"{two_years_ago} High",
                        "price": float(two_years_ago_high_candle.json_data.get("high", 0)),
                        "timestamp": two_years_ago_high_candle.timestamp.isoformat(),
                        "initialAVWAP": initial_avwap,
                        "cumulativeTPV": cumulative_tpv,
                        "cumulativeVolume": cumulative_volume,
                    }
                )

            if two_years_ago_low_candle:
                # Calculate cumulative values from anchor to first visible candle
                anchor_timestamp = two_years_ago_low_candle.timestamp
                first_visible_timestamp = display_candles[0].timestamp

                cumulative_tpv = 0
                cumulative_volume = 0
                initial_avwap = None

                if anchor_timestamp < first_visible_timestamp:
                    # Get all candles from anchor to first visible
                    candles_before = CandleData.objects.filter(
                        candle__code_name=code_name,
                        timestamp__gte=anchor_timestamp,
                        timestamp__lt=first_visible_timestamp,
                    ).order_by("timestamp")

                    for candle in candles_before:
                        high = float(candle.json_data.get("high", 0))
                        low = float(candle.json_data.get("low", 0))
                        close = float(candle.json_data.get("close", 0))
                        volume = float(candle.json_data.get("volume", 0))

                        if volume > 0:
                            typical_price = (high + low + close) / 3
                            cumulative_tpv += typical_price * volume
                            cumulative_volume += volume

                    # Calculate AVWAP value at first visible candle
                    if cumulative_volume > 0:
                        initial_avwap = cumulative_tpv / cumulative_volume

                yearly_anchors.append(
                    {
                        "type": "two_years_ago_low",
                        "label": f"{two_years_ago} Low",
                        "price": float(two_years_ago_low_candle.json_data.get("low", 0)),
                        "timestamp": two_years_ago_low_candle.timestamp.isoformat(),
                        "initialAVWAP": initial_avwap,
                        "cumulativeTPV": cumulative_tpv,
                        "cumulativeVolume": cumulative_volume,
                    }
                )
        else:
            yearly_anchors = []

        chart_data = []
        for cd in display_candles:
            chart_data.append(
                {
                    "timestamp": cd.timestamp.isoformat(),
                    "open": float(cd.json_data.get("open", 0)),
                    "high": float(cd.json_data.get("high", 0)),
                    "low": float(cd.json_data.get("low", 0)),
                    "close": float(cd.json_data.get("close", 0)),
                    "volume": float(cd.json_data.get("volume", 0)),
                    "buyVolume": float(cd.json_data.get("buyVolume", 0)),
                    "roundNotional": float(cd.json_data.get("roundNotional", 0)),
                    "roundBuyNotional": float(cd.json_data.get("roundBuyNotional", 0)),
                    "realizedVolatility": float(cd.json_data.get("realizedVariance", 0)) ** 0.5,
                }
            )

        context = self.admin_site.each_context(request)
        context["title"] = f"Order Flow - {code_name}"
        context["code_name"] = code_name
        context["candle_data"] = json.dumps(chart_data, default=str)
        context["yearly_anchors"] = json.dumps(yearly_anchors, default=str)
        return TemplateResponse(request, "admin/order_flow_chart.html", context)

    def chart_data_api(self, request: HttpRequest, code_name: str) -> JsonResponse:
        """API endpoint to fetch candle data chunks for lazy loading."""
        from quant_tick.models import CandleData

        direction = request.GET.get("direction", "older")
        limit = int(request.GET.get("limit", 100))
        before_timestamp = request.GET.get("before")
        after_timestamp = request.GET.get("after")

        queryset = CandleData.objects.filter(candle__code_name=code_name)

        if direction == "older" and before_timestamp:
            before_timestamp = before_timestamp.replace(" ", "+")
            before_dt = parse_datetime(before_timestamp)
            if not before_dt:
                try:
                    before_dt = datetime.fromisoformat(before_timestamp)
                except ValueError:
                    before_dt = datetime.strptime(before_timestamp, "%Y-%m-%d").replace(tzinfo=timezone.utc)

            queryset = queryset.filter(
                timestamp__lt=before_dt
            ).order_by("-timestamp")[:limit]
            candles = list(reversed(list(queryset)))
        elif direction == "newer" and after_timestamp:
            after_timestamp = after_timestamp.replace(" ", "+")
            after_dt = parse_datetime(after_timestamp)
            if not after_dt:
                try:
                    after_dt = datetime.fromisoformat(after_timestamp)
                except ValueError:
                    after_dt = datetime.strptime(after_timestamp, "%Y-%m-%d").replace(tzinfo=timezone.utc)

            queryset = queryset.filter(
                timestamp__gt=after_dt
            ).order_by("timestamp")[:limit]
            candles = list(queryset)
        else:
            queryset = queryset.order_by("-timestamp")[:limit]
            candles = list(reversed(list(queryset)))

        chart_data = []
        for cd in candles:
            chart_data.append(
                {
                    "timestamp": cd.timestamp.isoformat(),
                    "open": float(cd.json_data.get("open", 0)),
                    "high": float(cd.json_data.get("high", 0)),
                    "low": float(cd.json_data.get("low", 0)),
                    "close": float(cd.json_data.get("close", 0)),
                    "volume": float(cd.json_data.get("volume", 0)),
                    "buyVolume": float(cd.json_data.get("buyVolume", 0)),
                    "roundNotional": float(cd.json_data.get("roundNotional", 0)),
                    "roundBuyNotional": float(cd.json_data.get("roundBuyNotional", 0)),
                    "realizedVolatility": float(cd.json_data.get("realizedVariance", 0)) ** 0.5,
                }
            )

        return JsonResponse({"candles": chart_data})
