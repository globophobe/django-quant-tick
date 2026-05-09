from decimal import Decimal

from django.conf import settings
from django.contrib import admin
from django.contrib.humanize.templatetags.humanize import intcomma
from django.template.defaultfilters import floatformat
from django.urls import reverse
from django.utils.html import format_html

from quant_tick.filters import (
    CandleCacheFilter,
    CandleDataFilter,
    CandleFilter,
    ExchangeCandleDataFilter,
    FundingDataFilter,
    HAS_SEMANTIC_FILTERS,
    SymbolFilter,
    TaskStateFilter,
    TradeDataFilter,
)
from quant_tick.models import (
    Candle,
    CandleCache,
    CandleData,
    ExchangeCandleData,
    FundingData,
    Symbol,
    TaskState,
    TradeData,
)

USE_SEMANTIC_ADMIN = "semantic_admin" in settings.INSTALLED_APPS
if USE_SEMANTIC_ADMIN:
    from semantic_admin import SemanticModelAdmin
    ModelAdmin = SemanticModelAdmin
else:
    ModelAdmin = admin.ModelAdmin
USE_SEMANTIC_FILTERS = USE_SEMANTIC_ADMIN and HAS_SEMANTIC_FILTERS


def get_list_filter(*fields) -> tuple:
    """Use semantic filtersets instead of Django's stock sidebar filters."""
    return () if USE_SEMANTIC_FILTERS else fields


def format_display_number(value) -> str:
    if value is None:
        return "-"
    return intcomma(floatformat(value, 2))


def format_display_rate(value) -> str:
    if value is None:
        return "-"
    value = value.quantize(Decimal("0.00000001"))
    return format(value.normalize(), "f")


def format_admin_change_link(obj):
    if obj is None:
        return "-"
    meta = obj._meta
    view_name = f"admin:{meta.app_label}_{meta.model_name}_change"
    url = reverse(view_name, args=(obj.pk,))
    return format_html('<a href="{}">{}</a>', url, obj)


class DirectSymbolLinkMixin:
    @admin.display(description="symbol", ordering="symbol")
    def symbol_link(self, obj):
        return format_admin_change_link(obj.symbol)


class CandleSymbolLinkMixin:
    @admin.display(description="symbol", ordering="candle__symbol")
    def symbol_link(self, obj):
        return format_admin_change_link(obj.candle.symbol)


class CandleLinkMixin:
    @admin.display(description="candle", ordering="candle")
    def candle_link(self, obj):
        return format_admin_change_link(obj.candle)


class CandleSymbolListFilter(admin.SimpleListFilter):
    title = "symbol"
    parameter_name = "symbol"

    def lookups(self, request, model_admin):
        symbols = (
            Symbol.objects.filter(is_active=True, candle__isnull=False)
            .distinct()
            .order_by("exchange", "api_symbol")
        )
        return [(symbol.pk, str(symbol)) for symbol in symbols]

    def queryset(self, request, queryset):
        if self.value():
            return queryset.filter(candle__symbol_id=self.value())
        return queryset


class ReadOnlyAdmin(ModelAdmin):
    actions = None

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False

    def has_delete_permission(self, request, obj=None):
        return False


class CreateOnlyAdmin(ModelAdmin):
    def has_change_permission(self, request, obj=None):
        return obj is None

    def has_delete_permission(self, request, obj=None):
        return False


class NoCreateAdmin(ModelAdmin):
    def has_add_permission(self, request):
        return False


class SymbolAdmin(CreateOnlyAdmin):
    filterset_class = SymbolFilter
    list_display = (
        "code_name",
        "exchange",
        "api_symbol",
        "symbol_type",
        "exchange_candle_resolution",
        "date_from",
        "is_active",
    )
    list_filter = get_list_filter(
        "exchange",
        "symbol_type",
        "is_active",
        "save_raw",
        "save_aggregated",
    )


class CandleAdmin(DirectSymbolLinkMixin, CreateOnlyAdmin):
    filterset_class = CandleFilter
    list_display = ("code_name", "symbol_link", "date_from", "date_to", "is_active")
    list_filter = get_list_filter("symbol", "is_active")
    list_select_related = ("symbol",)


class TaskStateAdmin(NoCreateAdmin):
    filterset_class = TaskStateFilter
    list_display = (
        "task_type",
        "exchange",
        "api_symbol",
        "recent_error_count",
        "recent_error_at",
        "next_fetch_at",
        "locked_until",
    )
    list_filter = get_list_filter("exchange", "api_symbol", "task_type")


class TradeDataAdmin(DirectSymbolLinkMixin, ReadOnlyAdmin):
    filterset_class = TradeDataFilter
    list_display = ("timestamp", "symbol_link", "frequency", "ok")
    list_filter = get_list_filter("symbol", "frequency", "ok")
    ordering = ("-timestamp",)
    list_select_related = ("symbol",)


class CandleDataAdmin(CandleSymbolLinkMixin, CandleLinkMixin, ReadOnlyAdmin):
    filterset_class = CandleDataFilter
    list_display = (
        "timestamp",
        "symbol_link",
        "candle_link",
        "open_display",
        "high_display",
        "low_display",
        "close_display",
        "buy_volume_display",
        "volume_display",
    )
    list_filter = get_list_filter(CandleSymbolListFilter)
    ordering = ("-timestamp",)
    list_select_related = ("candle", "candle__symbol")

    @admin.display(description="open", ordering="open")
    def open_display(self, obj):
        return format_display_number(obj.open)

    @admin.display(description="high", ordering="high")
    def high_display(self, obj):
        return format_display_number(obj.high)

    @admin.display(description="low", ordering="low")
    def low_display(self, obj):
        return format_display_number(obj.low)

    @admin.display(description="close", ordering="close")
    def close_display(self, obj):
        return format_display_number(obj.close)

    @admin.display(description="volume", ordering="volume")
    def volume_display(self, obj):
        return format_display_number(obj.volume)

    @admin.display(description="buy volume", ordering="buy_volume")
    def buy_volume_display(self, obj):
        return format_display_number(obj.buy_volume)


class CandleCacheAdmin(CandleSymbolLinkMixin, CandleLinkMixin, ReadOnlyAdmin):
    filterset_class = CandleCacheFilter
    list_display = ("timestamp", "symbol_link", "candle_link", "frequency")
    list_filter = get_list_filter(
        CandleSymbolListFilter,
        "candle",
        "frequency",
    )
    list_select_related = ("candle", "candle__symbol")


class ExchangeCandleDataAdmin(DirectSymbolLinkMixin, ReadOnlyAdmin):
    filterset_class = ExchangeCandleDataFilter
    list_display = (
        "timestamp",
        "symbol_link",
        "frequency",
        "open_display",
        "high_display",
        "low_display",
        "close_display",
        "volume_display",
        "notional_display",
    )
    list_filter = get_list_filter("symbol", "frequency")
    list_select_related = ("symbol",)

    @admin.display(description="open", ordering="open")
    def open_display(self, obj):
        return format_display_number(obj.open)

    @admin.display(description="high", ordering="high")
    def high_display(self, obj):
        return format_display_number(obj.high)

    @admin.display(description="low", ordering="low")
    def low_display(self, obj):
        return format_display_number(obj.low)

    @admin.display(description="close", ordering="close")
    def close_display(self, obj):
        return format_display_number(obj.close)

    @admin.display(description="volume", ordering="volume")
    def volume_display(self, obj):
        return format_display_number(obj.volume)

    @admin.display(description="notional", ordering="notional")
    def notional_display(self, obj):
        return format_display_number(obj.notional)


class FundingDataAdmin(DirectSymbolLinkMixin, ReadOnlyAdmin):
    filterset_class = FundingDataFilter
    list_display = ("timestamp", "symbol_link", "funding_rate_display")
    list_filter = get_list_filter("symbol")
    list_select_related = ("symbol",)

    @admin.display(description="funding rate", ordering="funding_rate")
    def funding_rate_display(self, obj):
        return format_display_rate(obj.funding_rate)


if "django.contrib.admin" in settings.INSTALLED_APPS:
    admin.site.register(Symbol, SymbolAdmin)
    admin.site.register(Candle, CandleAdmin)
    admin.site.register(TaskState, TaskStateAdmin)
    admin.site.register(TradeData, TradeDataAdmin)
    admin.site.register(CandleData, CandleDataAdmin)
    admin.site.register(CandleCache, CandleCacheAdmin)
    admin.site.register(ExchangeCandleData, ExchangeCandleDataAdmin)
    admin.site.register(FundingData, FundingDataAdmin)
