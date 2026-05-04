from django.conf import settings
from django.contrib import admin

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

try:
    from semantic_admin import SemanticModelAdmin
except ImportError:
    ModelAdmin = admin.ModelAdmin
else:
    ModelAdmin = (
        SemanticModelAdmin
        if "semantic_admin" in settings.INSTALLED_APPS
        else admin.ModelAdmin
    )


class ReadOnlyAdmin(ModelAdmin):
    actions = None

    def has_add_permission(self, request):
        return False

    def has_change_permission(self, request, obj=None):
        return False

    def has_delete_permission(self, request, obj=None):
        return False


class SymbolAdmin(ModelAdmin):
    list_display = (
        "code_name",
        "exchange",
        "api_symbol",
        "symbol_type",
        "exchange_candle_resolution",
        "date_from",
        "is_active",
    )
    list_filter = (
        "exchange",
        "symbol_type",
        "is_active",
        "save_raw",
        "save_aggregated",
    )
    search_fields = ("code_name", "api_symbol")


class CandleAdmin(ModelAdmin):
    list_display = ("code_name", "symbol", "date_from", "date_to", "is_active")
    list_filter = ("is_active", "symbol__exchange")
    search_fields = ("code_name", "symbol__code_name", "symbol__api_symbol")
    autocomplete_fields = ("symbol",)


class TaskStateAdmin(ModelAdmin):
    list_display = (
        "task_type",
        "exchange",
        "recent_error_count",
        "recent_error_at",
        "next_fetch_at",
        "locked_until",
    )
    list_filter = ("task_type", "exchange")
    search_fields = ("task_type", "exchange")


class TradeDataAdmin(ReadOnlyAdmin):
    list_display = ("symbol", "timestamp", "frequency", "uid", "ok")
    list_filter = ("symbol__exchange", "frequency", "ok")
    search_fields = ("symbol__code_name", "symbol__api_symbol", "uid")
    date_hierarchy = "timestamp"
    autocomplete_fields = ("symbol",)


class CandleDataAdmin(ReadOnlyAdmin):
    list_display = (
        "candle",
        "timestamp",
        "volume",
        "buy_volume",
        "notional",
        "buy_notional",
        "ticks",
        "incomplete",
    )
    list_filter = ("candle__symbol__exchange", "candle__code_name", "incomplete")
    search_fields = ("candle__code_name", "candle__symbol__api_symbol")
    date_hierarchy = "timestamp"
    autocomplete_fields = ("candle",)


class CandleCacheAdmin(ReadOnlyAdmin):
    list_display = ("candle", "timestamp", "frequency")
    list_filter = ("candle__symbol__exchange", "candle__code_name", "frequency")
    search_fields = ("candle__code_name", "candle__symbol__api_symbol")
    date_hierarchy = "timestamp"
    autocomplete_fields = ("candle",)


class ExchangeCandleDataAdmin(ReadOnlyAdmin):
    list_display = (
        "symbol",
        "timestamp",
        "frequency",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "notional",
    )
    list_filter = ("symbol__exchange", "frequency")
    search_fields = ("symbol__code_name", "symbol__api_symbol")
    date_hierarchy = "timestamp"
    autocomplete_fields = ("symbol",)


class FundingDataAdmin(ReadOnlyAdmin):
    list_display = ("symbol", "timestamp", "funding_rate")
    list_filter = ("symbol__exchange",)
    search_fields = ("symbol__code_name", "symbol__api_symbol")
    date_hierarchy = "timestamp"
    autocomplete_fields = ("symbol",)


if "django.contrib.admin" in settings.INSTALLED_APPS:
    admin.site.register(Symbol, SymbolAdmin)
    admin.site.register(Candle, CandleAdmin)
    admin.site.register(TaskState, TaskStateAdmin)
    admin.site.register(TradeData, TradeDataAdmin)
    admin.site.register(CandleData, CandleDataAdmin)
    admin.site.register(CandleCache, CandleCacheAdmin)
    admin.site.register(ExchangeCandleData, ExchangeCandleDataAdmin)
    admin.site.register(FundingData, FundingDataAdmin)
