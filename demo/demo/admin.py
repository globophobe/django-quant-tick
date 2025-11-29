from django.contrib import admin
from django.contrib.auth.models import Group, User
from semantic_admin import SemanticModelAdmin, SemanticTabularInline

from quant_tick.models import GlobalSymbol, MLConfig, Symbol

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


@admin.register(MLConfig)
class MLConfigAdmin(SemanticModelAdmin):
    """MLConfig admin."""

    list_display = ["code_name", "candle", "symbol", "horizon_bars", "is_active"]
    list_filter = ["is_active", "candle", "symbol"]
    search_fields = ["code_name"]

    fieldsets = [
        (None, {
            "fields": ["candle", "symbol"]
        }),
        ("Training Parameters", {
            "fields": ["horizon_bars", "inference_lookback"]
        }),
        ("Strategy Parameters", {
            "fields": ["touch_tolerance", "min_hold_bars"]
        }),
        ("Configuration", {
            "fields": ["json_data", "is_active"]
        }),
        ("Metadata", {
            "fields": ["code_name", "last_processed_timestamp"],
            "classes": ["collapse"]
        }),
    ]

    readonly_fields = ["code_name", "last_processed_timestamp"]
