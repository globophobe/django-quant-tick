from typing import List

from django.contrib import admin
from django.contrib.auth.models import Group, User
from rest_framework.authtoken.models import TokenProxy
from semantic_admin import (
    SemanticModelAdmin,
    SemanticStackedInline,
    SemanticTabularInline,
)

from .models import Candle, CandleSymbol, GlobalSymbol, Symbol

admin.site.unregister(User)
admin.site.unregister(Group)
admin.site.unregister(TokenProxy)


class SymbolInline(SemanticTabularInline):
    model = Symbol
    extra = 0


@admin.register(GlobalSymbol)
class GlobalSymbolAdmin(SemanticModelAdmin):
    list_display = ("__str__",)
    fields = ("name",)
    inlines = (SymbolInline,)


@admin.register(Symbol)
class SymbolAdmin(SemanticModelAdmin):
    list_display = ("__str__",)
    fields = (
        ("global_symbol", "exchange"),
        ("api_symbol", "min_volume"),
    )


class CandleSymbolInline(SemanticStackedInline):
    model = CandleSymbol
    fields = (("symbol", "date_from", "date_to"),)
    extra = 0

    def get_readonly_fields(self, request, obj=None) -> List[str]:
        """Get readonly fields."""
        if obj and obj.pk:
            return ["symbol", "date_from", "date_to"]
        return []


@admin.register(Candle)
class CandleAdmin(SemanticModelAdmin):
    list_display = (
        "code_name",
        "candle_type",
        "threshold",
        "expected_number_of_candles",
        "moving_average_number_of_days",
        "is_ema",
        "cache_reset_frequency",
    )
    fields = (
        ("code_name", "candle_type"),
        (
            "threshold",
            "expected_number_of_candles",
            "moving_average_number_of_days",
            "is_ema",
        ),
        "cache_reset_frequency",
    )
    inlines = (CandleSymbolInline,)

    def get_readonly_fields(self, request, obj=None) -> List[str]:
        """Get readonly fields."""
        if obj and obj.pk:
            return self.list_display
        return []
