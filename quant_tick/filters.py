from importlib.util import find_spec

from django.conf import settings
from django.utils.translation import gettext_lazy as _

from quant_tick.constants import Exchange, Frequency, SymbolType, TaskType
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

HAS_SEMANTIC_FILTERS = (
    "semantic_forms" in settings.INSTALLED_APPS
    and find_spec("django_filters") is not None
)

if HAS_SEMANTIC_FILTERS:
    from django_filters.constants import EMPTY_VALUES
    from django_filters.filters import AllValuesFilter
    from semantic_forms.fields import SemanticChoiceField
    from semantic_forms.filters import (
        SemanticChoiceFilter,
        SemanticFilterSet,
        SemanticModelChoiceFilter,
    )
else:
    SymbolFilter = None
    CandleFilter = None
    TaskStateFilter = None
    TradeDataFilter = None
    CandleDataFilter = None
    CandleCacheFilter = None
    ExchangeCandleDataFilter = None
    FundingDataFilter = None

if HAS_SEMANTIC_FILTERS:
    BLANK_CHOICE = (("", ""),)
    BOOLEAN_CHOICES = (("true", _("Yes")), ("false", _("No")))

    def blank_choices(choices):
        return BLANK_CHOICE + tuple(choices)

    class SemanticBlankChoiceFilter(SemanticChoiceFilter):
        def __init__(self, *args, **kwargs):
            kwargs["choices"] = blank_choices(kwargs["choices"])
            super().__init__(*args, **kwargs)

    class SemanticBooleanFilter(SemanticChoiceFilter):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("choices", blank_choices(BOOLEAN_CHOICES))
            super().__init__(*args, **kwargs)

        def filter(self, qs, value):
            if value in EMPTY_VALUES:
                return qs
            return super().filter(qs, value == "true")

    class SemanticAllValuesFilter(AllValuesFilter):
        field_class = SemanticChoiceField

        @property
        def field(self):
            field = super().field
            choices = tuple(field.choices)
            if choices[:1] != BLANK_CHOICE:
                field.choices = blank_choices(choices)
            return field

    class SymbolFilter(SemanticFilterSet):
        exchange = SemanticBlankChoiceFilter(choices=Exchange.choices)
        symbol_type = SemanticBlankChoiceFilter(choices=SymbolType.choices)
        is_active = SemanticBooleanFilter(label=_("active"))
        save_raw = SemanticBooleanFilter(label=_("save raw"))
        save_aggregated = SemanticBooleanFilter(label=_("save aggregated"))

        class Meta:
            model = Symbol
            fields = (
                "exchange",
                "symbol_type",
                "is_active",
                "save_raw",
                "save_aggregated",
            )


    class CandleFilter(SemanticFilterSet):
        symbol = SemanticModelChoiceFilter(
            empty_label="",
            queryset=Symbol.objects.filter(
                is_active=True,
                pk__in=Candle.objects.values("symbol_id"),
            )
        )
        is_active = SemanticBooleanFilter(label=_("active"))

        class Meta:
            model = Candle
            fields = ("symbol", "is_active")


    class TaskStateFilter(SemanticFilterSet):
        exchange = SemanticBlankChoiceFilter(choices=Exchange.choices)
        task_type = SemanticBlankChoiceFilter(choices=TaskType.choices)

        class Meta:
            model = TaskState
            fields = ("exchange", "api_symbol", "task_type")


    class TradeDataFilter(SemanticFilterSet):
        symbol = SemanticModelChoiceFilter(
            empty_label="",
            queryset=Symbol.objects.filter(
                is_active=True,
                pk__in=TradeData.objects.values("symbol_id")
            )
        )
        frequency = SemanticBlankChoiceFilter(choices=Frequency.choices)
        ok = SemanticBooleanFilter(label=_("ok"))

        class Meta:
            model = TradeData
            fields = ("symbol", "frequency", "ok")


    class CandleDataFilter(SemanticFilterSet):
        symbol = SemanticModelChoiceFilter(
            field_name="candle__symbol",
            label=_("symbol"),
            empty_label="",
            queryset=Symbol.objects.filter(
                is_active=True,
                pk__in=CandleData.objects.values("candle__symbol_id")
            ),
        )

        class Meta:
            model = CandleData
            fields = ("symbol",)


    class CandleCacheFilter(SemanticFilterSet):
        symbol = SemanticModelChoiceFilter(
            field_name="candle__symbol",
            label=_("symbol"),
            empty_label="",
            queryset=Symbol.objects.filter(
                is_active=True,
                pk__in=CandleCache.objects.values("candle__symbol_id")
            ),
        )
        candle = SemanticModelChoiceFilter(
            empty_label="",
            queryset=Candle.objects.filter(pk__in=CandleCache.objects.values("candle_id"))
        )
        frequency = SemanticAllValuesFilter(label=_("frequency"))

        class Meta:
            model = CandleCache
            fields = ("symbol", "candle", "frequency")


    class ExchangeCandleDataFilter(SemanticFilterSet):
        symbol = SemanticModelChoiceFilter(
            empty_label="",
            queryset=Symbol.objects.filter(
                is_active=True,
                pk__in=ExchangeCandleData.objects.values("symbol_id")
            )
        )
        frequency = SemanticAllValuesFilter(label=_("frequency"))

        class Meta:
            model = ExchangeCandleData
            fields = ("symbol", "frequency")


    class FundingDataFilter(SemanticFilterSet):
        symbol = SemanticModelChoiceFilter(
            empty_label="",
            queryset=Symbol.objects.filter(
                is_active=True,
                pk__in=FundingData.objects.values("symbol_id")
            )
        )

        class Meta:
            model = FundingData
            fields = ("symbol",)
