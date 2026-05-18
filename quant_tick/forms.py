import pandas as pd
from django import forms

from quant_tick.constants import Exchange


def parse_time_delta(value: str) -> pd.Timedelta:
    try:
        delta = pd.Timedelta(value)
    except ValueError as exc:
        raise ValueError(f"Cannot parse {value}.") from exc
    if pd.isna(delta):
        raise ValueError(f"Cannot parse {value}.")
    return delta


def format_form_errors(form: forms.Form) -> str:
    errors = []
    for field_errors in form.errors.as_data().values():
        for error in field_errors:
            errors.append(str(error.message))
    return "; ".join(errors) or "Invalid request."


class TimeRangeRequestForm(forms.Form):
    time_ago = forms.CharField(required=False)

    def clean_time_ago(self):
        value = self.cleaned_data["time_ago"]
        if not value and "time_ago" not in self.data:
            value = "1d"
        try:
            return parse_time_delta(value)
        except ValueError as exc:
            raise forms.ValidationError(str(exc)) from exc


class SymbolTimeRangeRequestForm(TimeRangeRequestForm):
    api_symbol = forms.CharField(required=False)


class AggregateTradeRequestForm(SymbolTimeRangeRequestForm):
    exchange = forms.ChoiceField(
        choices=Exchange.choices,
        error_messages={"invalid_choice": "Invalid exchange."},
    )


class FetchExchangeDataRequestForm(SymbolTimeRangeRequestForm):
    exchange = forms.ChoiceField(
        choices=[("", "")] + list(Exchange.choices),
        required=False,
        error_messages={"invalid_choice": "Invalid exchange."},
    )
