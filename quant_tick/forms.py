import json

import pandas as pd
from django import forms
from django.forms import formset_factory
from django.http import HttpRequest

from quant_tick.constants import Exchange
from quant_tick.lib import get_min_time, parse_datetime

FORMSET_KEYS = ("exchange", "api_symbol", "timestamp_from", "time_ago")


def parse_time_delta(value: str) -> pd.Timedelta:
    try:
        delta = pd.Timedelta(value)
    except ValueError as exc:
        raise ValueError(f"Cannot parse {value}.") from exc
    if pd.isna(delta):
        raise ValueError(f"Cannot parse {value}.")
    return delta


def parse_timestamp_from(value: str):
    return get_min_time(parse_datetime(value), "1h")


def parse_request_body(request: HttpRequest) -> dict:
    if request.method != "POST":
        return request.GET.dict()
    if not request.body:
        return {}
    try:
        data = json.loads(request.body)
    except json.JSONDecodeError as exc:
        raise ValueError("Invalid JSON body.") from exc
    if not isinstance(data, dict):
        raise ValueError("JSON body must be an object.")
    return data


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


class AggregateCandleRequestForm(forms.Form):
    exchange = forms.ChoiceField(
        choices=[("", "")] + list(Exchange.choices),
        required=False,
    )
    api_symbol = forms.CharField(required=False)
    timestamp_from = forms.CharField(required=False)
    time_ago = forms.CharField(required=False)

    def clean_timestamp_from(self):
        value = self.cleaned_data["timestamp_from"]
        if not value:
            return None
        try:
            return parse_timestamp_from(value)
        except (TypeError, ValueError) as exc:
            raise forms.ValidationError("Invalid timestamp_from.") from exc

    def clean_time_ago(self):
        value = self.cleaned_data["time_ago"]
        if not value:
            return None
        try:
            return parse_time_delta(value)
        except ValueError as exc:
            raise forms.ValidationError("Invalid time_ago.") from exc


AggregateCandleRequestFormSet = formset_factory(AggregateCandleRequestForm, extra=0)


def get_formset_data(payloads: list[dict]) -> dict:
    data = {
        "form-TOTAL_FORMS": str(len(payloads)),
        "form-INITIAL_FORMS": "0",
    }
    for index, payload in enumerate(payloads):
        for key in FORMSET_KEYS:
            value = payload.get(key)
            if value is not None:
                data[f"form-{index}-{key}"] = value
    return data


def format_formset_errors(formset) -> str:
    errors = []
    for index, form_errors in enumerate(formset.errors):
        if form_errors:
            errors.append(f"request {index}: {form_errors.as_text()}")
    if formset.non_form_errors():
        errors.append(formset.non_form_errors().as_text())
    return "; ".join(errors) or "Invalid candle request."


def get_candle_request_data(payload: dict) -> list[dict]:
    candle_requests = payload.get("candle_requests")
    if candle_requests is None:
        candle_requests = [payload]
    elif not isinstance(candle_requests, list):
        raise ValueError("candle_requests must be a list.")
    for item in candle_requests:
        if not isinstance(item, dict):
            raise ValueError("candle_requests items must be objects.")
    formset = AggregateCandleRequestFormSet(data=get_formset_data(candle_requests))
    if not formset.is_valid():
        raise ValueError(format_formset_errors(formset))
    return [form.cleaned_data for form in formset.forms]
