from rest_framework import serializers

from quant_candles.constants import Exchange

from .base import BaseParameterSerializer


class QuantCandleParameterSerializer(BaseParameterSerializer):
    exchange = serializers.MultipleChoiceField(choices=Exchange.choices)
    retry = serializers.BooleanField(required=False, default=False)
