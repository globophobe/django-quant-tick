from rest_framework import serializers

from .base import BaseParameterSerializer


class TradeParameterSerializer(BaseParameterSerializer):
    retry = serializers.BooleanField(required=False, default=False)
