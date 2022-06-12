from rest_framework import serializers

from .base import BaseParameterSerializer


class TradeParameterSerializer(BaseParameterSerializer):
    time_ago = serializers.CharField(required=False, default="5t")
    retry = serializers.BooleanField(required=False, default=False)
