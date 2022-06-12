from rest_framework import serializers

from .base import BaseParameterSerializer


class ConvertToHourlyParameterSerializer(BaseParameterSerializer):
    time_ago = serializers.CharField(required=False, default="1d")
