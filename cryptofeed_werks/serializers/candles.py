from rest_framework import serializers

from .base import big_decimal


class CandleSerializer(serializers.Serializer):
    timestamp = serializers.DateTimeField()
    open = big_decimal()
    high = big_decimal()
    low = big_decimal()
    close = big_decimal()
    totalBuyVolume = big_decimal()
    totalVolume = big_decimal()
    totalBuyNotional = big_decimal()
    totalBuyNotional = big_decimal()
    totalBuyTicks = serializers.IntegerField()
    totalTicks = serializers.IntegerField()
