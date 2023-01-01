import logging

from django.contrib.auth import get_user_model
from django.db import models
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase

from quant_candles.lib import get_current_time, get_min_time, get_previous_time
from quant_candles.models import Candle, CandleData, CandleReadOnlyData


class CandleViewTest(APITestCase):
    databases = {"default", "read_only"}

    def setUp(self):
        User = get_user_model()
        user = User.objects.create(username="test")
        self.client.force_authenticate(user)

        now = get_current_time()
        self.timestamp = get_min_time(now, "1t")

    def get_url(self, candle: Candle) -> str:
        """Get URL."""
        return reverse("candles", kwargs={"code_name": candle.code_name})

    def get_isoformat(self, obj: models.Model) -> str:
        """Get isoformat."""
        value = obj.timestamp.isoformat()
        if value.endswith("+00:00"):
            value = value[:-6] + "Z"
        return value

    def test_get(self):
        """QuerySet results are combined."""
        candle = Candle.objects.create()
        candle_data = CandleData.objects.create(candle=candle, timestamp=self.timestamp)
        candle_read_only_data = CandleReadOnlyData.objects.create(
            candle_id=candle.id,
            timestamp=get_previous_time(self.timestamp, "1t"),
        )
        url = self.get_url(candle)
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertEqual(data[0]["timestamp"], self.get_isoformat(candle_data))
        self.assertEqual(
            data[1]["timestamp"], self.get_isoformat(candle_read_only_data)
        )

    def test_ordering(self):
        """QuerySet results are correctly ordered."""
        candle = Candle.objects.create()
        candle_data = CandleData.objects.create(candle=candle, timestamp=self.timestamp)
        candle_read_only_data1 = CandleReadOnlyData.objects.create(
            candle_id=candle.id,
            timestamp=get_previous_time(self.timestamp, "1t"),
        )
        candle_read_only_data2 = CandleReadOnlyData.objects.create(
            candle_id=candle.id,
            timestamp=get_previous_time(self.timestamp, "2t"),
        )
        url = self.get_url(candle)
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertEqual(data[0]["timestamp"], self.get_isoformat(candle_data))
        self.assertEqual(
            data[1]["timestamp"], self.get_isoformat(candle_read_only_data1)
        )
        self.assertEqual(
            data[2]["timestamp"], self.get_isoformat(candle_read_only_data2)
        )
