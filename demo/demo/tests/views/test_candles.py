from django.db import models
from django.urls import reverse
from rest_framework import status
from rest_framework.test import APITestCase

from quant_tick.lib import get_current_time, get_min_time, get_previous_time
from quant_tick.models import Candle, CandleData

from .base import BaseViewTest


class CandleDataViewTest(BaseViewTest, APITestCase):
    """Candle data view test."""

    def setUp(self):
        """Set up."""
        super().setUp()
        now = get_current_time()
        self.timestamp = get_min_time(now, "1min")

    def get_url(self, candle: Candle) -> str:
        """Get URL."""
        return reverse("candle_data", kwargs={"code_name": candle.code_name})

    def get_isoformat(self, obj: models.Model) -> str:
        """Get isoformat."""
        value = obj.timestamp.isoformat()
        if value.endswith("+00:00"):
            value = value[:-6] + "Z"
        return value

    def test_get(self):
        """Get."""
        candle = Candle.objects.create()
        candle_data = CandleData.objects.create(
            candle=candle, timestamp=get_previous_time(self.timestamp, "1min")
        )
        url = self.get_url(candle)
        response = self.client.get(url)
        self.assertEqual(response.status_code, status.HTTP_200_OK)
        data = response.json()
        self.assertEqual(data[0]["timestamp"], self.get_isoformat(candle_data))
