import decimal
from datetime import date
from decimal import Decimal
from io import BytesIO
from json import JSONDecoder
from typing import Any

import numpy as np
import pandas as pd
import randomname
from django.core.files.base import ContentFile
from django.core.serializers.json import DjangoJSONEncoder
from django.db import models
from pandas import DataFrame

from quant_candles.constants import NUMERIC_PRECISION, NUMERIC_SCALE
from quant_candles.lib import to_pydatetime
from quant_candles.utils import gettext_lazy as _


class QuantCandlesEncoder(DjangoJSONEncoder):
    def default(self, obj: Any):
        """Default."""
        if isinstance(obj, np.int64):
            return int(obj)
        return super().default(obj)


def quant_candles_json_decoder(data: dict) -> dict:
    """Quant candles JSON decoder."""
    for key, value in data.items():
        if isinstance(data[key], str):
            try:
                data[key] = Decimal(value)
            except decimal.InvalidOperation:
                pass
        if isinstance(data[key], str):
            try:
                data[key] = date.fromisoformat(value)
            except (TypeError, ValueError):
                pass
        if isinstance(data[key], str):
            try:
                data[key] = to_pydatetime(pd.to_datetime(value))
            except (TypeError, ValueError):
                pass
    return data


class QuantCandlesDecoder(JSONDecoder):
    def __init__(self, *args, **kwargs):
        if "object_hook" not in kwargs:
            kwargs["object_hook"] = quant_candles_json_decoder
        super().__init__(*args, **kwargs)


def JSONField(name: str, **kwargs) -> models.JSONField:
    if "encoder" not in kwargs:
        kwargs["encoder"] = QuantCandlesEncoder
    if "decoder" not in kwargs:
        kwargs["decoder"] = QuantCandlesDecoder
    return models.JSONField(name, **kwargs)


def BigDecimalField(name: str, **kwargs) -> models.DecimalField:
    """Big decimal."""
    if "max_digits" not in kwargs:
        kwargs["max_digits"] = NUMERIC_PRECISION
    if "decimal_places" not in kwargs:
        kwargs["decimal_places"] = NUMERIC_SCALE
    return models.DecimalField(name, **kwargs)


class AbstractCodeName(models.Model):
    code_name = models.SlugField(_("code name"), unique=True, max_length=255)

    def __str__(self):
        return self.code_name

    @classmethod
    def get_random_name(cls) -> str:
        """Get random name."""
        name = randomname.get_name()
        if cls.objects.filter(code_name=name).exists():
            return cls.get_random_name()
        else:
            return name

    def save(self, *args, **kwargs) -> models.Model:
        """Save."""
        if not self.pk and not self.code_name:
            self.code_name = self.get_random_name()
        return super().save(*args, **kwargs)

    class Meta:
        abstract = True


class AbstractDataStorage(models.Model):
    @classmethod
    def prepare_data(cls, data_frame: DataFrame) -> ContentFile:
        """Prepare data, exclude uid."""
        if "uid" in data_frame.columns:
            data_frame = data_frame.drop(columns=["uid"])
        data_frame.reset_index(drop=True)
        buffer = BytesIO()
        data_frame.to_parquet(buffer, engine="auto", compression="snappy")
        return ContentFile(buffer.getvalue(), "data.parquet")

    def get_data_frame(self, field: str = "file_data") -> DataFrame:
        """Get data frame."""
        data = getattr(self, field)
        if data.name:
            return pd.read_parquet(data.open())

    class Meta:
        abstract = True
