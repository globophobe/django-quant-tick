import decimal
from datetime import date, datetime
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

from quant_tick.constants import NUMERIC_PRECISION, NUMERIC_SCALE
from quant_tick.lib import to_pydatetime
from quant_tick.utils import gettext_lazy as _


class QuantTickEncoder(DjangoJSONEncoder):
    """Quant tick encoder."""

    def default(self, obj: Any) -> Any:
        """Default."""
        if isinstance(obj, np.int64):
            return int(obj)
        # Default DjangoJSONEncoder strips microseconds.
        # django.core.serializers.json.py#L87
        if isinstance(obj, datetime):
            date_string = obj.isoformat()
            if date_string.endswith("+00:00"):
                date_string = date_string.removesuffix("+00:00") + "Z"
            return date_string
        return super().default(obj)


def quant_tick_json_decoder(data: dict) -> dict:
    """Quant tick JSON decoder."""
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


class QuantTickDecoder(JSONDecoder):
    """Quant tick decoder."""

    def __init__(self, *args, **kwargs) -> None:
        """Initialize."""
        if "object_hook" not in kwargs:
            kwargs["object_hook"] = quant_tick_json_decoder
        super().__init__(*args, **kwargs)


def JSONField(name: str, **kwargs) -> models.JSONField:  # noqa: N802
    """JSON."""
    if "encoder" not in kwargs:
        kwargs["encoder"] = QuantTickEncoder
    if "decoder" not in kwargs:
        kwargs["decoder"] = QuantTickDecoder
    return models.JSONField(name, **kwargs)


def BigDecimalField(name: str, **kwargs) -> models.DecimalField:  # noqa: N802
    """Big decimal."""
    if "max_digits" not in kwargs:
        kwargs["max_digits"] = NUMERIC_PRECISION
    if "decimal_places" not in kwargs:
        kwargs["decimal_places"] = NUMERIC_SCALE
    return models.DecimalField(name, **kwargs)


class AbstractCodeName(models.Model):
    """Abstract code name."""

    code_name = models.SlugField(_("code name"), unique=True, max_length=255)

    def __str__(self) -> str:
        """String."""
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
    """Abstract data storage."""

    @classmethod
    def prepare_data(cls, data_frame: DataFrame) -> ContentFile:
        """Prepare data, exclude uid."""
        drop_columns = []
        if "index" in data_frame.columns:
            drop_columns.append("index")
        if "uid" in data_frame.columns:
            drop_columns.append("uid")
        if len(drop_columns):
            data_frame = data_frame.drop(columns=drop_columns)
        data_frame.reset_index(drop=True)
        buffer = BytesIO()
        data_frame.to_parquet(buffer, engine="auto", compression="snappy")
        return ContentFile(buffer.getvalue(), "data.parquet")

    def has_data_frame(self, field: str) -> bool:
        """Has data frame."""
        data = getattr(self, field)
        return data.name != ""

    def get_data_frame(self, field: str) -> DataFrame:
        """Get data frame."""
        if self.has_data_frame(field):
            return pd.read_parquet(getattr(self, field).open())

    class Meta:
        abstract = True
