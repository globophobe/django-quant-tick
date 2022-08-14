from io import BytesIO

import pandas as pd
from django.core.files.base import ContentFile
from django.db import models
from pandas import DataFrame

from quant_werks.constants import NUMERIC_PRECISION, NUMERIC_SCALE
from quant_werks.utils import gettext_lazy as _


def big_decimal(name: str, **kwargs) -> models.DecimalField:
    """Big decimal."""
    if "max_digits" not in kwargs:
        kwargs["max_digits"] = NUMERIC_PRECISION
    if "decimal_places" not in kwargs:
        kwargs["decimal_places"] = NUMERIC_SCALE
    return models.DecimalField(_(name), **kwargs)


class BaseDataStorage(models.Model):
    @classmethod
    def prepare_data(cls, data_frame: DataFrame) -> ContentFile:
        """Prepare data, exclude uid."""
        buffer = BytesIO()
        data_frame.to_parquet(buffer, engine="auto", compression="snappy")
        return ContentFile(buffer.getvalue(), "data.parquet")

    def get_data_frame(self) -> DataFrame:
        """Load data frame."""
        if self.data.name:
            return pd.read_parquet(self.data.open())

    class Meta:
        abstract = True
