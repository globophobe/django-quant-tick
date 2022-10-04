from io import BytesIO
from typing import Dict, List, Type

import pandas as pd
from django.contrib.contenttypes.models import ContentType
from django.core.files.base import ContentFile
from django.db import models
from pandas import DataFrame

from quant_werks.constants import NUMERIC_PRECISION, NUMERIC_SCALE
from quant_werks.utils import gettext_lazy as _


def limit_choices_to_subclass(cls: Type[models.Model]) -> Dict[str, List[int]]:
    """Limit choices to model subclass."""
    ctypes = ContentType.objects.all()
    ctype_models = [(ctype, ctype.model_class()) for ctype in ctypes]
    pks = [
        ctype.pk
        for ctype, model in ctype_models
        if model is not None and issubclass(model, cls) and model is not cls
    ]
    return {"pk__in": pks}


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
