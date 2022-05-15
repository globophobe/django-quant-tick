from typing import Callable, List

from django.db import models

from cryptofeed_werks.utils import gettext_lazy as _

# Similar to BigQuery BigNumeric
# https://cloud.google.com/bigquery/docs/reference/standard-sql/data-types#decimal_types
NUMERIC_PRECISION = 76  # 76.6
NUMERIC_SCALE = 38


def big_decimal(
    name: str, null: bool = False, validators: List[Callable] = []
) -> models.DecimalField:
    return models.DecimalField(
        _(name),
        null=null,
        max_digits=NUMERIC_PRECISION,
        decimal_places=NUMERIC_SCALE,
        validators=validators,
    )
