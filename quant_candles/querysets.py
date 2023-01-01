from datetime import datetime
from typing import Optional

from django.db import models
from django.db.models import Q, QuerySet


class TimeFrameQuerySet(models.QuerySet):
    def filter_by_timestamp(
        self,
        timestamp_from: Optional[datetime] = None,
        timestamp_to: Optional[datetime] = None,
    ) -> QuerySet:
        """Filter by timestamp."""
        q = Q()
        if timestamp_from:
            q |= Q(timestamp__gte=timestamp_from)
        if timestamp_to:
            q |= Q(timestamp__lt=timestamp_to)
        return self.filter(q)
