from typing import Optional, Type

from django.contrib.contenttypes.models import ContentType
from django.db.models import Model

from quant_candles.models import CandleReadOnlyData


class BaseRouter:
    def get_name(self, model: Optional[Type[Model]] = None) -> str:
        """Get name."""
        return "read_only" if model == CandleReadOnlyData else "default"

    def db_for_read(self, model: Type[Model], **hints) -> str:
        """DB for read."""
        return self.get_name(model)

    def db_for_write(self, model: Type[Model], **hints) -> str:
        """DB for write."""
        return self.get_name(model)

    def allow_relation(self, obj1: Model, obj2: Model, **hints) -> bool:
        """Allow relation.

        If no router has an opinion (i.e. all routers return None), only relations
        within the same database are allowed.
        Ref: https://docs.djangoproject.com/en/4.1/topics/db/multi-db/#allow_relation
        """
        return None


class DefaultRouter(BaseRouter):
    def allow_migrate(
        self, db: str, app_label: str, model_name: Optional[str] = None, **hints
    ):
        """Allow migrate."""
        is_read_only_db = db == "read_only"
        is_read_only_model = model_name == CandleReadOnlyData._meta.model_name
        if is_read_only_db:
            return is_read_only_model
        else:
            return not is_read_only_model

    def allow_relation(self, obj1: Model, obj2: Model, **hints) -> bool:
        """Allow relation."""
        is_read_only = isinstance(obj1, CandleReadOnlyData) or isinstance(
            obj2, CandleReadOnlyData
        )
        return not is_read_only


class ReadOnlyRouter(BaseRouter):
    def allow_migrate(
        self, db: str, app_label: str, model_name: Optional[str] = None, **hints
    ):
        """Allow migrate."""
        is_read_only_db = db == "read_only"
        is_read_only_model = model_name == CandleReadOnlyData._meta.model_name
        return is_read_only_db and is_read_only_model

    def allow_relation(self, obj1: Model, obj2: Model, **hints) -> bool:
        """Allow relation."""
        is_content_type = isinstance(obj1, ContentType) or isinstance(obj2, ContentType)
        is_read_only_model = isinstance(obj1, CandleReadOnlyData) or isinstance(
            obj2, CandleReadOnlyData
        )
        return is_content_type and is_read_only_model
