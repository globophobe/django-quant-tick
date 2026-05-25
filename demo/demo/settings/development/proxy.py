import os
from pathlib import Path

from quant_tick.testing import is_test

# ruff: noqa: F403, F405
from .base import *  # noqa

# Database
# https://docs.djangoproject.com/en/4.0/ref/settings/#databases

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql_psycopg2",
        "NAME": os.environ["DATABASE_NAME"],
        "USER": os.environ["DATABASE_USER"],
        "PASSWORD": os.environ["DATABASE_PASSWORD"],
        "HOST": os.environ["DATABASE_HOST"],
        "PORT": os.environ.get("PROXY_DATABASE_PORT"),
        "TEST": {"NAME": f'test_{os.environ["DATABASE_NAME"]}'},
    },
}

# Google Cloud
credentials = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "")
if credentials and not Path(credentials).is_absolute():
    credentials = Path.home() / "keys" / credentials
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(credentials.resolve())

STORAGES = {
    "default": {
        "BACKEND": "storages.backends.gcloud.GoogleCloudStorage",
    },
    "staticfiles": {
        "BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage",
    },
}

GS_BUCKET_NAME = (
    f'test-{os.environ["GCS_BUCKET_NAME"]}'
    if is_test()
    else os.environ["GCS_BUCKET_NAME"]
)
