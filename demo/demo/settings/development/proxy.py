import os

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

# GCP
CREDENTIALS = (
    BASE_DIR.parent.parent.parent / "keys" / os.environ["GOOGLE_APPLICATION_CREDENTIALS"]  # noqa: F405
)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(CREDENTIALS.resolve())

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
    if TEST
    else os.environ["GCS_BUCKET_NAME"]
)
