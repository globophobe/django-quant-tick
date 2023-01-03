import os
import sys

from decouple import config

from .base import *  # noqa

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = False
IS_LOCAL = True

# Database
# https://docs.djangoproject.com/en/4.0/ref/settings/#databases

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql_psycopg2",
        "NAME": config("DATABASE_NAME"),
        "USER": config("DATABASE_USER"),
        "PASSWORD": config("DATABASE_PASSWORD"),
        "HOST": config("DATABASE_HOST"),
        "PORT": config("PROXY_DATABASE_PORT"),
        "TEST": {"NAME": f'test_{config("DATABASE_NAME")}'},
    },
    "read_only": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR.parent / "db.sqlite3",  # noqa
        "TEST": {"NAME": BASE_DIR.parent / "test_db.sqlite3"},  # noqa
    },
}

DATABASE_ROUTERS = [
    "demo.db_routers.DefaultRouter",
    "demo.db_routers.ReadOnlyRouter",
]

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
        },
    },
    "root": {
        "handlers": ["console"],
        "level": "INFO",
    },
}

# GCP
CREDENTIALS = (
    BASE_DIR.parents[0] / "keys" / config("GOOGLE_APPLICATION_CREDENTIALS")  # noqa
)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(CREDENTIALS.resolve())

DEFAULT_FILE_STORAGE = "storages.backends.gcloud.GoogleCloudStorage"
GS_BUCKET_NAME = (
    f'test-{config("GCS_BUCKET_NAME")}'
    if "test" in sys.argv
    else config("GCS_BUCKET_NAME")
)
