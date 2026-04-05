import os
import sys

from ..base import *  # noqa

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = False

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
    BASE_DIR.parent.parent.parent / "keys" / os.environ["GOOGLE_APPLICATION_CREDENTIALS"]  # noqa: F405
)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(CREDENTIALS.resolve())

STORAGES = {
    "default": {
        "BACKEND": "storages.backends.gcloud.GoogleCloudStorage",
    },
}

GS_BUCKET_NAME = (
    f'test-{os.environ["GCS_BUCKET_NAME"]}'
    if "test" in sys.argv
    else os.environ["GCS_BUCKET_NAME"]
)
