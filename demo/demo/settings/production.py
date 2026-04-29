import os
import sys
from urllib.parse import urlparse

import sentry_sdk
from sentry_sdk.integrations.django import DjangoIntegration

from .base import *  # noqa

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = False

DATA_UPLOAD_MAX_MEMORY_SIZE = 1024 * 1024 * 10  # 10MB

ROOT_URLCONF = "demo.urls"


def get_allowed_host(value: str) -> str:
    parsed = urlparse(value.strip())
    if parsed.hostname:
        return parsed.hostname
    raise ValueError(f"invalid PRODUCTION_API_URL: {value}")


ALLOWED_HOSTS = [get_allowed_host(os.environ["PRODUCTION_API_URL"])]

USE_X_FORWARDED_HOST = True
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql_psycopg2",
        "NAME": os.environ["DATABASE_NAME"],
        "USER": os.environ["DATABASE_USER"],
        "PASSWORD": os.environ["DATABASE_PASSWORD"],
        "HOST": f"/cloudsql/{os.environ['PRODUCTION_DATABASE_HOST']}",
        "PORT": os.environ["DATABASE_PORT"],
        "TEST": {"NAME": f'test_{os.environ["DATABASE_NAME"]}'},
    },
}

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

sentry_sdk.init(
    dsn=os.environ["SENTRY_DSN"],
    integrations=[DjangoIntegration()],
    traces_sample_rate=0.5,
)
