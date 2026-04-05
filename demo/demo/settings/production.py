import os
import sys

import sentry_sdk
from sentry_sdk.integrations.django import DjangoIntegration

from .base import *  # noqa

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = False

DATA_UPLOAD_MAX_MEMORY_SIZE = 1024 * 1024 * 10  # 10MB

ROOT_URLCONF = "demo.urls"

ALLOWED_HOSTS = [os.environ["PRODUCTION_API_URL"]]

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
    "staticfiles": {
        "BACKEND": "django.contrib.staticfiles.storage.StaticFilesStorage",
    },
}

GS_BUCKET_NAME = (
    f'test-{os.environ["GCS_BUCKET_NAME"]}'
    if "test" in sys.argv
    else os.environ["GCS_BUCKET_NAME"]
)

INSTALLED_APPS = [
    "django.contrib.contenttypes",
    "django.contrib.postgres",
    "polymorphic",
    "quant_tick",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

sentry_sdk.init(
    dsn=os.environ["SENTRY_DSN"],
    integrations=[DjangoIntegration()],
    traces_sample_rate=0.5,
)
