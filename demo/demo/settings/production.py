import sys

import sentry_sdk
from decouple import config
from sentry_sdk.integrations.django import DjangoIntegration

from .base import *  # noqa

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = False

ALLOWED_HOSTS = [config("DOMAIN_NAME")]

USE_X_FORWARDED_HOST = True
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")

# Database
# https://docs.djangoproject.com/en/4.0/ref/settings/#databases

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql_psycopg2",
        "NAME": config("DATABASE_NAME"),
        "USER": config("DATABASE_USER"),
        "PASSWORD": config("DATABASE_PASSWORD"),
        "HOST": config("PRODUCTION_DATABASE_HOST"),
        "PORT": config("DATABASE_PORT"),
        "TEST": {"NAME": f'test_{config("DATABASE_NAME")}'},
    }
}

sentry_sdk.init(
    dsn=config("SENTRY_DSN"),
    integrations=[DjangoIntegration()],
    # Less transactions
    traces_sample_rate=0.01,
)

STATICFILES_STORAGE = "whitenoise.storage.CompressedStaticFilesStorage"

# GCP
DEFAULT_FILE_STORAGE = "storages.backends.gcloud.GoogleCloudStorage"
GS_BUCKET_NAME = (
    f'test-{config("GCS_BUCKET_NAME")}'
    if "test" in sys.argv
    else config("GCS_BUCKET_NAME")
)
