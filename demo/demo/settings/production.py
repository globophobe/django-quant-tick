import os
from urllib.parse import urlparse

import sentry_sdk
from sentry_sdk.integrations.django import DjangoIntegration

from quant_tick.testing import is_test

from .base import *  # noqa

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = False

DATA_UPLOAD_MAX_MEMORY_SIZE = 1024 * 1024 * 10  # 10MB

ROOT_URLCONF = "demo.urls.production"


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
    if is_test()
    else os.environ["GCS_BUCKET_NAME"]
)


def scrub_sentry_event(event, _hint):
    sensitive_fragments = (
        "authorization",
        "api_key",
        "apikey",
        "database_password",
        "password",
        "secret",
        "token",
    )

    def scrub(value):
        if isinstance(value, dict):
            return {
                key: (
                    "[Filtered]"
                    if any(fragment in str(key).lower() for fragment in sensitive_fragments)
                    else scrub(item)
                )
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [scrub(item) for item in value]
        return value

    return scrub(event)


sentry_sdk.init(
    dsn=os.environ["SENTRY_DSN"],
    integrations=[DjangoIntegration()],
    before_send=scrub_sentry_event,
    include_local_variables=False,
    send_default_pii=False,
    traces_sample_rate=1.0,
)
