# ruff: noqa: F403, F405
from ..base import *

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = True

# Database
# https://docs.djangoproject.com/en/4.0/ref/settings/#databases

ROOT_URLCONF = "demo.urls"

WSGI_APPLICATION = "demo.wsgi.development.application"

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR.parent.parent / "db.sqlite3",
        "TEST": {"NAME": "test_db.sqlite3"},
    }
}

if not TEST:
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
