from decouple import config

from .base import *  # noqa

ALLOWED_HOSTS = [config("PRODUCTION_API_URL")]

ROOT_URLCONF = "demo.urls.api"

INSTALLED_APPS = [
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    # PostgreSQL
    "django.contrib.postgres",
    # 3rd party
    "polymorphic",
    "rest_framework",
    # Main
    "quant_candles",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]
