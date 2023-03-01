import sys

from decouple import config

from ..base import *  # noqa

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = False
IS_LOCAL = False

DATA_UPLOAD_MAX_MEMORY_SIZE = 1024 * 1024 * 10  # 10MB

# Database
# https://docs.djangoproject.com/en/4.0/ref/settings/#databases

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.postgresql_psycopg2",
        "NAME": config("DATABASE_NAME"),
        "USER": config("DATABASE_USER"),
        "PASSWORD": config("DATABASE_PASSWORD"),
        "HOST": f"/cloudsql/{config('PRODUCTION_DATABASE_HOST')}",
        "PORT": config("DATABASE_PORT"),
        "TEST": {"NAME": f'test_{config("DATABASE_NAME")}'},
    },
    "read_only": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",  # noqa
    },
}

DATABASE_ROUTERS = [
    "demo.db_routers.DefaultRouter",
    "demo.db_routers.ReadOnlyRouter",
]

# GCP
DEFAULT_FILE_STORAGE = "storages.backends.gcloud.GoogleCloudStorage"
GS_BUCKET_NAME = (
    f'test-{config("GCS_BUCKET_NAME")}'
    if "test" in sys.argv
    else config("GCS_BUCKET_NAME")
)
