import os
import re

from decouple import config
from invoke import task


@task
def django_settings(ctx, proxy=False):
    name = "proxy" if proxy else "development"
    os.environ["DJANGO_SETTINGS_MODULE"] = f"demo.settings.{name}"
    import django

    django.setup()
    from django.conf import settings

    return settings


@task
def coverage(ctx):
    ctx.run("coverage run --source=../ manage.py test quant_candles; coverage report")


@task
def start_proxy(ctx):
    host = config("PRODUCTION_DATABASE_HOST")
    port = config("PROXY_DATABASE_PORT")
    ctx.run(f'cloud-tools/cloud-sql-proxy -instances="{host}"=tcp:{port}')


@task
def create_user(ctx, username, password, proxy=False):
    django_settings(ctx, proxy=proxy)
    from django.contrib.auth import get_user_model

    User = get_user_model()
    user = User.objects.create(username=username, is_superuser=True, is_staff=True)
    user.set_password(password)
    user.save()


@task
def get_container_name(ctx, hostname="asia.gcr.io"):
    project_id = ctx.run("gcloud config get-value project").stdout.strip()
    return f"{hostname}/{project_id}/django-quant-candles"


def docker_secrets():
    build_args = [
        f'{secret}="{config(secret)}"'
        for secret in (
            "SECRET_KEY",
            "SENTRY_DSN",
            "DATABASE_NAME",
            "DATABASE_USER",
            "DATABASE_PASSWORD",
            "PRODUCTION_DATABASE_HOST",
            "DATABASE_PORT",
            "GCS_BUCKET_NAME",
        )
    ]
    return " ".join([f"--build-arg {build_arg}" for build_arg in build_args])


def build_quant_candles(ctx):
    result = ctx.run("poetry build").stdout
    return re.search(r"django_quant_candles-.*\.whl", result).group()


@task
def build_container(ctx, hostname="asia.gcr.io"):
    wheel = build_quant_candles(ctx)
    ctx.run("echo yes | python manage.py collectstatic")
    name = get_container_name(ctx, hostname=hostname)
    # Requirements
    requirements = [
        "django-filter",
        "django-polymorphic",
        "djangorestframework",
        "django-semantic-admin",
        "django-storages[google]",
        "gunicorn",
        "https",
        "pandas",
        "pyarrow",
        "psycopg2-binary",
        "python-decouple",
        "randomname",
        "sentry-sdk",
        "whitenoise",
    ]
    # Versions
    reqs = " ".join(
        [
            req.split(";")[0]
            for req in ctx.run("poetry export --dev --without-hashes").stdout.split(
                "\n"
            )
            if req.split("==")[0] in requirements
        ]
    )
    # Build
    build_args = {"WHEEL": wheel, "POETRY_EXPORT": reqs}
    build_args = " ".join(
        [f'--build-arg {key}="{value}"' for key, value in build_args.items()]
    )
    with ctx.cd(".."):
        cmd = " ".join(
            [
                "docker build",
                build_args,
                docker_secrets(),
                f"--no-cache --file=Dockerfile --tag={name} .",
            ]
        )
        ctx.run(cmd)


@task
def push_container(ctx, hostname="asia.gcr.io"):
    name = get_container_name(ctx, hostname=hostname)
    # Push
    cmd = f"docker push {name}"
    ctx.run(cmd)
