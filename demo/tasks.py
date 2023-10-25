import json
import os
import re
import tempfile
from urllib.parse import urljoin

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
    ctx.run("coverage run --source=../ manage.py test quant_tick; coverage report")


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
def get_container_name(ctx, suffix, hostname="asia.gcr.io"):
    project_id = ctx.run("gcloud config get-value project").stdout.strip()
    return f"{hostname}/{project_id}/django-quant-candles-{suffix}"


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
            "PRODUCTION_API_URL",
        )
    ]
    return " ".join([f"--build-arg {build_arg}" for build_arg in build_args])


def get_common_requirements():
    return [
        "django-filter",
        "django-polymorphic",
        "djangorestframework",
        "django-storages",
        "google-cloud-storage",
        "gunicorn",
        "pandas",
        "pyarrow",
        "psycopg2-binary",
        "python-decouple",
        "randomname",
        "sentry-sdk",
    ]


def build_quant_tick(ctx):
    result = ctx.run("poetry build").stdout
    return re.search(r"django_quant_tick-.*\.whl", result).group()


def build_container(ctx, suffix: str, requirements: list[str], hostname="asia.gcr.io"):
    wheel = build_quant_tick(ctx)
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
    name = get_container_name(ctx, suffix=suffix, hostname=hostname)
    with ctx.cd(".."):
        cmd = " ".join(
            [
                "docker build",
                build_args,
                docker_secrets(),
                f"--no-cache --file=docker/Dockerfile.{suffix} --tag={name} .",
            ]
        )
        ctx.run(cmd)


@task
def build_frontend(ctx, hostname="asia.gcr.io"):
    ctx.run("echo yes | python manage.py collectstatic")
    requirements = get_common_requirements() + [
        "django-semantic-admin",
        "whitenoise",
    ]
    build_container(ctx, suffix="frontend", requirements=requirements)


@task
def build_api(ctx):
    build_container(ctx, suffix="api", requirements=get_common_requirements())


@task
def push_container(ctx, suffix, hostname="asia.gcr.io"):
    name = get_container_name(ctx, suffix, hostname=hostname)
    # Push
    cmd = f"docker push {name}"
    ctx.run(cmd)


def get_workflow(url, exchanges):
    aggregate_trades = urljoin(url, "aggregate-trades/")
    aggregate_candles = urljoin(url, "aggregate-candles/")
    return {
        "main": {
            "steps": [
                {
                    "getTradeData": {
                        "parallel": {
                            "for": {
                                "value": "exchange",
                                "in": exchanges,
                                "steps": [
                                    {
                                        "tradeData": {
                                            "try": {
                                                "call": "http.get",
                                                "args": {
                                                    "url": aggregate_trades,
                                                    "auth": {"type": "OIDC"},
                                                    "query": {
                                                        "exchange": "${exchange}"
                                                    },
                                                },
                                            },
                                            "except": {"as": "e", "steps": []},
                                        }
                                    }
                                ],
                            }
                        }
                    }
                },
                {
                    "aggregateCandles": {
                        "call": "http.get",
                        "args": {
                            "url": aggregate_candles,
                            "auth": {"type": "OIDC"},
                        },
                    }
                },
            ]
        }
    }


def push_workflow(ctx, name, workflow, location="asia-northeast1"):
    with tempfile.NamedTemporaryFile(mode="w") as f:
        json.dump(workflow, f)
        f.seek(0)
        ctx.run(
            f"gcloud workflows deploy {name} "
            f"--source={f.name} --location={location}"
        )


@task
def push_rest_workflow(ctx, location="asia-northeast1"):
    url = f'https://{config("PRODUCTION_API_URL")}'
    exchanges = ["bitfinex", "bitmex", "coinbase"]
    workflow = get_workflow(url, exchanges)
    push_workflow(ctx, "django-quant-candles-rest", workflow, location=location)


@task
def push_s3_workflow(ctx, location="asia-northeast1"):
    url = f'https://{config("PRODUCTION_API_URL")}'
    exchanges = ["bybit"]
    workflow = get_workflow(url, exchanges)
    push_workflow(ctx, "django-quant-candles-s3", workflow, location=location)
