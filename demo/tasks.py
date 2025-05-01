import json
import os
import re
import tempfile
from typing import Any
from urllib.parse import urljoin

from decouple import config
from invoke import task


@task
def django_settings(ctx: Any, proxy: bool = False) -> Any:
    """Django settings."""
    name = "proxy" if proxy else "development"
    os.environ["DJANGO_SETTINGS_MODULE"] = f"demo.settings.{name}"
    import django

    django.setup()
    from django.conf import settings

    return settings


@task
def coverage(ctx: Any) -> None:
    """Coverage."""
    ctx.run("coverage run --source=../ manage.py test quant_tick; coverage report")


@task
def start_proxy(ctx: Any) -> None:
    """Start proxy."""
    host = config("PRODUCTION_DATABASE_HOST")
    port = config("PROXY_DATABASE_PORT")
    ctx.run(f"cloud-tools/cloud-sql-proxy {host} -p {port}")


@task
def create_user(ctx: Any, username: str, password: str, proxy: bool = False) -> None:
    """Create user."""
    django_settings(ctx, proxy=proxy)
    from django.contrib.auth import get_user_model

    User = get_user_model()
    user = User.objects.create(username=username, is_superuser=True, is_staff=True)
    user.set_password(password)
    user.save()


@task
def get_container_name(ctx: Any, name: str, region: str = "asia-northeast1") -> str:
    """Get container name."""
    project_id = ctx.run("gcloud config get-value project").stdout.strip()
    return f"{region}-docker.pkg.dev/{project_id}/{name}/{name}"


def docker_secrets() -> str:
    """Docker secrets."""
    build_args = [
        f'{secret}="{config(secret)}"'
        for secret in (
            "SECRET_KEY",
            "SENTRY_DSN",
            "BINANCE_API_KEY",
            "BINANCE_API_SECRET",
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


def build_quant_tick(ctx: Any) -> str:
    """Build django-quant-tick."""
    result = ctx.run("poetry build").stdout
    return re.search(r"django_quant_tick-.*\.whl", result).group()


def build_container(
    ctx: Any, name: str, suffix: str, region: str = "asia-northeast1"
) -> None:
    """Build container."""
    wheel = build_quant_tick(ctx)
    settings = django_settings(ctx, target="production")
    with open(settings.BASE_DIR.parent.parent / "requirements.txt", "w") as f:
        reqs = ctx.run("poetry export --format=requirements.txt").stdout
        f.write(reqs)
    # Build
    build_args = {"WHEEL": wheel, "POETRY_EXPORT": reqs}
    build_args = " ".join(
        [f'--build-arg {key}="{value}"' for key, value in build_args.items()]
    )
    name = get_container_name(ctx, name)
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
def push_container(ctx: Any, suffix: str, hostname: str = "asia.gcr.io") -> None:
    """Push container."""
    name = get_container_name(ctx, suffix, hostname=hostname)
    # Push
    cmd = f"docker push {name}"
    ctx.run(cmd)


def get_workflow(url: str, exchanges: list[str]) -> dict:
    """Get workflow."""
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


def push_workflow(
    ctx: Any, name: str, workflow: dict, location: str = "asia-northeast1"
) -> None:
    """Push workflow."""
    with tempfile.NamedTemporaryFile(mode="w") as f:
        json.dump(workflow, f)
        f.seek(0)
        ctx.run(
            f"gcloud workflows deploy {name} "
            f"--source={f.name} --location={location}"
        )


@task
def push_rest_workflow(
    ctx: Any, name: str = "django-quant-tick-rest", location: str = "asia-northeast1"
) -> None:
    """Push REST workflow."""
    url = f'https://{config("PRODUCTION_API_URL")}'
    exchanges = ["bitfinex", "bitmex", "coinbase"]
    workflow = get_workflow(url, exchanges)
    push_workflow(ctx, name, workflow, location=location)


@task
def push_s3_workflow(
    ctx: Any, name: str = "django-quant-tick-s3", location: str = "asia-northeast1"
) -> None:
    """Push S3 workflow."""
    url = f'https://{config("PRODUCTION_API_URL")}'
    exchanges = ["bybit"]
    workflow = get_workflow(url, exchanges)
    push_workflow(ctx, name, workflow, location=location)
