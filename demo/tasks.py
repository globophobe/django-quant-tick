import json
import os
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

from invoke import task


@task
def django_settings(ctx: Any, proxy: bool = False) -> Any:
    name = "proxy" if proxy else "development"
    os.environ["DJANGO_SETTINGS_MODULE"] = f"demo.settings.{name}"
    import django

    django.setup()
    from django.conf import settings

    return settings


@task
def test(ctx: Any) -> None:
    ctx.run("python manage.py test quant_tick")


@task
def coverage(ctx: Any) -> None:
    ctx.run("coverage run --source=../ manage.py test quant_tick; coverage report")


@task
def lint(ctx: Any) -> None:
    ctx.run("ruff check ../")


@task
def format(ctx: Any) -> None:
    ctx.run("ruff check ../ --fix")


@task
def makemigrations(ctx: Any) -> None:
    ctx.run("python manage.py makemigrations quant_tick")


@task
def migrate(ctx: Any) -> None:
    ctx.run("python manage.py migrate")


@task
def start_proxy(ctx: Any) -> None:
    host = os.environ["PRODUCTION_DATABASE_HOST"]
    port = os.environ["PROXY_DATABASE_PORT"]
    ctx.run(f"cloud-tools/cloud-sql-proxy {host} -p {port}")


@task
def get_container_name(ctx: Any, name: str, region: str = "asia-northeast1") -> str:
    """Build the Artifact Registry image name."""
    project_id = ctx.run("gcloud config get-value project").stdout.strip()
    return f"{region}-docker.pkg.dev/{project_id}/{name}/{name}"


def docker_secrets() -> str:
    """Docker build args for the current deploy contract."""
    build_args = [
        f'{secret}="{os.environ[secret]}"'
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
    """Build the wheel and return its filename."""
    dist_dir = Path("dist")
    if dist_dir.exists():
        for wheel in dist_dir.glob("django_quant_tick-*.whl"):
            wheel.unlink()
    ctx.run("uv build --wheel")
    wheels = sorted(dist_dir.glob("django_quant_tick-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"expected exactly one built wheel, found {len(wheels)}")
    return wheels[0].name


def build_container(
    ctx: Any, name: str, region: str = "asia-northeast1"
) -> None:
    wheel = build_quant_tick(ctx)
    repo_root = Path(__file__).resolve().parent.parent
    requirements = repo_root / "requirements.txt"
    ctx.run(
        "uv export "
        "--format requirements.txt "
        "--group deploy "
        "--no-header "
        "--no-annotate "
        "--no-editable "
        "--no-hashes "
        "--no-emit-project "
        f"--output-file {requirements}"
    )
    # Build
    build_args = {"WHEEL": wheel}
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
                "--no-cache --file=Dockerfile",
                f"--tag={name} .",
            ]
        )
        ctx.run(cmd)


@task
def push_container(ctx: Any, name: str, region: str = "asia-northeast1") -> None:
    name = get_container_name(ctx, name, region=region)
    # Push
    cmd = f"docker push {name}"
    ctx.run(cmd)


def get_workflow(url: str, exchanges: list[str]) -> dict:
    """Build the Cloud Workflow definition for REST aggregation."""
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
                                                    "url": f"{aggregate_trades}${{exchange}}/",
                                                    "auth": {"type": "OIDC"},
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
    """Deploy a Cloud Workflow definition."""
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
    url = f'https://{os.environ["PRODUCTION_API_URL"]}'
    exchanges = ["binance", "bitfinex", "bitmex", "coinbase"]
    workflow = get_workflow(url, exchanges)
    push_workflow(ctx, name, workflow, location=location)
