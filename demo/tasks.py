import json
import os
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

from dotenv import load_dotenv
from invoke import task

load_dotenv(Path(__file__).resolve().with_name(".env"))


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
    credentials = Path(__file__).parent.parent.parent.joinpath("keys", os.environ["GOOGLE_APPLICATION_CREDENTIALS"]).resolve()
    ctx.run(f"cloud-tools/cloud-sql-proxy -c {credentials} {host} -p {port}")


@task
def get_container_name(
    ctx: Any,
    name: str = "django-quant-tick",
    region: str = "asia-northeast1",
) -> str:
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
    repo_root = Path(__file__).resolve().parent.parent
    dist_dir = repo_root / "dist"
    if dist_dir.exists():
        for wheel in dist_dir.glob("django_quant_tick-*.whl"):
            wheel.unlink()
    with ctx.cd(str(repo_root)):
        ctx.run("uv build --wheel")
    wheels = sorted(dist_dir.glob("django_quant_tick-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"expected exactly one built wheel, found {len(wheels)}")
    return wheels[0].name


@task
def build_container(
    ctx: Any,
    name: str = "django-quant-tick",
    region: str = "asia-northeast1",
) -> None:
    wheel = build_quant_tick(ctx)
    repo_root = Path(__file__).resolve().parent.parent
    requirements = repo_root / "requirements.txt"
    with ctx.cd(str(repo_root)):
        ctx.run(
            "uv export "
            "--format requirements.txt "
            "--group deploy "
            "--no-dev "
            "--no-header "
            "--no-annotate "
            "--no-editable "
            "--no-hashes "
            "--no-emit-project "
            "--frozen "
            f"--output-file {requirements.name}"
        )
        build_args = {"WHEEL": wheel}
        build_args = " ".join(
            [f'--build-arg {key}="{value}"' for key, value in build_args.items()]
        )
        name = get_container_name(ctx, name)
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
def push_container(
    ctx: Any,
    name: str = "django-quant-tick",
    region: str = "asia-northeast1",
) -> None:
    name = get_container_name(ctx, name, region=region)
    # Push
    cmd = f"docker push {name}"
    ctx.run(cmd)


def get_workflow(
    url: str,
    exchanges: list[str],
    callback_url: str | None = None,
    callback_interval_minutes: int | None = None,
) -> dict:
    """Build the Cloud Workflow definition for REST aggregation."""
    aggregate_trades = urljoin(url, "aggregate-trades/")
    aggregate_candles = urljoin(url, "aggregate-candles/")
    compact = urljoin(url, "compact/")
    steps = []
    if callback_url and callback_interval_minutes:
        steps.append(
            {
                "getRunTime": {
                    "assign": [
                        {"runTime": "${time.format(sys.now())}"},
                        {
                            "runMinutes": "${int(text.substring(runTime, 11, 13)) * 60 + int(text.substring(runTime, 14, 16))}"
                        },
                    ]
                }
            }
        )
    steps += [
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
                                            "url": '${"' + aggregate_trades + '" + exchange + "/?time_ago=7d"}',
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
                    "url": f"{aggregate_candles}?time_ago=7d",
                    "auth": {"type": "OIDC"},
                },
            }
        },
    ]
    if callback_url and callback_interval_minutes:
        steps += [
            {
                "maybeCallback": {
                    "switch": [
                        {
                            "condition": f"${{runMinutes % {callback_interval_minutes} == 0}}",
                            "next": "callback",
                        }
                    ],
                    "next": "compact",
                }
            },
            {
                "callback": {
                    "call": "http.post",
                    "args": {
                        "url": callback_url,
                        "auth": {"type": "OIDC"},
                        "body": {"timestamp": "${runTime}"},
                    },
                    "next": "compact",
                }
            },
        ]
    steps += [
        {
            "compact": {
                "call": "http.get",
                "args": {
                    "url": f"{compact}?time_ago=7d",
                    "auth": {"type": "OIDC"},
                },
            }
        },
    ]
    return {"main": {"steps": steps}}


@task
def push_workflow(
    ctx: Any, name: str = "django-quant-tick", location: str = "asia-northeast1"
) -> None:
    url = os.environ["PRODUCTION_API_URL"]
    callback_url = os.environ.get("CALLBACK_URL") or None
    callback_interval_minutes = os.environ.get("CALLBACK_INTERVAL_MINUTES") or None
    if callback_interval_minutes is not None:
        callback_interval_minutes = int(callback_interval_minutes)
        if callback_interval_minutes <= 0:
            raise ValueError("CALLBACK_INTERVAL_MINUTES must be positive.")
    if callback_url and callback_interval_minutes is None:
        raise ValueError("CALLBACK_INTERVAL_MINUTES is required when CALLBACK_URL is set.")
    exchanges = ["binance", "bitfinex", "bitmex", "coinbase"]
    workflow = get_workflow(
        url,
        exchanges,
        callback_url=callback_url,
        callback_interval_minutes=callback_interval_minutes,
    )
    with tempfile.NamedTemporaryFile(mode="w") as f:
        json.dump(workflow, f)
        f.seek(0)
        ctx.run(
            f"gcloud workflows deploy {name} "
            f"--source={f.name} --location={location}"
        )
