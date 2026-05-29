import json
import os
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import quote, urljoin

from django.db.models import Q
from dotenv import load_dotenv
from invoke import task

load_dotenv(Path(__file__).resolve().with_name(".env"))


@task
def django_settings(ctx: Any, proxy: bool = False) -> Any:
    os.environ["DJANGO_SETTINGS_MODULE"] = (
        "demo.settings.development.proxy" if proxy else "demo.settings.development.local"
    )
    import django

    django.setup()
    from django.conf import settings

    return settings


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
    home = Path.home()
    credentials = home.joinpath("keys", os.environ["GOOGLE_APPLICATION_CREDENTIALS"]).resolve()
    proxy = home.joinpath("cloud-tools", "cloud-sql-proxy").resolve()
    ctx.run(f"{proxy} -c {credentials} {host} -p {port}")


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
            "DATABASE_NAME",
            "DATABASE_USER",
            "DATABASE_PASSWORD",
            "PRODUCTION_DATABASE_HOST",
            "DATABASE_PORT",
            "GCS_BUCKET_NAME",
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


def _optional_int_env(name: str) -> int | None:
    value = os.environ.get(name) or None
    if value is None:
        return None
    return int(value)


def _validate_positive_minutes(name: str, value: int | None) -> None:
    if value is not None and value <= 0:
        raise ValueError(f"{name} must be positive.")


def _callback_condition(
    *,
    callback_window_period_minutes: int | None = None,
    callback_window_duration_minutes: int | None = None,
) -> str:
    if callback_window_period_minutes is None or callback_window_duration_minutes is None:
        raise ValueError(
            "CALLBACK_WINDOW_PERIOD_MINUTES and CALLBACK_WINDOW_DURATION_MINUTES are required when CALLBACK_URL is set."
        )
    _validate_positive_minutes("CALLBACK_WINDOW_PERIOD_MINUTES", callback_window_period_minutes)
    _validate_positive_minutes("CALLBACK_WINDOW_DURATION_MINUTES", callback_window_duration_minutes)
    if callback_window_duration_minutes > callback_window_period_minutes:
        raise ValueError("CALLBACK_WINDOW_DURATION_MINUTES must be <= CALLBACK_WINDOW_PERIOD_MINUTES.")
    return f"runMinutes % {callback_window_period_minutes} <= {callback_window_duration_minutes}"


def get_workflow(
    url: str,
    symbols: list[dict[str, str]],
    callback_url: str | None = None,
    callback_window_period_minutes: int | None = None,
    callback_window_duration_minutes: int | None = None,
) -> dict:
    """Get workflow."""
    aggregate_trades = urljoin(url, "aggregate-trades/")
    fetch_exchange_data_url = urljoin(url, "fetch-exchange-data/")
    compact = urljoin(url, "compact/")
    steps = []
    if callback_url:
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
                        "value": "item",
                        "in": [
                            {
                                "url": (
                                    f"{aggregate_trades}{item['exchange']}/"
                                    f"?time_ago=7d&api_symbol="
                                    f"{quote(item['api_symbol'], safe='')}"
                                )
                            }
                            for item in symbols
                        ],
                        "steps": [
                            {
                                "tradeData": {
                                    "try": {
                                        "call": "http.get",
                                        "args": {
                                            "url": "${item.url}",
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
    ]
    steps += [
        {
            "fetchExchangeData": {
                "try": {
                    "call": "http.get",
                    "args": {
                        "url": f"{fetch_exchange_data_url}?time_ago=7d",
                        "auth": {"type": "OIDC"},
                    },
                },
                "except": {"as": "e", "steps": []},
            }
        },
    ]
    if callback_url:
        condition = _callback_condition(
            callback_window_period_minutes=callback_window_period_minutes,
            callback_window_duration_minutes=callback_window_duration_minutes,
        )
        steps += [
            {
                "maybeCallback": {
                    "switch": [
                        {
                            "condition": f"${{{condition}}}",
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
                        "body": {
                            "as_of": "${runTime}",
                            "final_retry": (
                                "${"
                                f"runMinutes % {callback_window_period_minutes} == "
                                f"{callback_window_duration_minutes}"
                                "}"
                            ),
                        },
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
    ctx: Any,
    name: str = "django-quant-tick",
    location: str = "asia-northeast1",
) -> None:
    """Push workflow."""
    django_settings(ctx, proxy=True)
    from quant_tick.models import Symbol

    symbols = list(
        Symbol.objects.filter(is_active=True)
        .filter(
            Q(save_raw=True)
            | Q(save_aggregated=True)
            | Q(significant_trade_filter__gt=0)
        )
        .order_by("exchange", "api_symbol")
        .values("exchange", "api_symbol")
    )
    if not symbols:
        raise RuntimeError("No active symbols.")

    url = os.environ["PRODUCTION_API_URL"]
    callback_url = os.environ.get("CALLBACK_URL") or None
    callback_window_period_minutes = _optional_int_env("CALLBACK_WINDOW_PERIOD_MINUTES")
    callback_window_duration_minutes = _optional_int_env("CALLBACK_WINDOW_DURATION_MINUTES")

    workflow = get_workflow(
        url,
        symbols,
        callback_url=callback_url,
        callback_window_period_minutes=callback_window_period_minutes,
        callback_window_duration_minutes=callback_window_duration_minutes,
    )
    with tempfile.NamedTemporaryFile(mode="w") as f:
        json.dump(workflow, f)
        f.seek(0)
        ctx.run(
            f"gcloud workflows deploy {name} "
            f"--source={f.name} --location={location}"
        )
