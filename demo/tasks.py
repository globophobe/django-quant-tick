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
        name = get_container_name(ctx, name, region=region)
        cmd = " ".join(
            [
                "docker build",
                build_args,
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


@task
def update_container(
    ctx: Any,
    name: str = "django-quant-tick",
    region: str = "asia-northeast1",
) -> None:
    """Build, push, and deploy a traffic-serving Cloud Run revision."""
    build_container(ctx, name=name, region=region)
    push_container(ctx, name=name, region=region)
    image = get_container_name(ctx, name, region=region)
    ctx.run(
        " ".join(
            [
                "gcloud run deploy",
                name,
                f"--image={image}",
                f"--region={region}",
                "--platform=managed",
                "--quiet",
            ]
        )
    )


def _optional_int_env(name: str) -> int | None:
    value = os.environ.get(name) or None
    if value is None:
        return None
    return int(value)


def _parse_csv(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    values = list(
        dict.fromkeys(item.strip() for item in raw.split(",") if item.strip())
    )
    return values or None


def _parse_callback_strategies(raw: str | None) -> list[str] | None:
    return _parse_csv(raw)


def _exchange_base_url(
    url: str,
    exchange: str,
    intl_url: str | None,
    intl_exchanges: set[str],
) -> str:
    return intl_url if intl_url and exchange in intl_exchanges else url


def _aggregate_trades_url(
    url: str,
    exchange: str,
    api_symbol: str,
    intl_url: str | None,
    intl_exchanges: set[str],
) -> str:
    aggregate_trades = urljoin(
        _exchange_base_url(url, exchange, intl_url, intl_exchanges),
        "aggregate-trades/",
    )
    return (
        f"{aggregate_trades}{exchange}/"
        f"?time_ago=7d&api_symbol={quote(api_symbol, safe='')}"
    )


def _exchange_data_step(
    url: str,
    exchange_data_exchanges: list[str] | None,
    intl_url: str | None,
    intl_exchanges: set[str],
) -> dict:
    if exchange_data_exchanges is None:
        target_url = f"{urljoin(url, 'fetch-exchange-data/')}?time_ago=7d"
        return {
            "fetchExchangeData": {
                "try": {
                    "call": "http.get",
                    "args": {
                        "url": target_url,
                        "auth": {"type": "OIDC"},
                    },
                },
                "except": {"as": "e", "steps": []},
            }
        }

    targets = [
        {
            "url": (
                f"{urljoin(_exchange_base_url(url, exchange, intl_url, intl_exchanges), 'fetch-exchange-data/')}"
                f"?time_ago=7d&exchange={quote(exchange, safe='')}"
            )
        }
        for exchange in exchange_data_exchanges
    ]
    return {
        "fetchExchangeData": {
            "parallel": {
                "for": {
                    "value": "exchangeDataTarget",
                    "in": targets,
                    "steps": [
                        {
                            "exchangeData": {
                                "try": {
                                    "call": "http.get",
                                    "args": {
                                        "url": "${exchangeDataTarget.url}",
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
    }


def _validate_callback_strategies(callback_strategies: list[str] | None) -> None:
    if not callback_strategies:
        raise ValueError("CALLBACK_STRATEGIES is required when CALLBACK_URL is set.")


def _callback_strategy_url(callback_url: str, strategy_name: str) -> str:
    return f"{callback_url.rstrip('/')}/{quote(strategy_name, safe='')}/"


def _callback_notify_url(callback_url: str) -> str:
    return urljoin(f"{callback_url.rstrip('/')}/", "../notify/")


def _callback_ready_url(callback_url: str, strategy_name: str) -> str:
    url = urljoin(f"{callback_url.rstrip('/')}/", "../ready/")
    return f"{url.rstrip('/')}/{quote(strategy_name, safe='')}/"


def _validate_positive_minutes(name: str, value: int | None) -> None:
    if value is not None and value <= 0:
        raise ValueError(f"{name} must be positive.")


def _callback_ready_condition(
    *,
    callback_window_period_minutes: int | None = None,
) -> str:
    if callback_window_period_minutes is None:
        raise ValueError("CALLBACK_WINDOW_PERIOD_MINUTES is required when CALLBACK_URL is set.")
    _validate_positive_minutes("CALLBACK_WINDOW_PERIOD_MINUTES", callback_window_period_minutes)
    return f"runMinutes % {callback_window_period_minutes} == 0"


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
    callback_strategies: list[str] | None = None,
    exchange_data_exchanges: list[str] | None = None,
    intl_url: str | None = None,
    intl_exchanges: list[str] | None = None,
) -> dict:
    """Get workflow."""
    intl_exchange_set = set(intl_exchanges or ())
    if intl_exchange_set and not intl_url:
        raise ValueError(
            "PRODUCTION_INTL_API_URL is required when PRODUCTION_INTL_EXCHANGES is set."
        )
    if intl_exchange_set and exchange_data_exchanges is None:
        raise ValueError(
            "exchange_data_exchanges is required for international routing."
        )

    compact = urljoin(url, "compact/")
    all_steps = []
    if callback_url:
        all_steps.append(
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
    steps = [
        {
            "getTradeData": {
                "parallel": {
                    "for": {
                        "value": "item",
                        "in": [
                            {
                                "url": _aggregate_trades_url(
                                    url,
                                    item["exchange"],
                                    item["api_symbol"],
                                    intl_url,
                                    intl_exchange_set,
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
        _exchange_data_step(
            url,
            exchange_data_exchanges,
            intl_url,
            intl_exchange_set,
        ),
    ]
    if callback_url:
        _validate_callback_strategies(callback_strategies)
        ready_condition = _callback_ready_condition(
            callback_window_period_minutes=callback_window_period_minutes,
        )
        ready_targets = [
            {"url": _callback_ready_url(callback_url, strategy_name)}
            for strategy_name in callback_strategies
        ]
        all_steps += [
            {
                "process": {
                    "parallel": {
                        "branches": [
                            {
                                "maybeReady": {
                                    "steps": [
                                        {
                                            "checkReady": {
                                                "switch": [
                                                    {
                                                        "condition": f"${{{ready_condition}}}",
                                                        "next": "ready",
                                                    }
                                                ],
                                                "next": "skipReady",
                                            }
                                        },
                                        {
                                            "ready": {
                                                "parallel": {
                                                    "for": {
                                                        "value": "readyTarget",
                                                        "in": ready_targets,
                                                        "steps": [
                                                            {
                                                                "readyStrategy": {
                                                                    "try": {
                                                                        "call": "http.get",
                                                                        "args": {
                                                                            "url": "${readyTarget.url}",
                                                                            "auth": {"type": "OIDC"},
                                                                        },
                                                                    },
                                                                    "except": {"as": "e", "steps": []},
                                                                }
                                                            }
                                                        ],
                                                    }
                                                },
                                                "next": "readyDone",
                                            }
                                        },
                                        {
                                            "skipReady": {
                                                "assign": [{"readySkipped": True}],
                                                "next": "readyDone",
                                            }
                                        },
                                        {"readyDone": {"assign": [{"readyDone": True}]}},
                                    ]
                                }
                            },
                            {
                                "getData": {
                                    "steps": steps,
                                }
                            },
                        ]
                    }
                }
            }
        ]
    else:
        all_steps += steps
    if callback_url:
        condition = _callback_condition(
            callback_window_period_minutes=callback_window_period_minutes,
            callback_window_duration_minutes=callback_window_duration_minutes,
        )
        all_steps += [
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
                    "parallel": {
                        "for": {
                            "value": "callbackTarget",
                            "in": [
                                {"url": _callback_strategy_url(callback_url, strategy_name)}
                                for strategy_name in callback_strategies
                            ],
                            "steps": [
                                {
                                    "callbackStrategy": {
                                        "call": "http.post",
                                        "args": {
                                            "url": "${callbackTarget.url}",
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
                                    }
                                }
                            ],
                        }
                    },
                    "next": "notify",
                }
            },
            {
                "notify": {
                    "call": "http.post",
                    "args": {
                        "url": _callback_notify_url(callback_url),
                        "auth": {"type": "OIDC"},
                    },
                    "next": "compact",
                }
            },
        ]
    all_steps += [
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
    return {"main": {"steps": all_steps}}


@task
def push_workflow(
    ctx: Any,
    name: str = "django-quant-tick",
    location: str = "asia-northeast1",
) -> None:
    """Push workflow."""
    django_settings(ctx, proxy=True)
    from quant_tick.constants import Exchange
    from quant_tick.models import Symbol
    from quant_tick.views import FetchExchangeDataView

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
    exchange_data_exchanges = list(FetchExchangeDataView().get_configured_exchanges())

    url = os.environ["PRODUCTION_API_URL"]
    intl_url = os.environ.get("PRODUCTION_INTL_API_URL") or None
    intl_exchanges = _parse_csv(os.environ.get("PRODUCTION_INTL_EXCHANGES"))
    unknown_intl_exchanges = sorted(set(intl_exchanges or ()) - set(Exchange.values))
    if unknown_intl_exchanges:
        raise ValueError(
            "unknown PRODUCTION_INTL_EXCHANGES: " + ", ".join(unknown_intl_exchanges)
        )
    if intl_exchanges and not intl_url:
        raise ValueError(
            "PRODUCTION_INTL_API_URL is required when PRODUCTION_INTL_EXCHANGES is set."
        )
    callback_url = os.environ.get("CALLBACK_URL") or None
    callback_strategies = _parse_callback_strategies(os.environ.get("CALLBACK_STRATEGIES"))
    callback_window_period_minutes = _optional_int_env("CALLBACK_WINDOW_PERIOD_MINUTES")
    callback_window_duration_minutes = _optional_int_env("CALLBACK_WINDOW_DURATION_MINUTES")

    workflow = get_workflow(
        url,
        symbols,
        callback_url=callback_url,
        callback_window_period_minutes=callback_window_period_minutes,
        callback_window_duration_minutes=callback_window_duration_minutes,
        callback_strategies=callback_strategies,
        exchange_data_exchanges=exchange_data_exchanges,
        intl_url=intl_url,
        intl_exchanges=intl_exchanges,
    )
    with tempfile.NamedTemporaryFile(mode="w") as f:
        json.dump(workflow, f)
        f.seek(0)
        ctx.run(
            f"gcloud workflows deploy {name} "
            f"--source={f.name} --location={location}"
        )
