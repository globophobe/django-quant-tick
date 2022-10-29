import os

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
        f'{secret}="{config(secret)}"' for secret in ("SECRET_KEY", "SENTRY_DSN")
    ]
    return " ".join([f"--build-arg {build_arg}" for build_arg in build_args])
