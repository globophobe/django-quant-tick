FROM python:3.10-slim-bullseye

ARG POETRY_EXPORT
ARG SECRET_KEY
ARG SENTRY_DSN

ARG DATABASE_NAME
ARG DATABASE_USER
ARG DATABASE_PASSWORD
ARG PRODUCTION_DATABASE_HOST
ARG DATABASE_PORT

ENV SECRET_KEY $SECRET_KEY
ENV SENTRY_DSN $SENTRY_DSN

ENV DATABASE_NAME $DATABASE_NAME
ENV DATABASE_USER $DATABASE_USER
ENV DATABASE_PASSWORD $DATABASE_PASSWORD
ENV PRODUCTION_DATABASE_HOST $PRODUCTION_DATABASE_HOST
ENV DATABASE_PORT $DATABASE_PORT

COPY pyproject.toml /
copy poetry.lock /
COPY quant_candles/ /quant_candles
COPY demo/demo /demo/demo
COPY demo/static /demo/static
COPY demo/templates /demo/templates
COPY demo/db.sqlite3 /demo/db.sqlite3

RUN apt-get update \
    && pip install --no-cache-dir wheel \
    && pip install poetry \
    && poetry install \
    && apt-get clean  \
    && rm -rf /var/lib/apt/lists/* \
    && rm $WHEEL

# Start the server
ENTRYPOINT ["gunicorn", "--chdir", "/demo", "--bind", "0.0.0.0:8080", "demo.wsgi:application"]
