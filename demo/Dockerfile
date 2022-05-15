FROM python:3.9-slim-buster

ARG POETRY_EXPORT

ARG SECRET_KEY

ARG DATABASE_NAME
ARG DATABASE_USER
ARG DATABASE_PASSWORD
ARG PRODUCTION_DATABASE_HOST
ARG DATABASE_PORT

ARG PROJECT_ID
ARG BIGQUERY_LOCATION
ARG BIGQUERY_DATASET

ARG SENTRY_DSN


ENV SECRET_KEY $SECRET_KEY

ENV DATABASE_NAME $DATABASE_NAME
ENV DATABASE_USER $DATABASE_USER
ENV DATABASE_PASSWORD $DATABASE_PASSWORD
ENV PRODUCTION_DATABASE_HOST $PRODUCTION_DATABASE_HOST
ENV DATABASE_PORT $DATABASE_PORT

ENV PROJECT_ID $PROJECT_ID
ENV BIGQUERY_LOCATION $BIGQUERY_LOCATION
ENV BIGQUERY_DATASET $BIGQUERY_DATASET

ENV SENTRY_DSN $SENTRY_DSN

ADD cryptofeed_werks /cryptofeed_werks/
ADD main.py /

ADD requirements.txt /

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove build-essential \
    && apt-get clean  \
    && rm -rf /var/lib/apt/lists/*

ENTRYPOINT ["gunicorn", "demo.asgi:application", "--bind", "${PORT:8080}", "-k",  "uvicorn.workers.UvicornWorker", "--threads",  "8"]
