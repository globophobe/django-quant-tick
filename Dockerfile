FROM python:3.10-slim-bullseye

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

COPY README.md /
COPY pyproject.toml /
copy poetry.lock /
COPY quant_candles/ /quant_candles
COPY demo/demo /demo/demo
COPY demo/static /demo/static
COPY demo/templates /demo/templates
COPY demo/db.sqlite3 /demo/db.sqlite3

RUN pip install poetry \
    && poetry config installer.max-workers 10 \
    && poetry config virtualenvs.create false \
    && poetry install \
    && rm -rf ~/.cache

# Start the server
ENTRYPOINT ["gunicorn", "--chdir", "/demo", "--bind", "0.0.0.0:8080", "demo.wsgi:application"]
