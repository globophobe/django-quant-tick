FROM python:3.12-slim-bookworm

ARG WHEEL
ARG SECRET_KEY

ARG DATABASE_NAME
ARG DATABASE_USER
ARG DATABASE_PASSWORD
ARG PRODUCTION_DATABASE_HOST
ARG DATABASE_PORT
ARG GCS_BUCKET_NAME

ENV SECRET_KEY=$SECRET_KEY

ENV DATABASE_NAME=$DATABASE_NAME
ENV DATABASE_USER=$DATABASE_USER
ENV DATABASE_PASSWORD=$DATABASE_PASSWORD
ENV PRODUCTION_DATABASE_HOST=$PRODUCTION_DATABASE_HOST
ENV DATABASE_PORT=$DATABASE_PORT
ENV GCS_BUCKET_NAME=$GCS_BUCKET_NAME

COPY requirements.txt /tmp/requirements.txt
COPY dist/${WHEEL} /tmp/${WHEEL}
COPY demo/demo /demo/demo

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements.txt \
    && pip install --no-cache-dir /tmp/${WHEEL} \
    && rm /tmp/requirements.txt /tmp/${WHEEL}

ENTRYPOINT ["gunicorn", "--chdir", "/demo", "--bind", "0.0.0.0:8080", "--timeout", "0", "--preload", "demo.wsgi.production:application"]
