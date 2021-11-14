FROM python:3.9-slim-buster

ARG PROJECT_ID
ARG BIGQUERY_LOCATION
ARG BIGQUERY_DATASET

ENV PROJECT_ID $PROJECT_ID
ENV BIGQUERY_LOCATION $BIGQUERY_LOCATION
ENV BIGQUERY_DATASET $BIGQUERY_DATASET

ADD cryptofeed_werks /cryptofeed_werks/
ADD main.py /

ADD requirements.txt /

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove build-essential \
    && apt-get clean  \
    && rm -rf /var/lib/apt/lists/*

ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", ${VAR1:="8080"}]
