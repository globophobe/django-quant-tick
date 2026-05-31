FROM python:3.12-slim-bookworm

ARG WHEEL

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

COPY requirements.txt /tmp/requirements.txt
COPY dist/${WHEEL} /tmp/${WHEEL}
COPY demo/demo /demo/demo

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements.txt \
    && pip install --no-cache-dir /tmp/${WHEEL} \
    && rm /tmp/requirements.txt /tmp/${WHEEL}

ENTRYPOINT ["gunicorn", "--chdir", "/demo", "--bind", "0.0.0.0:8080", "--timeout", "0", "--preload", "demo.wsgi.production:application"]
