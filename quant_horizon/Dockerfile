# Multi-stage build for minimal image size
FROM python:3.10-slim as builder

WORKDIR /app

# Install poetry
RUN pip install poetry

# Copy quant_core (dependency)
COPY quant_core/ /app/quant_core/

# Copy quant_horizon pyproject files
COPY quant_horizon/pyproject.toml quant_horizon/poetry.lock* /app/quant_horizon/

# Export dependencies to requirements.txt
WORKDIR /app/quant_horizon
RUN poetry export -f requirements.txt --output requirements.txt --without-hashes || \
    (poetry lock && poetry export -f requirements.txt --output requirements.txt --without-hashes)

# Final stage
FROM python:3.10-slim

WORKDIR /app

# Copy requirements and install
COPY --from=builder /app/quant_horizon/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy quant_core
COPY quant_core/ /app/quant_core/
RUN pip install --no-cache-dir -e /app/quant_core

# Copy quant_horizon application
COPY quant_horizon/quant_horizon/ /app/quant_horizon/

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=""
ENV PORT=8080

# Run the application
CMD uvicorn quant_horizon.main:app --host 0.0.0.0 --port ${PORT}
