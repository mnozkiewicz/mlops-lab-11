# Build stage
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim AS builder

WORKDIR /app

# Install dependencies
COPY pyproject.toml ./
# Sync only inference group to .venv
RUN uv sync --group deployment --no-install-project --no-default-groups

# Runtime stage
FROM python:3.12-slim-bookworm

WORKDIR /app

# Copy virtual environment
COPY --from=builder /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# Copy application code
COPY sentiment_app ./sentiment_app
COPY src/scripts/settings.py ./src/scripts/settings.py
COPY .env ./.env

# Copy model artifacts (ONNX + tokenizer.json + classifier.joblib)
COPY artifacts/onnx_model ./artifacts/onnx_model

# Run the application
ENTRYPOINT ["python", "-m", "awslambdaric"]
CMD ["sentiment_app.app.handler"]