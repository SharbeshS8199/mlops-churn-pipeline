FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
COPY pyproject.toml ./
COPY src ./src
RUN pip install -e '.[api]'
COPY app ./app
COPY params.yaml ./
# The gated champion produced by CI (`python -m churn.train`) is baked into the image.
COPY models/champion ./models/champion
RUN useradd -m appuser && mkdir -p logs && chown -R appuser /app
USER appuser
EXPOSE 8080
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
