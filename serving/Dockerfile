FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first (layer caching)
COPY pyproject.toml .
RUN pip install --no-cache-dir -e ".[serving]" 2>/dev/null || pip install --no-cache-dir \
    fastapi>=0.111 \
    uvicorn[standard]>=0.29 \
    pydantic>=2.7 \
    scikit-learn>=1.4 \
    numpy>=1.26 \
    pandas>=2.2 \
    mlflow>=2.13 \
    joblib \
    httpx>=0.27 \
    python-dotenv>=1.0 \
    pyarrow>=16

# Copy source
COPY data/ ./data/
COPY models/ ./models/
COPY pipelines/ ./pipelines/
COPY serving/ ./serving/

# Non-root user
RUN useradd -m -u 1000 sensorops && chown -R sensorops:sensorops /app
USER sensorops

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "serving.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info"]
