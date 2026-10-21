# CPU-only serving image. Build from the repo root:
#   docker build -f serve/Dockerfile -t volsent-api .
#   docker run -p 8000:8000 volsent-api
# The checkpoint is a training artifact, not committed: `make deep` produces it, and CI downloads it
# from the `model-latest` GitHub release. To serve stored forecasts, mount the SQLite store:
#   docker run -p 8000:8000 -v $PWD/artifacts/store:/app/artifacts/store volsent-api
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 VOLSENT_THREADS=1

COPY serve/requirements.txt /app/serve/requirements.txt
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.5.1 \
 && pip install --no-cache-dir -r /app/serve/requirements.txt

COPY src/__init__.py src/config.py /app/src/
COPY src/models/__init__.py src/models/deep.py /app/src/models/
COPY serve/ /app/serve/
COPY artifacts/models/v2_lstm_price/lstm_seed0_test2023.pt /app/artifacts/models/v2_lstm_price/

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "serve.api:app", "--host", "0.0.0.0", "--port", "8000"]
