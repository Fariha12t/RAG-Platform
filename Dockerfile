FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 DATA_DIR=/data
RUN apt-get update && apt-get install -y --no-install-recommends tesseract-ocr poppler-utils libpq5 curl \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Optional semantic models:  docker build --build-arg WITH_ML=1 .
ARG WITH_ML=0
COPY requirements-ml.txt .
RUN if [ "$WITH_ML" = "1" ]; then pip install --no-cache-dir -r requirements-ml.txt; fi
COPY . .
RUN useradd -m app && mkdir -p /data && chown -R app /data /app
USER app
EXPOSE 8000 8501
HEALTHCHECK CMD curl -fs http://localhost:8000/health || exit 1
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
