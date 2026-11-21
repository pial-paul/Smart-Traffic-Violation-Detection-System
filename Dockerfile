FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.lock ./
RUN python -m pip install --require-hashes -r requirements.lock
COPY pyproject.toml README.md LICENSE ./
COPY trafficlab/ trafficlab/
COPY demos/ demos/
RUN python -m pip install --no-deps . \
    && useradd --system --uid 10001 --create-home reviewer \
    && install -d -o reviewer -g reviewer /app/instance
USER reviewer
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=4s --start-period=10s CMD python -c "from urllib.request import urlopen; urlopen('http://localhost:8080/healthz', timeout=3).close()"
ENTRYPOINT ["python", "-m", "trafficlab"]
CMD ["serve", "--host", "0.0.0.0"]
