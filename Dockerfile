FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY chordlistener ./chordlistener
RUN pip install --no-cache-dir ".[server]"

RUN useradd --create-home app && mkdir /data && chown app /data
USER app
ENV CL_DATA_DIR=/data
VOLUME /data
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/v1/health', timeout=4)"

CMD ["uvicorn", "chordlistener.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
