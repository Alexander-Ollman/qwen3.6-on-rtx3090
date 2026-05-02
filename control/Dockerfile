FROM python:3.12-slim

# docker CLI for falling back from python SDK if the host socket version mismatches
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates docker.io \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

COPY app/requirements.txt /srv/app/requirements.txt
RUN pip install --no-cache-dir -r /srv/app/requirements.txt

COPY app /srv/app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/srv \
    QWEN_CONTROL_STATE_DIR=/var/qwen-control \
    QWEN_CONTROL_PROFILES=/etc/qwen-control/profiles.yaml

EXPOSE 9000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://127.0.0.1:9000/healthz || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9000", "--proxy-headers"]
