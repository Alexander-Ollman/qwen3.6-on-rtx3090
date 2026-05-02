FROM python:3.12-slim

# Static docker CLI (the launcher scripts use `docker run`, so we need the
# client binary in the container). The `docker.io` Debian package ships
# dockerd but NOT the docker CLI — that's `docker-cli` upstream. Use the
# official static build instead; it's a single file and version-pinned.
ARG DOCKER_CLI_VERSION=27.5.1
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates \
 && curl -fsSL "https://download.docker.com/linux/static/stable/x86_64/docker-${DOCKER_CLI_VERSION}.tgz" \
      -o /tmp/docker.tgz \
 && tar -xzf /tmp/docker.tgz -C /tmp \
 && install -m 0755 /tmp/docker/docker /usr/local/bin/docker \
 && rm -rf /tmp/docker /tmp/docker.tgz \
 && rm -rf /var/lib/apt/lists/* \
 && docker --version

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
