#!/usr/bin/env bash
# qwen-control host installer.
#
# - Creates /var/qwen-control (state dir)
# - Generates initial API token (32-byte hex)
# - Prompts for admin password (or reads QWEN_ADMIN_PASSWORD env var); stores bcrypt hash
# - Disables auto-restart on GPU-competing containers and stops them
# - Builds + starts the qwen-control docker container
#
# Idempotent: re-runnable. Won't clobber an existing api token unless --reset is given.

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
CONTROL_DIR="$REPO_ROOT/control"
STATE_DIR="${QWEN_STATE_DIR:-/var/qwen-control}"
RESET=0

for arg in "$@"; do
  case "$arg" in
    --reset) RESET=1;;
    --help|-h)
      cat <<EOF
Usage: install.sh [--reset]

  --reset   Regenerate API token and admin password even if they exist.

Env overrides:
  QWEN_STATE_DIR        State directory (default /var/qwen-control)
  QWEN_ADMIN_PASSWORD   If set, used non-interactively for the admin password
EOF
      exit 0
      ;;
  esac
done

require() { command -v "$1" >/dev/null || { echo "ERROR: '$1' not found in PATH" >&2; exit 1; }; }
require docker
require openssl

echo "[install] state dir: $STATE_DIR"
# When run with sudo, `id -u` returns root's UID. Honor SUDO_UID/GID so the
# state dir + files are owned by the invoking user, not root.
TARGET_UID="${SUDO_UID:-$(id -u)}"
TARGET_GID="${SUDO_GID:-$(id -g)}"
sudo mkdir -p "$STATE_DIR"
sudo chown -R "$TARGET_UID:$TARGET_GID" "$STATE_DIR"
sudo chmod 750 "$STATE_DIR"

# --- API token -------------------------------------------------------------
TOKEN_FILE="$STATE_DIR/api_token"
if [[ ! -f "$TOKEN_FILE" || $RESET -eq 1 ]]; then
  openssl rand -hex 32 > "$TOKEN_FILE"
  chmod 600 "$TOKEN_FILE"
  echo "[install] generated new API token at $TOKEN_FILE"
else
  echo "[install] keeping existing API token at $TOKEN_FILE"
fi

# --- Admin password --------------------------------------------------------
PWHASH_FILE="$STATE_DIR/admin_password.bcrypt"
if [[ ! -f "$PWHASH_FILE" || $RESET -eq 1 ]]; then
  if [[ -n "${QWEN_ADMIN_PASSWORD:-}" ]]; then
    PW="$QWEN_ADMIN_PASSWORD"
  else
    echo
    echo "Set an admin password. This is what you'll use to log into the qwen-control web UI."
    echo "Pick something memorable but >= 12 chars; it can be rotated later from the UI."
    while true; do
      read -rs -p "Admin password: " PW; echo
      read -rs -p "Confirm:        " PW2; echo
      [[ -n "$PW" && "$PW" == "$PW2" && ${#PW} -ge 12 ]] && break
      echo "Passwords didn't match or were too short (<12). Try again."
    done
  fi
  # bcrypt the password by shelling out to a tiny python one-liner inside the
  # control image once it's built. To avoid chicken-and-egg, use 'docker run'
  # against python:3.12-slim with passlib installed.
  HASH=$(docker run --rm python:3.12-slim sh -c "
    pip install --quiet 'passlib[bcrypt]==1.7.4' bcrypt==4.0.1 >/dev/null
    python3 -c 'import sys; from passlib.hash import bcrypt; print(bcrypt.hash(sys.argv[1]))' \"$PW\"
  ")
  printf '%s' "$HASH" > "$PWHASH_FILE"
  chmod 600 "$PWHASH_FILE"
  unset PW PW2 HASH
  echo "[install] stored admin password hash at $PWHASH_FILE"
else
  echo "[install] keeping existing admin password hash at $PWHASH_FILE"
fi

# --- Disable GPU competitors ----------------------------------------------
echo "[install] disabling auto-restart on GPU-competing containers (one-time)…"
for c in 3-proxy-docker-vllm-1 sglang ollama; do
  if docker inspect "$c" >/dev/null 2>&1; then
    cur_policy=$(docker inspect -f '{{.HostConfig.RestartPolicy.Name}}' "$c" 2>/dev/null || echo "")
    if [[ "$cur_policy" != "no" ]]; then
      docker update --restart=no "$c" >/dev/null
      echo "    $c: restart policy → no"
    fi
    if docker ps -q -f name="^${c}$" | grep -q .; then
      docker stop "$c" >/dev/null && echo "    $c: stopped"
    fi
  fi
done

# --- Build + launch -------------------------------------------------------
cd "$CONTROL_DIR"

# Surface the repo root so the control container can mount the launch scripts.
echo "QWEN_REPO_ROOT=$REPO_ROOT" > .env

echo "[install] building qwen-control image…"
docker compose build --quiet

echo "[install] starting qwen-control…"
docker compose up -d

echo
echo "==============================================="
echo "  qwen-control is starting on port 9000."
echo
echo "  Web UI:   http://<this-host>:9000/"
echo "  API key:  $(cat "$TOKEN_FILE")"
echo
echo "  (Save the API key now — you can rotate it"
echo "  from the UI once you log in with the admin"
echo "  password you just set.)"
echo "==============================================="
