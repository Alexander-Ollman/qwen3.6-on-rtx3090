# qwen-control

A small web app that:

- Runs as a Docker container alongside the Qwen3.6 stacks.
- Knows how to **stop one model and start the other** without leaving GPUs in a wedged state.
- Exposes **one OpenAI-compatible endpoint** (`/v1/*`) backed by whichever model is currently loaded — so Cursor, Continue, OpenWebUI, your own agent, etc. point at one URL forever.
- Provides a **web UI** behind a tailnet perimeter and admin-password login for switching, rotating tokens, and watching activity.
- **Auto-restores the last-active model on reboot.**
- Survives the host coming back up: persistent state in `/var/qwen-control`.

```
   client (Cursor / Continue / your agent)
            │  Authorization: Bearer <api-token>
            ▼
   ┌─────────────────────────────────────┐
   │  http://<host>:9000/                │  ← qwen-control
   │  ├─ /              (UI)              │     (Docker, host network,
   │  ├─ /api/...       (REST control)    │      restart=always)
   │  └─ /v1/...        (OpenAI proxy)    │
   └────────────────┬────────────────────┘
            stops/starts via docker socket
                    ▼
        ┌────────────────────┐  ┌──────────────────┐
        │ 27B-dense stack    │  │ 35B-A3B MoE stack│
        │ (qwen36-vllm-1/2,  │  │ (qwen36-moe,     │
        │  qwen36-lb)        │  │  TP=2 + EP)      │
        └────────────────────┘  └──────────────────┘
```

## Security model

Three layers, in order:

1. **Tailnet perimeter** (`QWEN_REQUIRE_TAILNET=1`, default on). Requests from outside the configured Tailscale CIDRs (`100.64.0.0/10`, `fd7a:115c:a1e0::/48`) get a 403 before any other code runs. Loopback (`127.0.0.1`) is always allowed for healthchecks.
2. **Bearer token** for `/api/*` and `/v1/*`. 32-byte hex value at `/var/qwen-control/api_token` (chmod 600). Generate-once on install; rotate-able from the UI. **Required for every non-UI request.**
3. **Admin password** for the UI. Bcrypt hash at `/var/qwen-control/admin_password.bcrypt`. Logging in sets a 7-day signed session cookie. Once logged in, you can rotate the API token and change the admin password.

### Lost the admin password?

```bash
sudo bash control/install.sh --reset
```

Regenerates the API token AND prompts for a new admin password. Existing sessions become invalid.

## Quick start

Prerequisites: you've already done the work in the [main README](../README.md) — driver 580, vLLM image pulled, Genesis patches cloned, both Qwen3.6 model checkpoints downloaded.

```bash
# From the repo root:
bash control/install.sh
# Prompts for an admin password, generates an API token, disables auto-restart on
# the GPU-competing containers (3-proxy-docker-vllm-1, sglang, ollama),
# builds the qwen-control image, starts it.
```

When it finishes:

```
===============================================
  qwen-control is starting on port 9000.

  Web UI:   http://<this-host>:9000/
  API key:  <copy-this-now>
===============================================
```

Visit the URL on a tailnet-connected machine, log in with the admin password.

## Using the OpenAI-compatible endpoint

Treat `http://<host>:9000/v1/` exactly like an OpenAI API base URL.

```bash
curl http://<host>:9000/v1/chat/completions \
  -H "Authorization: Bearer <api-token>" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen36-27b",
    "messages": [{"role":"user","content":"Hi"}],
    "stream": true
  }'
```

The `model` field is informational — whichever model is currently active will respond. `GET /v1/models` returns the active model's id.

### What clients see during a switch

- New requests get **`HTTP 503 + Retry-After: <seconds>`** with body `{"error":{"message":"model switching in progress","type":"switching"}}`.
- In-flight streaming requests run to completion (or hit a 30 s grace timer).
- After ~90 s the new model is up and the endpoint resumes 200s.

Most OpenAI clients retry on 503 automatically.

## Configuration

`control/profiles.yaml` defines the two model profiles — what containers belong to each, what URL the readiness check uses, what the upstream port is. Edit it to add a third profile (e.g. a smaller model, or a future Qwen 3.7), then `docker compose -f control/docker-compose.yml restart`.

`control/.env` (created by install.sh) exposes:

| Variable | Default | Purpose |
|---|---|---|
| `QWEN_REPO_ROOT` | repo root | Mounted read-only at `/host/repo` so the container can call `launch-*.sh` |
| `QWEN_CONTROL_PORT` | `9000` | Listen port |
| `QWEN_TAILNET_CIDRS` | `100.64.0.0/10,fd7a:115c:a1e0::/48` | Comma-separated CIDRs allowed past the perimeter |
| `QWEN_REQUIRE_TAILNET` | `1` | Set to `0` to disable the tailnet check (testing only) |
| `QWEN_AUTO_RESTORE` | `1` | Restore the last-active profile on container startup |

## Files in this directory

| File | Role |
|---|---|
| `Dockerfile` | Builds the `qwen-control:latest` image (Python 3.12 + FastAPI + docker SDK) |
| `docker-compose.yml` | Runs the container with host networking, `restart=always`, docker socket mount |
| `install.sh` | One-shot host setup (state dir, token, password, GPU competitor disable, build, start) |
| `profiles.yaml` | Static profile definitions — edit to add models |
| `app/main.py` | FastAPI entrypoint: routes, middleware, startup hooks |
| `app/orchestrator.py` | State machine (`off` ↔ `switching` ↔ `active`) |
| `app/proxy.py` | Streaming OpenAI-compatible proxy (`/v1/*`) |
| `app/auth.py` | Bearer + admin password + tailnet middleware |
| `app/db.py` | SQLite schema and helpers |
| `app/docker_ops.py` | docker SDK wrappers + `nvidia-smi` reader |
| `app/profiles.py` | YAML parsing |
| `app/templates/` | Jinja2 + HTMX UI |
| `app/static/style.css` | Dark-theme stylesheet matching the blog |

## Operational notes

- **`docker exec qwen-control sqlite3 /var/qwen-control/qwen-control.db`** to poke at the activity log directly.
- **Logs**: `docker logs qwen-control -f` — structured per-line, includes orchestrator state transitions.
- **Update**: `git pull && docker compose -f control/docker-compose.yml up -d --build`.
- **Uninstall**: `docker compose -f control/docker-compose.yml down`. State directory `/var/qwen-control` is preserved unless you also `sudo rm -rf` it.

## Deliberate non-goals (v1)

- Per-user API keys / RBAC. One shared bearer token; token rotation is per-deploy, not per-user.
- Editing model profiles from the UI. YAML edit + restart.
- Real-time tok/s graphs. `/api/state` exposes the data; build a Grafana panel if you want one.
- Multi-host scheduling. One control plane per host.

If/when these get wanted, they're additive — the state machine and proxy are the load-bearing pieces and they don't preclude any of it.
