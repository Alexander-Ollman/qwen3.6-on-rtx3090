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

## Configuring coding agents

Every config below points at the same endpoint:

```
Base URL:   http://<host>:9000/v1
API key:    <contents of /var/qwen-control/api_token, or your rotated value>
Model:      qwen36-27b   (or qwen36-35b-moe — whichever profile is active)
```

Replace `<host>` with the host's tailnet IP (the perimeter middleware blocks LAN/public IPs by default).

### OpenCode (Sourcegraph CLI agent)

```bash
export OPENAI_BASE_URL=http://<host>:9000/v1
export OPENAI_API_KEY=<your-token>
opencode --model qwen36-27b 'refactor this module for clarity'
```

Streaming, tool calls, and JSON mode all work — the underlying vLLM stacks have `--tool-call-parser qwen3_coder` enabled.

### Continue (VSCode / JetBrains plugin)

`~/.continue/config.json`:

```json
{
  "models": [{
    "title": "Qwen3.6 (local)",
    "provider": "openai",
    "model": "qwen36-27b",
    "apiBase": "http://<host>:9000/v1",
    "apiKey": "<your-token>",
    "completionOptions": {
      "stream": true,
      "maxTokens": 4096,
      "temperature": 0.2
    }
  }],
  "tabAutocompleteModel": {
    "title": "Qwen3.6 autocomplete",
    "provider": "openai",
    "model": "qwen36-27b",
    "apiBase": "http://<host>:9000/v1",
    "apiKey": "<your-token>"
  }
}
```

For tab autocomplete specifically, the **27B-dense** profile is usually a better fit (more predictable latency on short-context completion); the **35B-MoE** wins for chat/agentic flows where multiple parallel calls dominate.

### Claude Code

Two paths, depending on your Claude Code version:

**Path A — newer versions with OpenAI-compatible mode** (recommended):

```bash
export ANTHROPIC_BASE_URL=http://<host>:9000
export ANTHROPIC_AUTH_TOKEN=<your-token>
export ANTHROPIC_MODEL=qwen36-27b
claude
```

Some Claude Code releases also honor `OPENAI_BASE_URL` / `OPENAI_API_KEY` directly. Check `claude config show` and the version notes for your release.

**Path B — older versions hardcoded to Anthropic API format**: our endpoint speaks OpenAI format, not `/v1/messages`. Use a translator like [LiteLLM](https://github.com/BerriAI/litellm) in front:

```bash
pip install 'litellm[proxy]'
litellm --model openai/qwen36-27b \
        --api_base http://<host>:9000/v1 \
        --api_key <your-token> \
        --port 4000

# Then point Claude Code at LiteLLM's Anthropic-format proxy:
export ANTHROPIC_BASE_URL=http://localhost:4000
export ANTHROPIC_AUTH_TOKEN=anything   # LiteLLM doesn't require its own auth by default
```

A native `/v1/messages` translation layer in qwen-control is a candidate for v2 if you find yourself running Claude Code daily.

### OpenClaw / other Claude-Code TUI forks

Same as Claude Code: most forks support `ANTHROPIC_BASE_URL` (some also `OPENAI_BASE_URL`). Try Path A first. If it 404s on `/v1/messages`, drop in LiteLLM as in Path B.

### "Claude Code via Ollama"

A common configuration mistake: people install Ollama and proxy Claude Code through it, hoping Ollama will translate Anthropic ↔ local. **Ollama doesn't natively serve the Anthropic API** — it serves its own `/api/generate` plus an OpenAI-compatible `/v1/chat/completions`.

If that's your setup, **skip Ollama entirely** — qwen-control's `/v1/*` is a drop-in replacement for Ollama's OpenAI-compatible mode, with much higher throughput on this hardware. Point Claude Code at `http://<host>:9000/v1` directly (Path A above).

### Compatibility matrix

| Client | Streaming | Tool calling | System prompt | Notes |
|---|---|---|---|---|
| OpenCode | ✅ | ✅ | from client | Just works. |
| Continue | ✅ | ✅ | from client | `provider: openai` |
| Cursor | ✅ | ✅ | from client | Settings → Models → Custom OpenAI |
| OpenWebUI | ✅ | ✅ | per-conversation | Admin → Connections → OpenAI API |
| Claude Code (new) | ✅ | ✅ | from client | `ANTHROPIC_BASE_URL` |
| Claude Code (old) | ⚠ via LiteLLM | ✅ via LiteLLM | from client | Anthropic-format → OpenAI translator needed |
| `openai` Python SDK | ✅ | ✅ | from client | `OpenAI(base_url=..., api_key=...)` |
| Ollama wrappers | ✅ | depends | depends | Treat us as the OpenAI-compatible upstream |

## Chat playground

Visit `http://<host>:9000/chat` after logging in. Multi-turn streaming chat with:

- System prompt textarea (pre-filled from the admin default if set, see below)
- **🧠 Reasoning effort** dropdown — Auto / Off / Light / Medium / Heavy / XHeavy. Backed by vLLM's `thinking_token_budget` sampling parameter (a *hard* cap on the `<think>` block; vLLM forces the model to emit `</think>` when the budget is hit).
  - Off: `chat_template_kwargs: {enable_thinking: false}`
  - Light: `thinking_token_budget: 128`
  - Medium: `thinking_token_budget: 512`
  - Heavy: `thinking_token_budget: 2048`
  - XHeavy: `thinking_token_budget: 8192`
  - Auto: no cap, model self-decides
- Temperature, top-p, max-tokens controls. **Use server cap** button fills max-tokens with the active profile's ceiling (15K / 31K).
- Stop button to abort a generation mid-stream
- Live stats: TTFT (ms), elapsed, tokens (real `usage.completion_tokens` from vLLM), tok/s end-to-end + decode-only, finish_reason warning when truncated.

State is browser-only — refresh = lose history. This is intentional; the playground is for testing/demos, not a daily chat client.

### Controlling reasoning from external clients

The `thinking_token_budget` parameter is a **vLLM SamplingParams field** (top-level in the request body), not a chat-template kwarg. To use it from any OpenAI-compatible client:

```python
from openai import OpenAI
client = OpenAI(base_url="http://<host>:9000/v1", api_key="<token>")
resp = client.chat.completions.create(
    model="qwen36-27b",
    messages=[{"role":"user","content":"..."}],
    extra_body={"thinking_token_budget": 512},   # hard cap on <think> block
    max_tokens=4096,                             # reply budget AFTER thinking
)
```

To disable thinking entirely:
```python
extra_body={"chat_template_kwargs": {"enable_thinking": False}}
```

These pass-through cleanly via our proxy — qwen-control doesn't strip or rewrite request bodies.

## Default system prompt (playground-only)

The dashboard's "Playground defaults" card lets you set a default system prompt that pre-fills the chat playground. **It is NOT injected into `/v1/*` requests from external clients** — they always control their own system message. This avoids breaking tool-call schemas, project-context prompts, and other client-side prompting that production agents rely on.

Stored in SQLite at `/var/qwen-control/qwen-control.db`. Edit via the dashboard or directly:

```bash
docker exec -it qwen-control python3 -c "
from app import db
db.set_state('playground_system_prompt', 'You are a helpful assistant.')
"
```

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
