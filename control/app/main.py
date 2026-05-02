"""qwen-control FastAPI app.

Entrypoint. Wires together:
- perimeter middleware (tailnet allowlist)
- request-logging middleware
- auth (bearer or session)
- /v1/* OpenAI-compatible proxy
- /api/* control-plane REST
- /  HTMX UI
"""
from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import auth, db, docker_ops
from .orchestrator import Orchestrator, Status
from .profiles import load_config
from .proxy import serve_v1


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger("qwen-control")


APP_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))


# ---------- App ------------------------------------------------------------

app = FastAPI(title="qwen-control", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")


# Boot
db.init_db()
config = load_config()
orchestrator = Orchestrator(config)


@app.on_event("startup")
async def _startup() -> None:
    if __import__("os").environ.get("QWEN_AUTO_RESTORE", "1") == "1":
        asyncio.create_task(orchestrator.maybe_restore_last())
    asyncio.create_task(_gc_loop())


async def _gc_loop() -> None:
    while True:
        try:
            n_sess = db.gc_sessions()
            n_req = db.gc_requests()
            if n_sess or n_req:
                log.info("gc: dropped %d sessions, %d request rows", n_sess, n_req)
        except Exception:
            log.exception("gc loop error")
        await asyncio.sleep(3600)


# ---------- Middleware -----------------------------------------------------

@app.middleware("http")
async def perimeter(request: Request, call_next):
    return await auth.perimeter_middleware(request, call_next)


@app.middleware("http")
async def request_log(request: Request, call_next):
    start = time.perf_counter()
    response: Response = await call_next(request)
    duration_ms = int((time.perf_counter() - start) * 1000)
    if request.url.path.startswith(("/v1/", "/api/")):
        try:
            db.log_request(
                method=request.method,
                path=request.url.path,
                status=response.status_code,
                duration_ms=duration_ms,
                profile=orchestrator.state.active_profile,
                tokens=None,
                remote_ip=auth.client_ip(request),
            )
        except Exception:
            log.exception("request log error")
    return response


# ---------- Health ---------------------------------------------------------

@app.get("/healthz", include_in_schema=False)
async def healthz() -> JSONResponse:
    return JSONResponse({"ok": True, "status": orchestrator.state.status.value})


# ---------- Login flow -----------------------------------------------------

@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        "login.html", {"request": request, "error": None}
    )


@app.post("/login")
async def login_submit(request: Request, password: str = Form(...)) -> Response:
    if not auth.verify_admin_password(password):
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": "Wrong password."},
            status_code=401,
        )
    session = auth.make_session(request)
    response = RedirectResponse(url="/", status_code=303)
    response.set_cookie(
        auth.SESSION_COOKIE,
        session,
        max_age=auth.SESSION_TTL,
        httponly=True,
        samesite="strict",
        secure=False,  # tailnet/HTTP is fine; flip to True if fronted by HTTPS
    )
    return response


@app.post("/logout")
async def logout(request: Request) -> RedirectResponse:
    cookie = request.cookies.get(auth.SESSION_COOKIE)
    if cookie:
        auth.revoke_session(cookie)
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie(auth.SESSION_COOKIE)
    return response


# ---------- UI -------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def ui_index(request: Request) -> Response:
    redirect = auth.require_auth_or_redirect(request)
    if redirect:
        return redirect
    return templates.TemplateResponse("index.html", _ui_context(request))


@app.get("/partial/state", response_class=HTMLResponse)
async def ui_partial_state(request: Request) -> Response:
    redirect = auth.require_auth_or_redirect(request)
    if redirect:
        return redirect
    return templates.TemplateResponse(
        "partials/state_card.html", _ui_context(request)
    )


@app.get("/partial/activity", response_class=HTMLResponse)
async def ui_partial_activity(request: Request) -> Response:
    redirect = auth.require_auth_or_redirect(request)
    if redirect:
        return redirect
    return templates.TemplateResponse(
        "partials/activity.html",
        {"request": request, "rows": db.recent_requests(limit=30)},
    )


def _ui_context(request: Request) -> dict[str, Any]:
    return {
        "request": request,
        "state": orchestrator.state,
        "profiles": list(config.profiles.values()),
        "active_profile": orchestrator.state.active_profile,
        "served_model_id": orchestrator.served_model_id() or "",
        "rows": db.recent_requests(limit=20),
        "api_token": auth.read_api_token() if auth.is_session_authenticated(request) else None,
    }


# ---------- API ------------------------------------------------------------

def _gate_api(request: Request) -> Response | None:
    return auth.require_auth_or_401(request)


@app.get("/api/state")
async def api_state(request: Request) -> Response:
    bad = _gate_api(request)
    if bad:
        return bad
    s = orchestrator.state
    gpus = await docker_ops.gpu_memory_used_mib()
    return JSONResponse(
        {
            "status": s.status.value,
            "active_profile": s.active_profile,
            "started_at": s.started_at,
            "served_model_id": orchestrator.served_model_id(),
            "error_message": s.error_message,
            "progress": {
                "target": s.progress.target,
                "started_at": s.progress.started_at,
                "expected_seconds": s.progress.expected_seconds,
                "current_step": s.progress.current_step,
                "log_tail": s.progress.log_lines[-30:],
            },
            "gpus": [{"index": i, "used_mib": m} for i, m in gpus],
            "profiles": [
                {
                    "name": p.name,
                    "display_name": p.display_name,
                    "served_model_id": p.served_model_id,
                    "description": p.description,
                    "expected_load_seconds": p.expected_load_seconds,
                }
                for p in config.profiles.values()
            ],
        }
    )


@app.post("/api/switch/{profile_name}")
async def api_switch(profile_name: str, request: Request) -> Response:
    bad = _gate_api(request)
    if bad:
        return bad
    if profile_name not in config.profiles:
        raise HTTPException(404, f"unknown profile: {profile_name}")
    if orchestrator.state.status == Status.SWITCHING:
        return JSONResponse({"error": "already switching"}, status_code=409)
    await orchestrator.switch_to(profile_name, reason=f"api by {auth.client_ip(request)}")
    return JSONResponse(
        {"status": "switching", "target": profile_name,
         "expected_seconds": config.profiles[profile_name].expected_load_seconds}
    )


@app.post("/api/stop")
async def api_stop(request: Request) -> Response:
    bad = _gate_api(request)
    if bad:
        return bad
    if orchestrator.state.status == Status.SWITCHING:
        return JSONResponse({"error": "switch in progress"}, status_code=409)
    await orchestrator.stop(reason=f"api by {auth.client_ip(request)}")
    return JSONResponse({"status": "off"})


@app.post("/api/rotate-token")
async def api_rotate_token(request: Request) -> Response:
    """Generate a new API token. Requires session auth (UI-driven)."""
    if not auth.is_session_authenticated(request):
        return JSONResponse({"error": "session required"}, status_code=401)
    new_token = auth.rotate_api_token()
    return JSONResponse({"api_token": new_token})


@app.post("/api/change-password")
async def api_change_password(
    request: Request,
    current_password: str = Form(...),
    new_password: str = Form(...),
) -> Response:
    if not auth.is_session_authenticated(request):
        return JSONResponse({"error": "session required"}, status_code=401)
    if not auth.verify_admin_password(current_password):
        return JSONResponse({"error": "wrong current password"}, status_code=401)
    try:
        auth.set_admin_password(new_password)
    except ValueError as e:
        return JSONResponse({"error": str(e)}, status_code=400)
    return JSONResponse({"ok": True})


# ---------- /v1 catch-all proxy --------------------------------------------

@app.api_route("/v1/{rest_path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def v1_proxy(rest_path: str, request: Request) -> Response:
    bad = _gate_api(request)
    if bad:
        return bad
    return await serve_v1(request, rest_path, orchestrator)
