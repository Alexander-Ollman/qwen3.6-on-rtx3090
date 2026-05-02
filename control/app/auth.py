"""Two-tier auth:
- API token (long-lived bearer) for /v1/* and /api/* programmatic access.
- Admin password → session cookie for the web UI; can rotate the API token.

Plus an outer perimeter: optional tailnet CIDR allowlist enforced as middleware.
"""
from __future__ import annotations

import ipaddress
import os
import secrets
import time
from pathlib import Path
from typing import Optional

from passlib.hash import bcrypt
from starlette.requests import Request
from starlette.responses import JSONResponse, RedirectResponse, Response

from . import db


STATE_DIR = Path(os.environ.get("QWEN_CONTROL_STATE_DIR", "/var/qwen-control"))
TOKEN_FILE = STATE_DIR / "api_token"
PWHASH_FILE = STATE_DIR / "admin_password.bcrypt"

SESSION_COOKIE = "qwen_session"
SESSION_TTL = 7 * 24 * 3600  # 7 days


# --- API token -------------------------------------------------------------

def read_api_token() -> str:
    if not TOKEN_FILE.exists():
        raise RuntimeError(f"API token missing at {TOKEN_FILE}; run install.sh first.")
    return TOKEN_FILE.read_text().strip()


def rotate_api_token() -> str:
    new = secrets.token_hex(32)
    TOKEN_FILE.write_text(new)
    TOKEN_FILE.chmod(0o600)
    return new


# --- Admin password --------------------------------------------------------

def verify_admin_password(plaintext: str) -> bool:
    if not PWHASH_FILE.exists():
        return False
    stored = PWHASH_FILE.read_text().strip()
    try:
        return bcrypt.verify(plaintext, stored)
    except Exception:
        return False


def set_admin_password(plaintext: str) -> None:
    if len(plaintext) < 12:
        raise ValueError("Admin password must be at least 12 characters.")
    h = bcrypt.hash(plaintext)
    PWHASH_FILE.write_text(h)
    PWHASH_FILE.chmod(0o600)


# --- Sessions --------------------------------------------------------------

def make_session(request: Request) -> str:
    token = secrets.token_urlsafe(32)
    db.create_session(
        token,
        ttl_seconds=SESSION_TTL,
        user_agent=request.headers.get("user-agent", "")[:256],
        remote_ip=client_ip(request),
    )
    return token


def session_valid(token: Optional[str]) -> bool:
    if not token:
        return False
    return db.lookup_session(token) is not None


def revoke_session(token: str) -> None:
    db.revoke_session(token)


# --- Request authentication ------------------------------------------------

def client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def bearer_token(request: Request) -> Optional[str]:
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth.split(" ", 1)[1].strip()
    return None


def is_authenticated(request: Request) -> bool:
    """True if the request carries either a valid bearer or a valid session cookie."""
    token = bearer_token(request)
    if token and secrets.compare_digest(token, read_api_token()):
        return True
    cookie = request.cookies.get(SESSION_COOKIE)
    return session_valid(cookie)


def is_session_authenticated(request: Request) -> bool:
    return session_valid(request.cookies.get(SESSION_COOKIE))


# --- Tailnet perimeter middleware -----------------------------------------

def _parse_cidrs(env_value: str) -> list:
    out = []
    for piece in env_value.split(","):
        piece = piece.strip()
        if not piece:
            continue
        try:
            out.append(ipaddress.ip_network(piece, strict=False))
        except ValueError:
            pass
    return out


_TAILNET_CIDRS = _parse_cidrs(
    os.environ.get("QWEN_TAILNET_CIDRS", "100.64.0.0/10,fd7a:115c:a1e0::/48")
)
_REQUIRE_TAILNET = os.environ.get("QWEN_REQUIRE_TAILNET", "1") == "1"


def in_tailnet(ip_str: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_str)
    except ValueError:
        return False
    return any(ip in cidr for cidr in _TAILNET_CIDRS)


async def perimeter_middleware(request: Request, call_next):
    """Reject any request whose source IP isn't on the tailnet (configurable)."""
    if _REQUIRE_TAILNET:
        ip = client_ip(request)
        # Allow loopback so health checks and same-host scripts work
        if ip not in ("127.0.0.1", "::1") and not in_tailnet(ip):
            return JSONResponse(
                {"error": "forbidden: not on tailnet"}, status_code=403
            )
    return await call_next(request)


# --- Auth gates for routes -------------------------------------------------

def require_auth_or_redirect(request: Request) -> Optional[Response]:
    """For UI routes — redirect to /login if not authenticated."""
    if is_session_authenticated(request):
        return None
    bearer = bearer_token(request)
    if bearer and secrets.compare_digest(bearer, read_api_token()):
        return None
    return RedirectResponse(url="/login", status_code=303)


def require_auth_or_401(request: Request) -> Optional[Response]:
    """For API routes — return 401 if not authenticated."""
    if is_authenticated(request):
        return None
    return JSONResponse({"error": "unauthorized"}, status_code=401)
