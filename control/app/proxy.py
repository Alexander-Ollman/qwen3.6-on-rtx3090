"""Streaming OpenAI-compatible proxy.

Forwards `/v1/*` requests to the active profile's upstream. Streaming responses
(SSE for chat completions) are passed through chunk-by-chunk without buffering.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Optional

import httpx
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse

from .orchestrator import Orchestrator, Status


log = logging.getLogger("qwen-control.proxy")


HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailers", "transfer-encoding", "upgrade", "host", "content-length",
}


def _filter_request_headers(req: Request) -> dict:
    out = {}
    for k, v in req.headers.items():
        kl = k.lower()
        if kl in HOP_BY_HOP:
            continue
        # Don't forward our own auth onward — vLLM doesn't need it.
        if kl == "authorization":
            continue
        out[k] = v
    return out


def _filter_response_headers(headers: httpx.Headers) -> list[tuple[str, str]]:
    return [(k, v) for k, v in headers.items() if k.lower() not in HOP_BY_HOP]


async def serve_v1(
    request: Request,
    rest_path: str,
    orch: Orchestrator,
) -> Response:
    """Handle any /v1/<rest_path> request."""
    # /v1/models is answered locally so it always reflects the active model
    # without needing a network round trip.
    if request.method == "GET" and rest_path == "models":
        return _models_response(orch)

    # Everything else: must have an active backend.
    if orch.state.status == Status.SWITCHING:
        return JSONResponse(
            {"error": {"message": "model switching in progress", "type": "switching"}},
            status_code=503,
            headers={"Retry-After": str(max(orch.state.progress.expected_seconds, 30))},
        )
    if orch.state.status == Status.ERROR:
        return JSONResponse(
            {"error": {"message": f"control plane error: {orch.state.error_message}", "type": "error"}},
            status_code=503,
        )
    if orch.state.status == Status.OFF or not orch.upstream_for_active():
        return JSONResponse(
            {"error": {"message": "no model active", "type": "off"}},
            status_code=503,
            headers={"Retry-After": "30"},
        )

    upstream = orch.upstream_for_active()
    target_url = f"{upstream}/v1/{rest_path}"

    body = await request.body()
    headers = _filter_request_headers(request)

    # Determine whether the client wants streaming. For OpenAI-style chat
    # requests, the request body has "stream": true.
    stream = False
    try:
        if body:
            data = json.loads(body)
            if isinstance(data, dict):
                stream = bool(data.get("stream"))
    except (ValueError, TypeError):
        pass

    client = httpx.AsyncClient(timeout=httpx.Timeout(900.0, connect=10.0))

    try:
        if stream:
            req = client.build_request(
                request.method, target_url, content=body, headers=headers
            )
            upstream_resp = await client.send(req, stream=True)

            async def gen():
                try:
                    async for chunk in upstream_resp.aiter_raw():
                        yield chunk
                finally:
                    await upstream_resp.aclose()
                    await client.aclose()

            return StreamingResponse(
                gen(),
                status_code=upstream_resp.status_code,
                headers=dict(_filter_response_headers(upstream_resp.headers)),
                media_type=upstream_resp.headers.get("content-type"),
            )

        # Non-streaming
        upstream_resp = await client.request(
            request.method, target_url, content=body, headers=headers
        )
        await client.aclose()
        return Response(
            content=upstream_resp.content,
            status_code=upstream_resp.status_code,
            headers=dict(_filter_response_headers(upstream_resp.headers)),
            media_type=upstream_resp.headers.get("content-type"),
        )

    except httpx.RequestError as e:
        await client.aclose()
        log.warning("upstream request error: %s", e)
        return JSONResponse(
            {"error": {"message": f"upstream unreachable: {e}", "type": "upstream_error"}},
            status_code=502,
        )


def _models_response(orch: Orchestrator) -> JSONResponse:
    """Mimic OpenAI's /v1/models with whichever model is currently active."""
    active_id = orch.served_model_id()
    if not active_id:
        return JSONResponse({"object": "list", "data": []})
    return JSONResponse(
        {
            "object": "list",
            "data": [
                {
                    "id": active_id,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "qwen-control",
                }
            ],
        }
    )
