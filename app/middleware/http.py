"""HTTP middleware — request ID tracking, context propagation, and response timing."""

import time
import uuid

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from app.config import settings


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Attach a unique request ID and propagate tenant/user context."""

    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        # Propagate downstream context headers into request.state so that
        # API handlers and services can access them without parsing headers
        # at every layer.
        request.state.request_id = request_id
        request.state.tenant_id = request.headers.get("X-Tenant-Id", "default")
        request.state.user_id = request.headers.get("X-User-Id", "")
        request.state.endpoint = request.url.path

        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response


class TimingMiddleware(BaseHTTPMiddleware):
    """Measure and record request processing time."""

    async def dispatch(self, request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        elapsed = time.time() - start
        response.headers["X-Process-Time"] = f"{elapsed:.4f}"
        return response


def register_middleware(app: FastAPI) -> None:
    """Register all HTTP middleware on the application."""
    app.add_middleware(TimingMiddleware)
    app.add_middleware(RequestIDMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.server.cors_allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
