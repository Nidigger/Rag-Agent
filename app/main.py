"""FastAPI application factory.

Creates and configures the application with:
- Logging setup
- CORS and timing middleware
- Global exception handlers
- API v1 routes
"""

from fastapi import FastAPI

from app.api.v1.router import api_v1_router
from app.config import settings
from app.common.exceptions import register_exception_handlers
from app.middleware.http import register_middleware
from app.observability.logging import setup_logging


def create_app() -> FastAPI:
    setup_logging()

    app = FastAPI(
        title=settings.server.project_name,
        version=settings.server.version,
        docs_url="/api/docs",
        openapi_url="/api/openapi.json",
    )

    register_middleware(app)
    register_exception_handlers(app)

    app.include_router(api_v1_router, prefix=settings.server.api_v1_prefix)

    return app


app = create_app()
