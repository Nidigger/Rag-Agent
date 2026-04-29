"""FastAPI application factory.

Creates and configures the application with:
- Logging setup
- CORS and timing middleware
- Global exception handlers
- API v1 routes
"""

from fastapi import FastAPI

from app.api.v1.router import api_v1_router
from app.core.config import settings
from app.core.errors import register_exception_handlers
from app.core.logging import setup_logging
from app.core.middleware import register_middleware


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
