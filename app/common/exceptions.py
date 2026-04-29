"""Application error types and global exception handlers.

Defines custom exception classes and registers handlers on the FastAPI app
to ensure consistent error responses across all endpoints.
"""

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.common.response import ErrorCode, error

logger = logging.getLogger("rag-agent.errors")


class AppException(Exception):
    """Base application exception with structured error info."""

    def __init__(
        self,
        error_code: str,
        message: str,
        status_code: int = 500,
    ):
        self.error_code = error_code
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class AgentGenerationError(AppException):
    """Raised when the Agent or streaming phase fails."""

    def __init__(self, message: str = "Agent generation failed"):
        super().__init__(
            error_code=ErrorCode.AGENT_GENERATION_FAILED,
            message=message,
            status_code=500,
        )


class RAGRetrievalError(AppException):
    """Raised when RAG vector retrieval fails."""

    def __init__(self, message: str = "RAG retrieval failed"):
        super().__init__(
            error_code=ErrorCode.RAG_RETRIEVAL_FAILED,
            message=message,
            status_code=500,
        )


class ModelTimeoutError(AppException):
    """Raised when an LLM call exceeds the timeout threshold."""

    def __init__(self, message: str = "Model call timed out"):
        super().__init__(
            error_code=ErrorCode.MODEL_TIMEOUT,
            message=message,
            status_code=504,
        )


class ToolCallError(AppException):
    """Raised when a tool invocation fails."""

    def __init__(self, message: str = "Tool call failed"):
        super().__init__(
            error_code=ErrorCode.TOOL_CALL_FAILED,
            message=message,
            status_code=500,
        )


class SessionNotFoundError(AppException):
    """Raised when a referenced session does not exist."""

    def __init__(self, message: str = "Session not found"):
        super().__init__(
            error_code=ErrorCode.SESSION_NOT_FOUND,
            message=message,
            status_code=404,
        )


class VectorStoreUnavailableError(AppException):
    """Raised when the vector store is not reachable."""

    def __init__(self, message: str = "Vector store unavailable"):
        super().__init__(
            error_code=ErrorCode.VECTOR_STORE_UNAVAILABLE,
            message=message,
            status_code=503,
        )


class DocumentNotFoundError(AppException):
    """Raised when a requested document is not found."""

    def __init__(self, message: str = "Document not found"):
        super().__init__(
            error_code=ErrorCode.DOCUMENT_NOT_FOUND,
            message=message,
            status_code=404,
        )


class IngestFailedError(AppException):
    """Raised when document ingestion fails."""

    def __init__(self, message: str = "Document ingestion failed"):
        super().__init__(
            error_code=ErrorCode.INGEST_FAILED,
            message=message,
            status_code=500,
        )


class UnauthorizedError(AppException):
    """Raised when an internal API call lacks valid authentication."""

    def __init__(self, message: str = "Unauthorized"):
        super().__init__(
            error_code=ErrorCode.UNAUTHORIZED,
            message=message,
            status_code=401,
        )


def register_exception_handlers(app: FastAPI) -> None:
    """Register global exception handlers on the FastAPI app."""

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        request: Request, exc: StarletteHTTPException
    ):
        logger.warning(
            "[errors] HTTP %d: %s %s",
            exc.status_code,
            request.method,
            request.url.path,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=error(
                status_code=exc.status_code,
                error_code=ErrorCode.INTERNAL_SERVER_ERROR,
                message=str(exc.detail),
                request_id=request.headers.get("X-Request-ID"),
            ),
        )

    @app.exception_handler(AppException)
    async def app_exception_handler(request: Request, exc: AppException):
        logger.error(
            "[errors] AppException: code=%s, status=%d, msg=%s",
            exc.error_code,
            exc.status_code,
            exc.message,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=error(
                status_code=exc.status_code,
                error_code=exc.error_code,
                message=exc.message,
                request_id=request.headers.get("X-Request-ID"),
            ),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ):
        details = str(exc.errors())
        logger.warning(
            "[errors] Validation error: %s %s — %s",
            request.method,
            request.url.path,
            details[:200],
        )
        return JSONResponse(
            status_code=422,
            content=error(
                status_code=422,
                error_code=ErrorCode.VALIDATION_ERROR,
                message=f"Request validation error: {details}",
                request_id=request.headers.get("X-Request-ID"),
            ),
        )

    @app.exception_handler(Exception)
    async def generic_error_handler(request: Request, exc: Exception):
        logger.error(
            "[errors] Unhandled exception: %s %s — %s",
            request.method,
            request.url.path,
            exc,
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content=error(
                status_code=500,
                error_code=ErrorCode.INTERNAL_SERVER_ERROR,
                message="Internal server error",
                request_id=request.headers.get("X-Request-ID"),
            ),
        )
