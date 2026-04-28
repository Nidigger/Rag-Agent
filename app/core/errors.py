import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.response import ErrorCode, error

logger = logging.getLogger("rag-agent.errors")


class AppException(Exception):
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
    def __init__(self, message: str = "Agent generation failed"):
        super().__init__(
            error_code=ErrorCode.AGENT_GENERATION_FAILED,
            message=message,
            status_code=500,
        )


class RAGRetrievalError(AppException):
    def __init__(self, message: str = "RAG retrieval failed"):
        super().__init__(
            error_code=ErrorCode.RAG_RETRIEVAL_FAILED,
            message=message,
            status_code=500,
        )


class ModelTimeoutError(AppException):
    def __init__(self, message: str = "Model call timed out"):
        super().__init__(
            error_code=ErrorCode.MODEL_TIMEOUT,
            message=message,
            status_code=504,
        )


class ToolCallError(AppException):
    def __init__(self, message: str = "Tool call failed"):
        super().__init__(
            error_code=ErrorCode.TOOL_CALL_FAILED,
            message=message,
            status_code=500,
        )


class SessionNotFoundError(AppException):
    def __init__(self, message: str = "Session not found"):
        super().__init__(
            error_code=ErrorCode.SESSION_NOT_FOUND,
            message=message,
            status_code=404,
        )


def register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        request: Request, exc: StarletteHTTPException
    ):
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
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content=error(
                status_code=500,
                error_code=ErrorCode.INTERNAL_SERVER_ERROR,
                message="Internal server error",
                request_id=request.headers.get("X-Request-ID"),
            ),
        )
