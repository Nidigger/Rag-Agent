from typing import Any


class ErrorCode:
    VALIDATION_ERROR = "VALIDATION_ERROR"
    AGENT_GENERATION_FAILED = "AGENT_GENERATION_FAILED"
    RAG_RETRIEVAL_FAILED = "RAG_RETRIEVAL_FAILED"
    MODEL_TIMEOUT = "MODEL_TIMEOUT"
    TOOL_CALL_FAILED = "TOOL_CALL_FAILED"
    SESSION_NOT_FOUND = "SESSION_NOT_FOUND"
    INTERNAL_SERVER_ERROR = "INTERNAL_SERVER_ERROR"
    VECTOR_STORE_UNAVAILABLE = "VECTOR_STORE_UNAVAILABLE"
    DOCUMENT_NOT_FOUND = "DOCUMENT_NOT_FOUND"
    INGEST_FAILED = "INGEST_FAILED"
    UNAUTHORIZED = "UNAUTHORIZED"


def success(data: Any = None, message: str = "success") -> dict:
    return {"code": 0, "message": message, "data": data}


def error(
    status_code: int,
    error_code: str,
    message: str,
    request_id: str | None = None,
) -> dict:
    resp: dict[str, Any] = {
        "code": status_code,
        "message": message,
        "error_code": error_code,
    }
    if request_id:
        resp["request_id"] = request_id
    return resp
