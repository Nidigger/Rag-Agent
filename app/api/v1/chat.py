"""Chat API endpoints — SSE streaming and synchronous response.

Provides two endpoints:
- POST /         : Synchronous chat (returns complete reply).
- POST /stream   : SSE streaming chat (token-by-token via two-phase architecture).

The streaming endpoint consumes StreamEvent dicts from ChatService and
serialises them directly to SSE — no hand-crafted event structures here.
"""

import json
import logging

from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

from app.agent.tools.request_context import set_perf_request_id
from app.common.exceptions import AgentGenerationError
from app.common.response import ErrorCode, success
from app.schemas.chat import ChatRequest
from app.services.chat_service import get_chat_service
from app.services.stream_events import done_event, error_event
from app.services.session_service import get_session_service
from app.utils.perf import elapsed_ms, log_perf, now_ms

logger = logging.getLogger("rag-agent.chat_api")
router = APIRouter()


@router.post("")
async def chat(req: ChatRequest):
    """Synchronous chat endpoint — returns the complete reply at once."""
    chat_service = get_chat_service()
    session_service = get_session_service()

    session_id = req.session_id or session_service.create_session()
    history = session_service.get_messages(session_id)

    logger.info(
        "[chat] Sync request: session=%s, message=%r",
        session_id,
        req.message[:80],
    )

    collected_chunks: list[str] = []
    try:
        async for event in chat_service.stream_chat(
            query=req.message, session_id=session_id, messages=history
        ):
            # Extract content tokens from message events
            if event["event"] == "message":
                collected_chunks.append(event["data"]["content"])
    except AgentGenerationError:
        raise

    reply = "".join(collected_chunks)
    session_service.add_message(session_id, "user", req.message)
    session_service.add_message(session_id, "assistant", reply)

    logger.info(
        "[chat] Sync response: session=%s, reply_len=%d",
        session_id,
        len(reply),
    )

    return success(
        data={
            "session_id": session_id,
            "reply": reply,
            "metadata": {"used_tools": [], "trace_id": req.request_id},
        }
    )


@router.post("/stream")
async def chat_stream(req: ChatRequest):
    """SSE streaming chat endpoint.

    Uses the two-phase architecture:
    1. Agent executes tools (non-streaming) and collects context.
    2. Final answer is streamed token-by-token to the client via SSE.

    Events emitted:
    - event: status   → data: {"phase": "...", "message": "..."}
    - event: message  → data: {"content": "..."}
    - event: done     → data: {"session_id": "..."}
    - event: error    → data: {"code": "...", "message": "..."}
    """
    chat_service = get_chat_service()
    session_service = get_session_service()

    session_id = req.session_id or session_service.create_session()
    session_service.add_message(session_id, "user", req.message)

    # Exclude the just-added user message from history for the agent
    history = session_service.get_messages(session_id)
    history_for_agent = history[:-1] if len(history) > 1 else []

    request_id = req.request_id or session_id
    set_perf_request_id(request_id)

    logger.info(
        "[chat/stream] SSE request: session=%s, message=%r",
        session_id,
        req.message[:80],
    )

    sse_start = now_ms()
    log_perf("chat_api", "sse_start",
             request_id=request_id,
             session_id=session_id,
             message_len=len(req.message))

    async def event_generator():
        collected_chunks: list[str] = []
        try:
            async for event in chat_service.stream_chat(
                query=req.message,
                session_id=session_id,
                messages=history_for_agent,
                request_id=request_id,
            ):
                if event["event"] == "message":
                    collected_chunks.append(event["data"]["content"])

                yield {
                    "event": event["event"],
                    "data": json.dumps(event["data"], ensure_ascii=False),
                }

            # Save assistant reply after successful streaming
            reply = "".join(collected_chunks)
            if reply:
                session_service.add_message(session_id, "assistant", reply)

            log_perf("chat_api", "sse_done",
                     request_id=request_id,
                     session_id=session_id,
                     elapsed_ms=elapsed_ms(sse_start),
                     reply_len=len(reply))

            logger.info(
                "[chat/stream] SSE done: session=%s, reply_len=%d",
                session_id,
                len(reply),
            )

            ev = done_event(session_id)
            yield {
                "event": ev["event"],
                "data": json.dumps(ev["data"], ensure_ascii=False),
            }
        except AgentGenerationError as e:
            log_perf("chat_api", "sse_error",
                     request_id=request_id,
                     session_id=session_id,
                     elapsed_ms=elapsed_ms(sse_start),
                     error=str(e)[:80])
            logger.error(
                "[chat/stream] Agent generation failed: session=%s, %s",
                session_id,
                e,
            )
            ev = error_event(
                ErrorCode.AGENT_GENERATION_FAILED, "生成失败，请稍后重试"
            )
            yield {
                "event": ev["event"],
                "data": json.dumps(ev["data"], ensure_ascii=False),
            }
        except Exception as e:
            log_perf("chat_api", "sse_error",
                     request_id=request_id,
                     session_id=session_id,
                     elapsed_ms=elapsed_ms(sse_start),
                     error=str(e)[:80])
            logger.error(
                "[chat/stream] Unexpected error: session=%s, %s",
                session_id,
                e,
                exc_info=True,
            )
            ev = error_event(
                ErrorCode.INTERNAL_SERVER_ERROR, "服务内部错误"
            )
            yield {
                "event": ev["event"],
                "data": json.dumps(ev["data"], ensure_ascii=False),
            }

    return EventSourceResponse(event_generator())
