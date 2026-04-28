import json
import logging

from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

from app.core.errors import AgentGenerationError
from app.core.response import ErrorCode, success
from app.schemas.chat import ChatRequest
from app.services.chat_service import get_chat_service
from app.services.session_service import get_session_service

logger = logging.getLogger("rag-agent.chat_api")
router = APIRouter()


@router.post("")
async def chat(req: ChatRequest):
    chat_service = get_chat_service()
    session_service = get_session_service()

    session_id = req.session_id or session_service.create_session()

    history = session_service.get_messages(session_id)

    collected_chunks: list[str] = []
    try:
        async for chunk in chat_service.stream_chat(
            query=req.message, session_id=session_id, messages=history
        ):
            collected_chunks.append(chunk)
    except AgentGenerationError:
        raise

    reply = "".join(collected_chunks)
    session_service.add_message(session_id, "user", req.message)
    session_service.add_message(session_id, "assistant", reply)

    return success(
        data={
            "session_id": session_id,
            "reply": reply,
            "metadata": {"used_tools": [], "trace_id": req.request_id},
        }
    )


@router.post("/stream")
async def chat_stream(req: ChatRequest):
    chat_service = get_chat_service()
    session_service = get_session_service()

    session_id = req.session_id or session_service.create_session()
    session_service.add_message(session_id, "user", req.message)

    history = session_service.get_messages(session_id)
    # Exclude the just-added user message from history for the agent
    history_for_agent = history[:-1] if len(history) > 1 else []

    async def event_generator():
        collected_chunks: list[str] = []
        try:
            async for chunk in chat_service.stream_chat(
                query=req.message, session_id=session_id, messages=history_for_agent
            ):
                collected_chunks.append(chunk)
                yield {
                    "event": "message",
                    "data": json.dumps(
                        {"content": chunk}, ensure_ascii=False
                    ),
                }

            # Save assistant reply after successful streaming
            reply = "".join(collected_chunks)
            if reply:
                session_service.add_message(session_id, "assistant", reply)

            yield {
                "event": "done",
                "data": json.dumps({"session_id": session_id}),
            }
        except AgentGenerationError as e:
            logger.error(f"Stream chat failed: {e}")
            yield {
                "event": "error",
                "data": json.dumps(
                    {
                        "code": ErrorCode.AGENT_GENERATION_FAILED,
                        "message": "生成失败，请稍后重试",
                    },
                    ensure_ascii=False,
                ),
            }
        except Exception as e:
            logger.error(f"Unexpected stream error: {e}", exc_info=True)
            yield {
                "event": "error",
                "data": json.dumps(
                    {
                        "code": ErrorCode.INTERNAL_SERVER_ERROR,
                        "message": "服务内部错误",
                    },
                    ensure_ascii=False,
                ),
            }

    return EventSourceResponse(event_generator())
