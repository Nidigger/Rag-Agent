"""Report API endpoint — SSE streaming report generation.

Uses the two-phase architecture:
1. Agent executes tools (non-streaming) to gather data.
2. Report is streamed token-by-token to the client via SSE.

The endpoint consumes StreamEvent dicts from ReportService and
serialises them directly to SSE — no hand-crafted event structures.
"""

import json
import logging

from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

from app.agent.tools.request_context import set_perf_request_id
from app.common.exceptions import AgentGenerationError
from app.common.response import ErrorCode
from app.schemas.report import ReportRequest
from app.services.report_service import get_report_service
from app.services.stream_events import done_event, error_event
from app.services.session_service import get_session_service
from app.utils.perf import elapsed_ms, log_perf, now_ms

logger = logging.getLogger("rag-agent.report_api")
router = APIRouter()


@router.post("/stream")
async def report_stream(req: ReportRequest):
    """SSE streaming report generation endpoint.

    Events emitted:
    - event: status     → data: {"phase": "...", "message": "..."}
    - event: tool_start → data: {"tool": "...", "message": "..."}
    - event: tool_done  → data: {"tool": "...", "message": "..."}
    - event: message    → data: {"content": "..."}
    - event: done       → data: {"session_id": "..."}
    - event: error      → data: {"code": "...", "message": "..."}
    """
    report_service = get_report_service()
    session_service = get_session_service()

    session_id = req.session_id or session_service.create_session()

    # Build the natural-language query from request parameters
    if req.user_id and req.month:
        query = f"请为用户{req.user_id}生成{req.month}的扫地机器人使用报告"
    elif req.user_id:
        query = f"请为用户{req.user_id}生成使用报告"
    elif req.month:
        query = f"请生成{req.month}的使用报告"
    else:
        query = "给我生成我的使用报告"

    request_id = req.request_id or session_id
    set_perf_request_id(request_id)

    logger.info(
        "[report/stream] SSE request: session=%s, user_id=%s, month=%s",
        session_id,
        req.user_id,
        req.month,
    )

    sse_start = now_ms()
    log_perf("report_api", "sse_start",
             request_id=request_id,
             session_id=session_id,
             query_len=len(query))

    async def event_generator():
        try:
            async for event in report_service.stream_report(
                query=query,
                session_id=session_id,
                user_id=req.user_id,
                month=req.month,
                device_id=req.device_id,
                request_id=request_id,
            ):
                yield {
                    "event": event["event"],
                    "data": json.dumps(event["data"], ensure_ascii=False),
                }

            log_perf("report_api", "sse_done",
                     request_id=request_id,
                     session_id=session_id,
                     elapsed_ms=elapsed_ms(sse_start))

            logger.info(
                "[report/stream] SSE done: session=%s", session_id
            )

            ev = done_event(session_id)
            yield {
                "event": ev["event"],
                "data": json.dumps(ev["data"], ensure_ascii=False),
            }
        except AgentGenerationError:
            log_perf("report_api", "sse_error",
                     request_id=request_id,
                     session_id=session_id,
                     elapsed_ms=elapsed_ms(sse_start),
                     error="AgentGenerationError")
            logger.error(
                "[report/stream] Report generation failed: session=%s",
                session_id,
            )
            ev = error_event(
                ErrorCode.AGENT_GENERATION_FAILED, "报告生成失败，请稍后重试"
            )
            yield {
                "event": ev["event"],
                "data": json.dumps(ev["data"], ensure_ascii=False),
            }
        except Exception as e:
            log_perf("report_api", "sse_error",
                     request_id=request_id,
                     session_id=session_id,
                     elapsed_ms=elapsed_ms(sse_start),
                     error=str(e)[:80])
            logger.error(
                "[report/stream] Unexpected error: session=%s, %s",
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
