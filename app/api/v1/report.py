import json
import logging

from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse

from app.core.errors import AgentGenerationError
from app.core.response import ErrorCode
from app.schemas.report import ReportRequest
from app.services.report_service import get_report_service
from app.services.session_service import get_session_service

logger = logging.getLogger("rag-agent.report_api")
router = APIRouter()


@router.post("/stream")
async def report_stream(req: ReportRequest):
    report_service = get_report_service()
    session_service = get_session_service()

    session_id = req.session_id or session_service.create_session()

    if req.user_id and req.month:
        query = f"请为用户{req.user_id}生成{req.month}的扫地机器人使用报告"
    elif req.user_id:
        query = f"请为用户{req.user_id}生成使用报告"
    elif req.month:
        query = f"请生成{req.month}的使用报告"
    else:
        query = "给我生成我的使用报告"

    async def event_generator():
        try:
            async for chunk in report_service.stream_report(
                query=query,
                session_id=session_id,
                user_id=req.user_id,
                month=req.month,
                device_id=req.device_id,
            ):
                yield {
                    "event": "message",
                    "data": json.dumps(
                        {"content": chunk}, ensure_ascii=False
                    ),
                }

            yield {
                "event": "done",
                "data": json.dumps({"session_id": session_id}),
            }
        except AgentGenerationError:
            yield {
                "event": "error",
                "data": json.dumps(
                    {
                        "code": ErrorCode.AGENT_GENERATION_FAILED,
                        "message": "报告生成失败，请稍后重试",
                    },
                    ensure_ascii=False,
                ),
            }
        except Exception as e:
            logger.error(
                f"Report stream error: {e}", exc_info=True
            )
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
