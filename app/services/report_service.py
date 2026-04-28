import logging

from app.agent.react_agent import ReactAgent
from app.core.errors import AgentGenerationError
from app.services.sync_stream import async_wrap_sync_generator

logger = logging.getLogger("rag-agent.report_service")


class ReportService:
    def __init__(self):
        self._agent = ReactAgent()

    async def stream_report(
        self,
        query: str,
        session_id: str | None = None,
        user_id: str | None = None,
        month: str | None = None,
        device_id: str | None = None,
    ):
        try:
            context: dict = {"report": True}
            if user_id:
                context["user_id"] = user_id
            if month:
                context["month"] = month
            if device_id:
                context["device_id"] = device_id
            sync_gen = self._agent.execute_stream(query, context=context)
            async for chunk in async_wrap_sync_generator(sync_gen):
                yield chunk
        except Exception as e:
            logger.error(f"Report stream error: {e}", exc_info=True)
            raise AgentGenerationError(str(e))


_report_service: ReportService | None = None


def get_report_service() -> ReportService:
    global _report_service
    if _report_service is None:
        _report_service = ReportService()
    return _report_service
