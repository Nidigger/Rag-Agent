"""Report service — two-phase architecture (non-blocking).

Phase 1: ReactAgent.execute() runs tool orchestration in a thread pool
         via asyncio.to_thread() so the event loop stays responsive.
Phase 2: FinalAnswerStreamer streams the report token-by-token.
"""

import asyncio
import logging

from app.agent.react_agent import ReactAgent
from app.common.exceptions import AgentGenerationError
from app.services.final_answer_streamer import get_final_answer_streamer
from app.services.sync_stream import async_wrap_sync_generator

logger = logging.getLogger("rag-agent.report_service")


class ReportService:
    def __init__(self):
        self._agent = ReactAgent()
        self._streamer = get_final_answer_streamer()

    async def stream_report(
        self,
        query: str,
        session_id: str | None = None,
        user_id: str | None = None,
        month: str | None = None,
        device_id: str | None = None,
    ):
        """Two-phase streaming report generation.

        1. Agent phase: execute tools in a thread (non-blocking).
        2. Stream phase: generate structured report token-by-token.
        """
        try:
            context: dict = {"report": True}
            if user_id:
                context["user_id"] = user_id
            if month:
                context["month"] = month
            if device_id:
                context["device_id"] = device_id

            logger.info(
                "[ReportService] Phase 1 — Agent execution: query=%r, "
                "context=%s",
                query[:80],
                context,
            )

            # Phase 1: run Agent in a thread to avoid blocking the event loop
            agent_result = await asyncio.to_thread(
                self._agent.execute,
                query=query,
                context=context,
            )

            logger.info(
                "[ReportService] Phase 2 — Report streaming: "
                "used_tools=%s",
                agent_result.used_tools,
            )

            # Phase 2: Stream final report token-by-token
            sync_gen = self._streamer.stream_final_answer(
                query=query,
                tool_context=agent_result.tool_context,
                final_draft=agent_result.final_draft,
                report=True,
            )

            async for chunk in async_wrap_sync_generator(sync_gen):
                yield chunk

        except AgentGenerationError:
            raise
        except Exception as e:
            logger.error(
                "[ReportService] Report stream error: %s", e, exc_info=True
            )
            raise AgentGenerationError(str(e))


_report_service: ReportService | None = None


def get_report_service() -> ReportService:
    """Get or create the singleton ReportService instance."""
    global _report_service
    if _report_service is None:
        _report_service = ReportService()
    return _report_service
