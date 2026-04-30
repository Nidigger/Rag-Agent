"""Report service — two-phase architecture (non-blocking).

Phase 1: ReactAgent.execute() runs tool orchestration in a thread pool
         via asyncio.to_thread() so the event loop stays responsive.
Phase 2: FinalAnswerStreamer streams the report token-by-token.

Yields StreamEvent dicts including status events and tool-level events
(via AgentEventSink) so that the API layer only serialises them to SSE.
"""

import asyncio
import logging

from app.agent.react_agent import ReactAgent
from app.agent.tools.request_context import set_perf_request_id
from app.common.exceptions import AgentGenerationError
from app.schemas.stream import StreamEvent
from app.services.agent_event_sink import AgentEventSink
from app.services.final_answer_streamer import get_final_answer_streamer
from app.services.stream_events import message_event, status_event
from app.services.sync_stream import async_wrap_sync_generator
from app.utils.perf import elapsed_ms, log_perf, now_ms

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
        request_id: str | None = None,
    ):
        """Two-phase streaming report generation.

        Yields StreamEvent dicts:
          1. status(thinking)  — analysing the report request.
          2. status(tool_calling) — agent is gathering data.
          3. tool_start / tool_done — individual tool status (V2).
          4. status(generating) — report streaming begins.
          5. message(token) — content tokens.
        """
        rid = request_id or "internal"
        set_perf_request_id(rid)

        try:
            context: dict = {"report": True, "request_id": rid}
            if user_id:
                context["user_id"] = user_id
            if month:
                context["month"] = month
            if device_id:
                context["device_id"] = device_id

            # --- Phase 1a: Thinking ---
            yield status_event("thinking", "正在分析报告请求")

            # Set up event sink for tool-level events (V2)
            loop = asyncio.get_running_loop()
            tool_queue: asyncio.Queue[StreamEvent | None] = asyncio.Queue()
            event_sink = AgentEventSink(loop, tool_queue)
            context["event_sink"] = event_sink

            logger.info(
                "[ReportService] Phase 1 — Agent execution: query=%r, "
                "context=%s",
                query[:80],
                {k: v for k, v in context.items() if k != "event_sink"},
            )

            # --- Phase 1b: Tool calling ---
            yield status_event("tool_calling", "正在获取报告所需数据")

            phase1_start = now_ms()
            log_perf("report_service", "phase1_start",
                     request_id=rid,
                     session_id=session_id,
                     query_len=len(query))

            # Run agent in a thread; tool middleware emits events via sink
            agent_future = loop.run_in_executor(
                None,
                lambda: self._agent.execute(query=query, context=context),
            )

            # Drain tool events from the queue while the agent is running
            while not agent_future.done():
                try:
                    event = await asyncio.wait_for(tool_queue.get(), timeout=0.5)
                    if event is not None:
                        yield event
                except asyncio.TimeoutError:
                    continue

            # Propagate any exception from the agent thread
            agent_result = agent_future.result()

            # Drain remaining tool events
            while not tool_queue.empty():
                event = tool_queue.get_nowait()
                if event is not None:
                    yield event

            phase1_elapsed = elapsed_ms(phase1_start)
            log_perf("report_service", "phase1_done",
                     request_id=rid,
                     elapsed_ms=phase1_elapsed,
                     used_tools=",".join(agent_result.used_tools) if agent_result.used_tools else "none",
                     unique_tool_count=len(agent_result.used_tools),
                     tool_call_count=agent_result.tool_call_count)

            # --- Phase 2: Stream final report ---
            logger.info(
                "[ReportService] Phase 2 — Report streaming: "
                "used_tools=%s",
                agent_result.used_tools,
            )

            yield status_event("generating", "正在生成报告")

            phase2_start = now_ms()
            log_perf("report_service", "phase2_start",
                     request_id=rid,
                     draft_len=len(agent_result.final_draft),
                     context_len=len(agent_result.tool_context))

            sync_gen = self._streamer.stream_final_answer(
                query=query,
                tool_context=agent_result.tool_context,
                final_draft=agent_result.final_draft,
                report=True,
                request_id=rid,
            )

            output_len = 0
            async for chunk in async_wrap_sync_generator(sync_gen):
                output_len += len(chunk)
                yield message_event(chunk)

            log_perf("report_service", "phase2_done",
                     request_id=rid,
                     elapsed_ms=elapsed_ms(phase2_start),
                     output_len=output_len)

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
