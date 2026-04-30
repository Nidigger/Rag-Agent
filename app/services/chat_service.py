"""Chat service — two-phase architecture (non-blocking).

Phase 1: ReactAgent.execute() runs tool orchestration in a thread pool
         via asyncio.to_thread() so the event loop stays responsive.
Phase 2: FinalAnswerStreamer streams the final answer token-by-token.

Yields StreamEvent dicts so that the API layer only needs to serialize
them to SSE — no hand-crafted event structures in the endpoint code.
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

logger = logging.getLogger("rag-agent.chat_service")


class ChatService:
    def __init__(self):
        self._agent = ReactAgent()
        self._streamer = get_final_answer_streamer()

    async def stream_chat(
        self,
        query: str,
        session_id: str | None = None,
        messages: list | None = None,
        request_id: str | None = None,
    ):
        """Two-phase streaming chat.

        Yields StreamEvent dicts:
          1. status(thinking) — agent is analysing the user message.
          2. status(generating) — final answer streaming begins.
          3. message(token) — content tokens.
        """
        rid = request_id or "internal"
        set_perf_request_id(rid)

        try:
            # --- Phase 1: Agent thinking & tool execution ---
            yield status_event("thinking", "正在理解你的问题")

            logger.info(
                "[ChatService] Phase 1 — Agent execution: query=%r",
                query[:80],
            )

            loop = asyncio.get_running_loop()
            tool_queue: asyncio.Queue[StreamEvent | None] = asyncio.Queue()
            event_sink = AgentEventSink(loop, tool_queue)
            context = {
                "report": False,
                "event_sink": event_sink,
                "request_id": rid,
            }

            phase1_start = now_ms()
            log_perf("chat_service", "phase1_start",
                     request_id=rid,
                     session_id=session_id,
                     query_len=len(query))

            agent_future = loop.run_in_executor(
                None,
                lambda: self._agent.execute(
                    query=query,
                    messages=messages,
                    context=context,
                ),
            )

            while not agent_future.done():
                try:
                    event = await asyncio.wait_for(tool_queue.get(), timeout=0.5)
                    if event is not None:
                        yield event
                except asyncio.TimeoutError:
                    continue

            agent_result = agent_future.result()

            while not tool_queue.empty():
                event = tool_queue.get_nowait()
                if event is not None:
                    yield event

            phase1_elapsed = elapsed_ms(phase1_start)
            log_perf("chat_service", "phase1_done",
                     request_id=rid,
                     elapsed_ms=phase1_elapsed,
                     used_tools=",".join(agent_result.used_tools) if agent_result.used_tools else "none",
                     unique_tool_count=len(agent_result.used_tools),
                     tool_call_count=agent_result.tool_call_count)

            # --- Phase 2: Stream final answer ---
            logger.info(
                "[ChatService] Phase 2 — Final answer streaming: "
                "used_tools=%s",
                agent_result.used_tools,
            )

            yield status_event("generating", "正在生成最终回答")

            phase2_start = now_ms()
            log_perf("chat_service", "phase2_start",
                     request_id=rid,
                     draft_len=len(agent_result.final_draft),
                     context_len=len(agent_result.tool_context))

            sync_gen = self._streamer.stream_final_answer(
                query=query,
                tool_context=agent_result.tool_context,
                final_draft=agent_result.final_draft,
                report=False,
                request_id=rid,
            )

            output_len = 0
            async for chunk in async_wrap_sync_generator(sync_gen):
                output_len += len(chunk)
                yield message_event(chunk)

            log_perf("chat_service", "phase2_done",
                     request_id=rid,
                     elapsed_ms=elapsed_ms(phase2_start),
                     output_len=output_len)

        except AgentGenerationError:
            raise
        except Exception as e:
            logger.error(
                "[ChatService] Agent stream error: %s", e, exc_info=True
            )
            raise AgentGenerationError(str(e))


_chat_service: ChatService | None = None


def get_chat_service() -> ChatService:
    """Get or create the singleton ChatService instance."""
    global _chat_service
    if _chat_service is None:
        _chat_service = ChatService()
    return _chat_service
