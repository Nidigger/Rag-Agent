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
from app.common.exceptions import AgentGenerationError
from app.schemas.stream import StreamEvent
from app.services.agent_event_sink import AgentEventSink
from app.services.final_answer_streamer import get_final_answer_streamer
from app.services.stream_events import message_event, status_event
from app.services.sync_stream import async_wrap_sync_generator

logger = logging.getLogger("rag-agent.chat_service")


class ChatService:
    def __init__(self):
        self._agent = ReactAgent()
        self._streamer = get_final_answer_streamer()

    async def stream_chat(
        self, query: str, session_id: str | None = None, messages: list | None = None
    ):
        """Two-phase streaming chat.

        Yields StreamEvent dicts:
          1. status(thinking) — agent is analysing the user message.
          2. status(generating) — final answer streaming begins.
          3. message(token) — content tokens.
        """
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
            context = {"report": False, "event_sink": event_sink}

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

            # --- Phase 2: Stream final answer ---
            logger.info(
                "[ChatService] Phase 2 — Final answer streaming: "
                "used_tools=%s",
                agent_result.used_tools,
            )

            yield status_event("generating", "正在生成最终回答")

            sync_gen = self._streamer.stream_final_answer(
                query=query,
                tool_context=agent_result.tool_context,
                final_draft=agent_result.final_draft,
                report=False,
            )

            async for chunk in async_wrap_sync_generator(sync_gen):
                yield message_event(chunk)

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
