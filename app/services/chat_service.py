"""Chat service — two-phase architecture (non-blocking).

Phase 1: ReactAgent.execute() runs tool orchestration in a thread pool
         via asyncio.to_thread() so the event loop stays responsive.
Phase 2: FinalAnswerStreamer streams the final answer token-by-token.
"""

import asyncio
import logging

from app.agent.react_agent import ReactAgent
from app.core.errors import AgentGenerationError
from app.services.final_answer_streamer import get_final_answer_streamer
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

        1. Agent phase: execute tools in a thread (non-blocking).
        2. Stream phase: generate final answer token-by-token.
        """
        try:
            logger.info(
                "[ChatService] Phase 1 — Agent execution: query=%r",
                query[:80],
            )

            # Phase 1: run Agent in a thread to avoid blocking the event loop
            agent_result = await asyncio.to_thread(
                self._agent.execute,
                query=query,
                messages=messages,
                context={"report": False},
            )

            logger.info(
                "[ChatService] Phase 2 — Final answer streaming: "
                "used_tools=%s",
                agent_result.used_tools,
            )

            # Phase 2: Stream final answer token-by-token
            sync_gen = self._streamer.stream_final_answer(
                query=query,
                tool_context=agent_result.tool_context,
                final_draft=agent_result.final_draft,
                report=False,
            )

            async for chunk in async_wrap_sync_generator(sync_gen):
                yield chunk

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
