import logging

from app.agent.react_agent import ReactAgent
from app.core.errors import AgentGenerationError
from app.services.sync_stream import async_wrap_sync_generator

logger = logging.getLogger("rag-agent.chat_service")


class ChatService:
    def __init__(self):
        self._agent = ReactAgent()

    async def stream_chat(
        self, query: str, session_id: str | None = None, messages: list | None = None
    ):
        try:
            sync_gen = self._agent.execute_stream(query, messages=messages)
            async for chunk in async_wrap_sync_generator(sync_gen):
                yield chunk
        except Exception as e:
            logger.error(f"Agent stream error: {e}", exc_info=True)
            raise AgentGenerationError(str(e))


_chat_service: ChatService | None = None


def get_chat_service() -> ChatService:
    global _chat_service
    if _chat_service is None:
        _chat_service = ChatService()
    return _chat_service
