import logging
import uuid
from typing import Any

logger = logging.getLogger("rag-agent.session_service")

_sessions: dict[str, dict[str, Any]] = {}


class SessionService:
    def create_session(self) -> str:
        session_id = str(uuid.uuid4())
        _sessions[session_id] = {"messages": []}
        return session_id

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        return _sessions.get(session_id)

    def add_message(
        self, session_id: str, role: str, content: str
    ) -> None:
        if session_id not in _sessions:
            _sessions[session_id] = {"messages": []}
        _sessions[session_id]["messages"].append(
            {"role": role, "content": content}
        )

    def get_messages(self, session_id: str) -> list[dict]:
        session = _sessions.get(session_id)
        if not session:
            return []
        return session["messages"]


_session_service: SessionService | None = None


def get_session_service() -> SessionService:
    global _session_service
    if _session_service is None:
        _session_service = SessionService()
    return _session_service
